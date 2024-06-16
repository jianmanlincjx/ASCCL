import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
import torch
from torch import nn
import random
import torch.nn.functional as F
from sklearn.manifold import TSNE
import os
from model.base_model import iresnet50, mlp, Normalize, TimeSeriesTransformer


class Spatial_Coherent_Correlation_Learning(nn.Module):
    def __init__(self):
        super(Spatial_Coherent_Correlation_Learning, self).__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss().cuda()
        
        self.resnet50 = iresnet50().cuda()
        self.size = 224
        self.mlp = mlp.cuda()
        self.start_layer = 0
        self.end_layer = 5
        self._img_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((self.size, self.size)),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.Normalize = Normalize(2)

    def img2intermediate(self, input):
        return self.resnet50(input)

    def location2neighborhood(self, location):
        sample_nums = location.shape[0]
        offsets = torch.tensor([[-1, -1], [-1, 0], [-1, 1],
                        [0, -1],           [0, 1],
                        [1, -1], [1, 0], [1, 1]]).reshape(1, 8, 2).repeat(sample_nums, 1, 1)
        neighbors = location.reshape(sample_nums,1, 2).repeat(1, 8, 1) + offsets
        return location, neighbors

    def sample_location(self, feature_map_size, sub_region_size, samples_num):
        # 计算子区域边界大小
        border_size = (feature_map_size - sub_region_size) // 2
        # 生成采样索引
        indices = np.indices((sub_region_size, sub_region_size)).reshape(2, -1).T + border_size
        np.random.shuffle(indices)
        ## torch.Size(num, 2])
        sampled_indices = torch.from_numpy(indices[:samples_num])
        # torch.Size([num, 2])
        # torch.Size([num, 8, 2])
        location, neighborhood = self.location2neighborhood(sampled_indices)
        location = location.reshape(samples_num,1,2).repeat(1,8,1)
        return location.reshape(-1, 2).cuda(), neighborhood.reshape(-1, 2).cuda()
    
    ## PatchNCELoss code from: https://github.com/taesungp/contrastive-unpaired-translation 
    def PatchNCELoss(self, f_q, f_k, weight_pos, tau=0.07):
        # batch size, channel size, and number of sample locations
        B, C, S = f_q.shape
        f_k = f_k.detach()
        # calculate v * v+: BxSx1
        l_pos = (f_k * f_q).sum(dim=1)[:, :, None]
        # calculate v * v-: BxSxS
        l_neg = torch.bmm(f_q.transpose(1, 2), f_k)
        # The diagonal entries are not negatives. Remove them.
        identity_matrix = torch.eye(S,dtype=torch.bool)[None, :, :].to(f_q.device)
        l_neg.masked_fill_(identity_matrix, -float('inf'))
        # calculate logits: (B)x(S)x(S+1)
        logits = torch.cat((l_pos, l_neg), dim=2) / tau
        # return PatchNCE loss
        predictions = logits.flatten(0, 1)
        targets = torch.zeros(B * S, dtype=torch.long).to(f_q.device)
        return torch.mean(self.cross_entropy_loss(predictions, targets) * weight_pos)
        # return torch.mean(self.cross_entropy_loss(predictions, targets))

    def warp_landmarks2img(self, source_landmarks, target_landmarks, source_image, target_image):
        B = source_landmarks.shape[0]
        source_image_result = []
        target_image_result = []
        for b in range(B):
            affine_matrix, _ = cv2.estimateAffine2D(target_landmarks[b], source_landmarks[b])
            output_size = (512, 512)

            target_image_warpAffine = cv2.warpAffine(target_image[b], affine_matrix, output_size)
            binary_matrix = np.where(target_image_warpAffine == 0, 0, 1)
            source_image_mask = source_image[b] * binary_matrix
            source_image_result.append(self._img_transform(source_image_mask.astype(np.float32)))
            target_image_result.append(self._img_transform(target_image_warpAffine.astype(np.float32)))
        s = torch.stack(source_image_result)
        t = torch.stack(target_image_result)
        
        return s.cuda(), t.cuda()
    
    def forward(self, model_input_img, model_out_img, sample_nums=[32, 16, 8, 4], tau=0.07):
        loss_ccp = 0.0
        pos = 0.0
        neg = 0.0
    
        input_feat = self.img2intermediate(model_input_img)
        model_feat = self.img2intermediate(model_out_img)
        # NCE
        for i in range(self.start_layer, self.end_layer-1):
            assert input_feat[i].shape == model_feat[i].shape
            B, C, H, W = input_feat[i].shape
            feat_q = input_feat[i]
            feat_k = model_feat[i]
            location, neighborhood = self.sample_location(H, int(((H//2) + H*0.45)), sample_nums[i])
            
            feat_q_location = feat_q[:, :, location[:,0], location[:,1]]
            feat_q_neighborhood = feat_q[:, :, neighborhood[:,0], neighborhood[:,1]]
            f_q = (feat_q_location - feat_q_neighborhood).permute(0, 2, 1)
            
            ####
            t = torch.nn.functional.sigmoid(torch.abs((feat_q_location - feat_q_neighborhood)))
            adaptive_weight = torch.ones_like(t)
            adaptive_weight[t > 0.8] = 2 * (t[t > 0.8]) ** 2
            ####
            for j in range(3):
                f_q =self.mlp[3*i+j](f_q)
            flow_q = self.Normalize(f_q.permute(0, 2, 1))
     
            feat_k_location = feat_k[:, :, location[:,0], location[:,1]] 
            feat_k_neighborhood = feat_k[:, :, neighborhood[:,0], neighborhood[:,1]] 
            f_k = (feat_k_location - feat_k_neighborhood).permute(0, 2, 1)
            for j in range(3):
                f_k =self.mlp[3*i+j](f_k)
            flow_k = self.Normalize(f_k.permute(0, 2, 1))   

            ## 计算正负样本的相似性
            # 获取最后一个维度的大小
            last_dimension_size = flow_k.size(-1)
            # 生成一个随机的索引排列
            permuted_indices = torch.randperm(last_dimension_size)
            # 使用 permuted_indices 对最后一个维度进行打乱
            shuffled_flow_k = flow_k[..., permuted_indices].detach().cpu()
            cosine_similarity_pos = torch.mean(F.cosine_similarity(flow_q.detach().cpu(), flow_k.detach().cpu(), dim=-1))
            cosine_similarity_neg = torch.mean(F.cosine_similarity(flow_q.detach().cpu(), shuffled_flow_k.detach().cpu(), dim=-1))
            pos += cosine_similarity_pos
            neg += cosine_similarity_neg

            loss_ccp += self.PatchNCELoss(flow_q, flow_k, adaptive_weight, tau)

        return loss_ccp*0.3, pos/4, neg/4
    


def fixed_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark =  True


import torchvision

if __name__ == "__main__":
    fixed_seed()
    Trans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    model = Temporal_Context_Loss()
    model_in = Trans(cv2.imread("/data2/JM/MEAD/M003/align_img/sad/001/000000.jpg")).unsqueeze(0).repeat(2, 1, 1, 1).cuda()
    model_out = Trans(cv2.imread("/data2/JM/MEAD/M003/align_img/sad/001/000010.jpg")).unsqueeze(0).repeat(2, 1, 1, 1).cuda()
    state_dict = torch.load("visual_correlated_modules/model_ckpt/20-512_224.pth")
    model.load_state_dict(state_dict)

    loss = model(model_in, model_out)
    print(loss)


