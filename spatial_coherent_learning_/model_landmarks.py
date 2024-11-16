import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
import sys
import torch
from torch import nn
import random
import torch.nn.functional as F
from sklearn.manifold import TSNE
import os
import time

__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    fc_scale = 32 * 32
    def __init__(self,
                 block, layers, dropout=0, num_features=7, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x, return_features=True):
        out = []
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            out.append(x)
            x = self.layer2(x)
            out.append(x)
            x = self.layer3(x)
            out.append(x)
            x = self.layer4(x)
            out.append(x)
        if return_features:
            out.append(x)
            return out
        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)





mlp = nn.ModuleList([nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 16),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 32),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 128)]) 


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out



class Temporal_Context_Loss(nn.Module):
    def __init__(self):
        super(Temporal_Context_Loss, self).__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        
        self.resnet50 = iresnet50()
        self.size = 224
        self.mlp = mlp
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
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
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


