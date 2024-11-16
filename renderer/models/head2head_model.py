import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
import renderer.util.util as util
from .base_model import BaseModel
from . import networks
import sys
sys.path.append(os.getcwd())
from visual_correlated_modules.model import Temporal_Context_Loss
import copy

########################
#### Discriminators ####
########################


class Head2HeadModelD(BaseModel):
    def name(self):
        return 'Head2HeadModelD'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.gpu_ids = opt.gpu_ids
        self.output_nc = opt.output_nc
        self.input_nc = opt.input_nc
        # TCCL
        self.TCCL_model = Temporal_Context_Loss().cuda()
        self.TCCL_model.load_state_dict(torch.load("/home/JM/visual_correlated_modules/model_ckpt/90-224_landmarks_align.pth"))
        for param in self.TCCL_model.parameters():
            param.requires_grad = False
        # Image discriminator
        netD_input_nc = self.input_nc + opt.output_nc
        self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm,
                                      opt.num_D, not opt.no_ganFeat, gpu_ids=self.gpu_ids, opt=opt)

        # Mouth, Eyes discriminator
        if not opt.no_mouth_D:
             self.netDm = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm,
                                            opt.num_D, not opt.no_ganFeat, gpu_ids=self.gpu_ids, opt=opt)
        if opt.use_eyes_D:
             self.netDe = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm,
                                            opt.num_D, not opt.no_ganFeat, gpu_ids=self.gpu_ids, opt=opt)

        # load networks
        if (opt.continue_train or opt.load_pretrain):
            self.load_network(self.netD, 'D', opt.which_epoch, opt.load_pretrain)
            if not opt.no_mouth_D:
                self.load_network(self.netDm, 'Dm', opt.which_epoch, opt.load_pretrain)
            if opt.use_eyes_D:
                self.load_network(self.netDe, 'De', opt.which_epoch, opt.load_pretrain)
            print('---------- Discriminators loaded -------------')
        else:
            print('---------- Discriminators initialized -------------')

        # set loss functions and optimizers
        self.old_lr = opt.lr
        # define loss functions
        self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.Tensor)
        self.criterionFeat = torch.nn.L1Loss()
        if not opt.no_vgg:
            self.criterionVGG = networks.VGGLoss(self.gpu_ids[0])

        self.loss_names = ['G_VGG', 'G_GAN', 'G_GAN_Feat', 'D_real', 'D_fake', "tccl_loss_back"]
        if not opt.no_mouth_D:
            self.loss_names += ['Gm_GAN', 'Gm_GAN_Feat', 'Dm_real', 'Dm_fake']
        if opt.use_eyes_D:
            self.loss_names += ['Ge_GAN', 'Ge_GAN_Feat', 'De_real', 'De_fake']

        beta1, beta2 = opt.beta1, 0.999
        lr = opt.lr
        # initialize optimizers
        params = list(self.netD.parameters())
        if not opt.no_mouth_D:
            params += list(self.netDm.parameters())
        if opt.use_eyes_D:
            params += list(self.netDe.parameters())
        self.optimizer_D = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))

    def compute_D_losses(self, netD, real_A, real_B, fake_B):
        # TCCL
        loss_tccl = self.TCCL_model(fake_B, real_B)
        # Input
        real_AB = torch.cat((real_A, real_B), dim=1)
        fake_AB = torch.cat((real_A, fake_B), dim=1)
        # D losses
        pred_real = netD.forward(real_AB)
        pred_fake = netD.forward(fake_AB.detach())
        loss_D_real = self.criterionGAN(pred_real, True, isG=False)
        loss_D_fake = self.criterionGAN(pred_fake, False, isG=False)
        # G losses
        pred_fake = netD.forward(fake_AB)
        loss_G_GAN = self.criterionGAN(pred_fake, True, isG=True)
        loss_G_GAN_Feat = self.FM_loss(pred_real, pred_fake)
        return loss_D_real, loss_D_fake, loss_G_GAN, loss_G_GAN_Feat, loss_tccl

    def FM_loss(self, pred_real, pred_fake):
        if not self.opt.no_ganFeat:
            loss_G_GAN_Feat = 0
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(min(len(pred_fake), self.opt.num_D)):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
        else:
            loss_G_GAN_Feat = torch.zeros(1, 1).cuda()
        return loss_G_GAN_Feat

    def forward(self, tensors_list, mouth_centers=None, eyes_centers=None):
        lambda_feat = self.opt.lambda_feat

        real_B, fake_B, real_A = tensors_list
        _, _, self.height, self.width = real_B.size()
        #################### Losses ####################
        # VGG loss
        loss_G_VGG = (self.criterionVGG(fake_B, real_B) * lambda_feat) if not self.opt.no_vgg else torch.zeros(1, 1).cuda()
        # GAN loss for Generator
        loss_D_real, loss_D_fake, loss_G_GAN, loss_G_GAN_Feat, tccl_loss = self.compute_D_losses(self.netD, real_A, real_B, fake_B)
        tccl_loss_back = tccl_loss[0]
        loss_list = [loss_G_VGG, loss_G_GAN, loss_G_GAN_Feat, loss_D_real, loss_D_fake, tccl_loss_back]

        if not self.opt.no_mouth_D:
            # Extract mouth region around the center.
            real_A_mouth, real_B_mouth, fake_B_mouth = util.get_ROI([real_A, real_B, fake_B], mouth_centers, self.opt)
            # Losses for mouth discriminator
            loss_Dm_real, loss_Dm_fake, loss_Gm_GAN, loss_Gm_GAN_Feat, _ = self.compute_D_losses(self.netDm, real_A_mouth, real_B_mouth, fake_B_mouth)
            mouth_weight = 1
            loss_Gm_GAN *= mouth_weight
            loss_Gm_GAN_Feat *= mouth_weight
            loss_list += [loss_Gm_GAN, loss_Gm_GAN_Feat, loss_Dm_real, loss_Dm_fake]
        if self.opt.use_eyes_D:
            # Extract eyes region around the center.
            real_A_eyes, real_B_eyes, fake_B_eyes = util.get_ROI([real_A, real_B, fake_B], eyes_centers, self.opt)
            # Losses for eyes discriminator
            loss_De_real, loss_De_fake, loss_Ge_GAN, loss_Ge_GAN_Feat, _ = self.compute_D_losses(self.netDe, real_A_eyes, real_B_eyes, fake_B_eyes)
            eyes_weight = 1
            loss_Ge_GAN *= eyes_weight
            loss_Ge_GAN_Feat *= eyes_weight
            loss_list += [loss_Ge_GAN, loss_Ge_GAN_Feat, loss_De_real, loss_De_fake]

        loss_list = [loss.unsqueeze(0) for loss in loss_list]
        return loss_list, tccl_loss

    def save(self, label):
        self.save_network(self.netD, 'D', label, self.gpu_ids)
        if not self.opt.no_mouth_D:
            self.save_network(self.netDm, 'Dm', label, self.gpu_ids)
        if self.opt.use_eyes_D:
            self.save_network(self.netDe, 'De', label, self.gpu_ids)

    def update_learning_rate(self, epoch):
        lr = self.opt.lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

##########################
#### Generator  model ####
##########################

class Head2HeadModelG(BaseModel):
    def name(self):
        return 'Head2HeadModelG'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.n_frames_G = opt.n_frames_G
        input_nc = opt.input_nc
        netG_input_nc = input_nc * self.n_frames_G
        prev_output_nc = (self.n_frames_G - 1) * opt.output_nc

        self.netG = networks.define_G(netG_input_nc, opt.output_nc, prev_output_nc, opt.ngf, opt.n_downsample_G, opt.norm, self.gpu_ids, opt)

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            self.load_network(self.netG, 'G', opt.which_epoch, opt.load_pretrain)
            print('---------- Generator loaded -------------')
        else:
            print('---------- Generator initialized -------------')

        # Optimizer for G
        if self.isTrain:
            self.old_lr = opt.lr
            self.n_frames_backpropagate = self.opt.n_frames_backpropagate
            self.n_frames_load = min(self.opt.max_frames_per_gpu, self.opt.n_frames_total)
            # initialize optimizer G
            params = list(self.netG.parameters())
            beta1, beta2 = opt.beta1, 0.999
            lr = opt.lr
            self.optimizer_G = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))

    def encode_input(self, input_map, real_image):
        size = input_map.size()
        self.bs, _, self.height, self.width = size[0], size[1], size[3], size[4]
        input_map = input_map.data.cuda()
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())
        return input_map, real_image

    def forward(self, input_A, input_B, fake_B_prev):
        # Feed forward for training
        real_A, real_B = self.encode_input(input_A, input_B)
        gpu_id = real_A.get_device()

        is_first_frame = fake_B_prev is None
        if is_first_frame:
            if self.opt.no_first_img:
                fake_B_prev = Variable(self.Tensor(self.bs, self.n_frames_G-1, self.opt.output_nc, self.height, self.width).zero_())
            else:
                fake_B_prev = real_B[:,:self.n_frames_G-1,...]

        ### generate frames sequentially
        for t in range(self.n_frames_load):
            _, _, _, h, w = real_A.size()
            real_A_reshaped = real_A[:, t:t+self.n_frames_G,...].view(self.bs, -1, h, w).cuda(gpu_id)

            fake_B_prevs = fake_B_prev[:, t:t+self.n_frames_G-1,...].cuda(gpu_id)
            if (t % self.n_frames_backpropagate) == 0:
                fake_B_prevs = fake_B_prevs.detach()
            fake_B_prevs_reshaped = fake_B_prevs.view(self.bs, -1, h, w)

            fake_B = self.netG.forward(real_A_reshaped, fake_B_prevs_reshaped)

            fake_B_prev = self.concatenate_tensors([fake_B_prev, fake_B.unsqueeze(1).cuda(gpu_id)], dim=1)
        # fake_B, real_A, real_Bp, fake_B_last
        fake_B = fake_B_prev[:, self.n_frames_G-1:]
        fake_B_prev = fake_B_prev[:, -self.n_frames_G+1:].detach()
        return fake_B, real_A[:,self.n_frames_G-1:], real_B[:,self.n_frames_G-2:], fake_B_prev

    def inference(self, input_A, input_B):
        # Feed forward for test
        with torch.no_grad():
            real_A, real_B = self.encode_input(input_A, input_B)
            self.is_first_frame = not hasattr(self, 'fake_B_prev') or self.fake_B_prev is None
            if self.is_first_frame:
                if self.opt.no_first_img:
                    fake_B_prev = Variable(self.Tensor(self.bs, self.n_frames_G-1, self.opt.output_nc, self.height, self.width).zero_())
                else:
                    fake_B_prev = real_B[:,:self.n_frames_G-1,...]
                self.fake_B_prev = fake_B_prev[0]

            _, _, _, h, w = real_A.size()

            real_As_reshaped = real_A[0,:self.n_frames_G].view(1, -1, h, w)
            fake_B_prevs_reshaped = self.fake_B_prev.view(1, -1, h, w)

            fake_B = self.netG.forward(real_As_reshaped, fake_B_prevs_reshaped)
            self.fake_B_prev = torch.cat([self.fake_B_prev[1:,...], fake_B])
        return fake_B

    def concatenate_tensors(self, tensors, dim=0):
        if tensors[0] is not None and tensors[1] is not None:
            if isinstance(tensors[0], list):
                tensors_cat = []
                for i in range(len(tensors[0])):
                    tensors_cat.append(self.concatenate_tensors([tensors[0][i], tensors[1][i]], dim=dim))
                return tensors_cat
            return torch.cat([tensors[0], tensors[1]], dim=dim)
        elif tensors[0] is not None:
            return tensors[0]
        else:
            return tensors[1]

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)

    def update_learning_rate(self, epoch):
        lr = self.opt.lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


def create_model(opt):
    modelG = Head2HeadModelG()
    modelG.initialize(opt)
    if opt.isTrain and len(opt.gpu_ids):
        modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids)
        modelD = Head2HeadModelD()
        modelD.initialize(opt)
        modelD = nn.DataParallel(modelD, device_ids=opt.gpu_ids)
        return [modelG, modelD]
    else:
        return modelG

## TCCL
# class Head2HeadModelD(BaseModel):
#     def name(self):
#         return 'Head2HeadModelD'

#     def initialize(self, opt):
#         BaseModel.initialize(self, opt)
#         self.gpu_ids = opt.gpu_ids
#         self.output_nc = opt.output_nc
#         self.input_nc = opt.input_nc

#         self.TCloss = Temporal_Context_Loss().cuda()

#         # Image discriminator
#         netD_input_nc = self.input_nc + opt.output_nc
#         self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm,
#                                       opt.num_D, not opt.no_ganFeat, gpu_ids=self.gpu_ids, opt=opt)

#         # Mouth, Eyes discriminator
#         if not opt.no_mouth_D:
#              self.netDm = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm,
#                                             opt.num_D, not opt.no_ganFeat, gpu_ids=self.gpu_ids, opt=opt)
#         if opt.use_eyes_D:
#              self.netDe = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm,
#                                             opt.num_D, not opt.no_ganFeat, gpu_ids=self.gpu_ids, opt=opt)

#         # load networks
#         if (opt.continue_train or opt.load_pretrain):
#             self.load_network(self.netD, 'D', opt.which_epoch, opt.load_pretrain)
#             if not opt.no_mouth_D:
#                 self.load_network(self.netDm, 'Dm', opt.which_epoch, opt.load_pretrain)
#             if opt.use_eyes_D:
#                 self.load_network(self.netDe, 'De', opt.which_epoch, opt.load_pretrain)
#             print('---------- Discriminators loaded -------------')
#         else:
#             print('---------- Discriminators initialized -------------')

#         # set loss functions and optimizers
#         self.old_lr = opt.lr
#         # define loss functions
#         self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.Tensor)
#         self.criterionFeat = torch.nn.L1Loss()
#         if not opt.no_vgg:
#             self.criterionVGG = networks.VGGLoss(self.gpu_ids[0])

#         self.loss_names = ['G_VGG', 'G_GAN', 'G_GAN_Feat', 'D_real', 'D_fake', 'TCCL_G']
#         # self.loss_names = ['G_VGG', 'G_GAN', 'G_GAN_Feat', 'D_real', 'D_fake']
#         if not opt.no_mouth_D:
#             # self.loss_names += ['Gm_GAN', 'Gm_GAN_Feat', 'Dm_real', 'Dm_fake', 'TCCL_mouth']
#             self.loss_names += ['Gm_GAN', 'Gm_GAN_Feat', 'Dm_real', 'Dm_fake']
#         if opt.use_eyes_D:
#             self.loss_names += ['Ge_GAN', 'Ge_GAN_Feat', 'De_real', 'De_fake']
      

#         beta1, beta2 = opt.beta1, 0.999
#         lr = opt.lr
#         # initialize optimizers
#         params = list(self.netD.parameters())
#         if not opt.no_mouth_D:
#             params += list(self.netDm.parameters())
#         if opt.use_eyes_D:
#             params += list(self.netDe.parameters())
#         self.optimizer_D = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))

#     def compute_D_losses(self, netD, real_A, real_B, fake_B):
#         # Input
#         # real_B torch.Size([4, 3, 256, 256])
#         # fake_B torch.Size([4, 3, 256, 256])

#         loss_TCCL = self.TCloss(real_B, fake_B)

#         real_AB = torch.cat((real_A, real_B), dim=1)
#         fake_AB = torch.cat((real_A, fake_B), dim=1)
#         # D losses
#         pred_real = netD.forward(real_AB)
#         pred_fake = netD.forward(fake_AB.detach())
#         loss_D_real = self.criterionGAN(pred_real, True, isG=False)
#         loss_D_fake = self.criterionGAN(pred_fake, False, isG=False)
#         # G losses
#         pred_fake = netD.forward(fake_AB)
#         loss_G_GAN = self.criterionGAN(pred_fake, True, isG=True)
#         loss_G_GAN_Feat = self.FM_loss(pred_real, pred_fake)
#         return loss_D_real, loss_D_fake, loss_G_GAN, loss_G_GAN_Feat, loss_TCCL
#         # return loss_D_real, loss_D_fake, loss_G_GAN, loss_G_GAN_Feat    


#     def FM_loss(self, pred_real, pred_fake):
#         if not self.opt.no_ganFeat:
#             loss_G_GAN_Feat = 0
#             feat_weights = 4.0 / (self.opt.n_layers_D + 1)
#             D_weights = 1.0 / self.opt.num_D
#             for i in range(min(len(pred_fake), self.opt.num_D)):
#                 for j in range(len(pred_fake[i])-1):
#                     loss_G_GAN_Feat += D_weights * feat_weights * \
#                         self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
#         else:
#             loss_G_GAN_Feat = torch.zeros(1, 1).cuda()
#         return loss_G_GAN_Feat

#     def forward(self, tensors_list, mouth_centers=None, eyes_centers=None):
#         lambda_feat = self.opt.lambda_feat

#         real_B, fake_B, real_A = tensors_list
#         _, _, self.height, self.width = real_B.size()
#         #################### Losses ####################
#         # VGG loss
#         loss_G_VGG = (self.criterionVGG(fake_B, real_B) * lambda_feat) if not self.opt.no_vgg else torch.zeros(1, 1).cuda()
#         # GAN loss for Generator
#         loss_D_real, loss_D_fake, loss_G_GAN, loss_G_GAN_Feat, loss_TCCL_G = self.compute_D_losses(self.netD, real_A, real_B, fake_B)
#         # loss_D_real, loss_D_fake, loss_G_GAN, loss_G_GAN_Feat = self.compute_D_losses(self.netD, real_A, real_B, fake_B)
#         loss_list = [loss_G_VGG, loss_G_GAN, loss_G_GAN_Feat, loss_D_real, loss_D_fake, loss_TCCL_G*0.15]
#         # loss_list = [loss_G_VGG, loss_G_GAN, loss_G_GAN_Feat, loss_D_real, loss_D_fake]

#         if not self.opt.no_mouth_D:
#             # Extract mouth region around the center.
#             real_A_mouth, real_B_mouth, fake_B_mouth = util.get_ROI([real_A, real_B, fake_B], mouth_centers, self.opt)
#             # Losses for mouth discriminator
#             # loss_Dm_real, loss_Dm_fake, loss_Gm_GAN, loss_Gm_GAN_Feat, loss_TCCL_mouth = self.compute_D_losses(self.netDm, real_A_mouth, real_B_mouth, fake_B_mouth)
#             loss_Dm_real, loss_Dm_fake, loss_Gm_GAN, loss_Gm_GAN_Feat, _ = self.compute_D_losses(self.netDm, real_A_mouth, real_B_mouth, fake_B_mouth)
#             mouth_weight = 1
#             loss_Gm_GAN *= mouth_weight
#             loss_Gm_GAN_Feat *= mouth_weight
#             # loss_TCCL_mouth *= 0.15
#             # loss_list += [loss_Gm_GAN, loss_Gm_GAN_Feat, loss_Dm_real, loss_Dm_fake, loss_TCCL_mouth]
#             loss_list += [loss_Gm_GAN, loss_Gm_GAN_Feat, loss_Dm_real, loss_Dm_fake]
#         if self.opt.use_eyes_D:
#             # Extract eyes region around the center.
#             real_A_eyes, real_B_eyes, fake_B_eyes = util.get_ROI([real_A, real_B, fake_B], eyes_centers, self.opt)
#             # Losses for eyes discriminator
#             loss_De_real, loss_De_fake, loss_Ge_GAN, loss_Ge_GAN_Feat, _ = self.compute_D_losses(self.netDe, real_A_eyes, real_B_eyes, fake_B_eyes)
#             eyes_weight = 1
#             loss_Ge_GAN *= eyes_weight
#             loss_Ge_GAN_Feat *= eyes_weight
#             loss_list += [loss_Ge_GAN, loss_Ge_GAN_Feat, loss_De_real, loss_De_fake]

#         loss_list = [loss.unsqueeze(0) for loss in loss_list]
#         return loss_list

#     def save(self, label):
#         self.save_network(self.netD, 'D', label, self.gpu_ids)
#         if not self.opt.no_mouth_D:
#             self.save_network(self.netDm, 'Dm', label, self.gpu_ids)
#         if self.opt.use_eyes_D:
#             self.save_network(self.netDe, 'De', label, self.gpu_ids)

#     def update_learning_rate(self, epoch):
#         lr = self.opt.lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
#         for param_group in self.optimizer_D.param_groups:
#             param_group['lr'] = lr
#         print('update learning rate: %f -> %f' % (self.old_lr, lr))
#         self.old_lr = lr

# ##########################
# #### Generator  model ####
# ##########################

# class Head2HeadModelG(BaseModel):
#     def name(self):
#         return 'Head2HeadModelG'

#     def initialize(self, opt):
#         BaseModel.initialize(self, opt)
#         self.isTrain = opt.isTrain
#         self.n_frames_G = opt.n_frames_G
#         input_nc = opt.input_nc
#         netG_input_nc = input_nc * self.n_frames_G
#         prev_output_nc = (self.n_frames_G - 1) * opt.output_nc

#         self.netG = networks.define_G(netG_input_nc, opt.output_nc, prev_output_nc, opt.ngf, opt.n_downsample_G, opt.norm, self.gpu_ids, opt)

#         # load networks
#         if not self.isTrain or opt.continue_train or opt.load_pretrain:
#             self.load_network(self.netG, 'G', opt.which_epoch, opt.load_pretrain)
#             print('---------- Generator loaded -------------')
#         else:
#             print('---------- Generator initialized -------------')

#         # Optimizer for G
#         if self.isTrain:
#             self.old_lr = opt.lr
#             self.n_frames_backpropagate = self.opt.n_frames_backpropagate
#             self.n_frames_load = min(self.opt.max_frames_per_gpu, self.opt.n_frames_total)
#             # initialize optimizer G
#             params = list(self.netG.parameters())
#             beta1, beta2 = opt.beta1, 0.999
#             lr = opt.lr
#             self.optimizer_G = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))

#     def encode_input(self, input_map, real_image):
#         size = input_map.size()
#         self.bs, _, self.height, self.width = size[0], size[1], size[3], size[4]
#         input_map = input_map.data.cuda()
#         if real_image is not None:
#             real_image = Variable(real_image.data.cuda())
#         return input_map, real_image

#     def forward(self, input_A, input_B, fake_B_prev):
#         # Feed forward for training
#         real_A, real_B = self.encode_input(input_A, input_B)
#         gpu_id = real_A.get_device()

#         is_first_frame = fake_B_prev is None
#         if is_first_frame:
#             if self.opt.no_first_img:
#                 fake_B_prev = Variable(self.Tensor(self.bs, self.n_frames_G-1, self.opt.output_nc, self.height, self.width).zero_())
#             else:
#                 fake_B_prev = real_B[:,:self.n_frames_G-1,...]

#         ### generate frames sequentially
#         for t in range(self.n_frames_load):
#             _, _, _, h, w = real_A.size()
#             real_A_reshaped = real_A[:, t:t+self.n_frames_G,...].view(self.bs, -1, h, w).cuda(gpu_id)

#             fake_B_prevs = fake_B_prev[:, t:t+self.n_frames_G-1,...].cuda(gpu_id)
#             if (t % self.n_frames_backpropagate) == 0:
#                 fake_B_prevs = fake_B_prevs.detach()
#             fake_B_prevs_reshaped = fake_B_prevs.view(self.bs, -1, h, w)

#             fake_B = self.netG.forward(real_A_reshaped, fake_B_prevs_reshaped)

#             fake_B_prev = self.concatenate_tensors([fake_B_prev, fake_B.unsqueeze(1).cuda(gpu_id)], dim=1)

#         fake_B = fake_B_prev[:, self.n_frames_G-1:]
#         fake_B_prev = fake_B_prev[:, -self.n_frames_G+1:].detach()
#         return fake_B, real_A[:,self.n_frames_G-1:], real_B[:,self.n_frames_G-2:], fake_B_prev

#     def inference(self, input_A, input_B):
#         # Feed forward for test
#         with torch.no_grad():
#             real_A, real_B = self.encode_input(input_A, input_B)
#             self.is_first_frame = not hasattr(self, 'fake_B_prev') or self.fake_B_prev is None
#             if self.is_first_frame:
#                 if self.opt.no_first_img:
#                     fake_B_prev = Variable(self.Tensor(self.bs, self.n_frames_G-1, self.opt.output_nc, self.height, self.width).zero_())
#                 else:
#                     fake_B_prev = real_B[:,:self.n_frames_G-1,...]
#                 self.fake_B_prev = fake_B_prev[0]

#             _, _, _, h, w = real_A.size()

#             real_As_reshaped = real_A[0,:self.n_frames_G].view(1, -1, h, w)
#             fake_B_prevs_reshaped = self.fake_B_prev.view(1, -1, h, w)

#             fake_B = self.netG.forward(real_As_reshaped, fake_B_prevs_reshaped)
#             self.fake_B_prev = torch.cat([self.fake_B_prev[1:,...], fake_B])
#         return fake_B

#     def concatenate_tensors(self, tensors, dim=0):
#         if tensors[0] is not None and tensors[1] is not None:
#             if isinstance(tensors[0], list):
#                 tensors_cat = []
#                 for i in range(len(tensors[0])):
#                     tensors_cat.append(self.concatenate_tensors([tensors[0][i], tensors[1][i]], dim=dim))
#                 return tensors_cat
#             return torch.cat([tensors[0], tensors[1]], dim=dim)
#         elif tensors[0] is not None:
#             return tensors[0]
#         else:
#             return tensors[1]

#     def save(self, label):
#         self.save_network(self.netG, 'G', label, self.gpu_ids)

#     def update_learning_rate(self, epoch):
#         lr = self.opt.lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
#         for param_group in self.optimizer_G.param_groups:
#             param_group['lr'] = lr
#         print('update learning rate: %f -> %f' % (self.old_lr, lr))
#         self.old_lr = lr


# def create_model(opt):
#     modelG = Head2HeadModelG()
#     modelG.initialize(opt)
#     if opt.isTrain and len(opt.gpu_ids):
#         modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids)
#         modelD = Head2HeadModelD()
#         modelD.initialize(opt)
#         modelD = nn.DataParallel(modelD, device_ids=opt.gpu_ids)
#         return [modelG, modelD]
#     else:
#         return modelG

