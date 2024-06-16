import numpy as np
import torch
import torch.nn as nn
import cv2
import torch
from torch import nn
import random
import torch.nn.functional as F
import time
from itertools import combinations
import torchvision
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
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
    
class TemporalModel(nn.Module):
    def __init__(self, input_features, hidden_size, num_layers=1):
        """
        初始化模型。

        参数:
            input_features (int): 每个时间步的特征数量，这里是C*N。
            hidden_size (int): LSTM层的隐藏状态大小。
            num_layers (int): LSTM堆叠的层数。
        """
        super(TemporalModel, self).__init__()
        self.lstm = nn.LSTM(input_features, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x: 输入的特征张量，维度为(B, T, C, N)。

        返回:
            一个形状为(B, hidden_size)的张量，表示每个序列的特征表示。
        """
        B, T, C, N = x.size()
        # 重塑输入以匹配LSTM的期望输入维度(B, T, C*N)
        x = x.reshape(B, T, C*N)
        
        # LSTM处理时间序列
        # 只保留最后一个时间步的隐藏状态
        _, (hidden, _) = self.lstm(x)
        
        # 选择最后一层的最后一个时间步的隐藏状态
        # 维度为(B, hidden_size)
        output = hidden[-1]

        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Register buffer to not be considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)    

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_features, num_heads, num_layers, dim_feedforward, dropout=0.1, max_seq_length=5000):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_features = input_features
        self.dim_feedforward = dim_feedforward
        
        # Embedding layer for input features
        self.input_embedding = nn.Linear(input_features, dim_feedforward)
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(dim_feedforward, dropout, max_seq_length)
        
        # Transformer Encoder Layer
        encoder_layers = TransformerEncoderLayer(d_model=dim_feedforward,
                                                 nhead=num_heads,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers,
                                                      num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(dim_feedforward, input_features)
        
    def forward(self, src):
        # src shape: (B, T, C, N)
        B, T, C, N = src.shape
        src = src.reshape(B, T, C*N)  # Combine C and N into one dimension
        
        # Flatten the batch and feature dimensions
        src = src.reshape(B*T, C*N)  # Now src is (B*T, C*N)
        
        # Apply input embedding
        src = self.input_embedding(src)  # Embedding to dim_feedforward
        
        # Reshape back to (T, B, dim_feedforward) for transformer
        src = src.reshape(T, B, self.dim_feedforward)
        
        # Add positional encoding
        src = self.positional_encoding(src)
        
        # Transformer encoder
        output = self.transformer_encoder(src)
        
        # Output layer and reshape
        output = self.output_layer(output)
        output = output.reshape(B, T, C, N)  # Back to (B, T, C, N)
        
        return output
    
# 随机初始化输入特征张量
input_features = torch.rand(8, 4, 64, 256)  # 对应于(B, T, C, N)形状

# 定义模型参数
input_features_num = 64*256  # C*N
num_heads = 8
num_layers = 4
dim_feedforward = 512

# 初始化模型
model = TimeSeriesTransformer(input_features=(64*256), num_heads=8, num_layers=4, dim_feedforward=512)

# 前向传播
output_features = model(input_features)

print("Output shape:", output_features.shape)