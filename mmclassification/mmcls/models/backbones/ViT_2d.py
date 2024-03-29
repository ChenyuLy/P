import torch.nn as nn
import torch
from functools import partial

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import BaseModule, ModuleList

from mmcls.utils import get_root_logger
from ..builder import BACKBONES
from ..utils import MultiheadAttention, resize_pos_embed, to_2tuple
from .base_backbone import BaseBackbone


class Depthwise_conv(BaseModule):
    "Depthwise conv + Pointwise conv"

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, H, W):
        super(Depthwise_conv, self).__init__()
        # self.conv = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=1, stride=1, padding=0)
        # self.conv1 = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=kernel_size, stride=stride,
        #                        padding=padding,
        #                        groups=in_channels, bias=False)
        # self.bn1 = nn.BatchNorm2d(in_channels * 2)
        # self.conv2 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_channels * 2)

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding,
                               groups=in_channels, bias=False)
        self.conv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, stride=1,
                              padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.extend = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.ln1 = nn.LayerNorm([in_channels, int(H), int(W)])

    def forward(self, x):
        t = x
        # 使用ConvNeXt结构

        x = self.conv1(x)
        x = self.ln1(x)
        x = self.conv(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + self.extend(t)

        # x = self.conv(x)
        # x = self.bn1(x)
        # x = F.gelu(x)
        # x = self.conv1(x)
        # x = self.bn2(x)
        # x = F.gelu(x)
        # x = self.conv2(x)
        # x = x + t
        # return x


class MutiHeadAttentnion_2d(BaseModule):
    def __init__(self, in_dim, out_dim, H, W, head_num=12, padding=(3, 3), stride=1, kernel_size=(7, 7),
                 norm=nn.LayerNorm):
        super(MutiHeadAttentnion_2d, self).__init__()
        self.heads_num = head_num
        self.heads_dim = in_dim // head_num
        self.qk_scale = self.heads_dim ** -0.5
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.padding = padding
        self.stride = stride
        self.kernel = kernel_size

        # self.proj_q = nn.Conv2d(self.in_dim, self.out_dim, self.kernel, self.stride, self.padding)
        # self.proj_k = nn.Conv2d(self.in_dim, self.out_dim, self.kernel, self.stride, self.padding)
        # self.proj_v = nn.Conv2d(self.in_dim, self.out_dim, self.kernel, self.stride, self.padding)

        # self.proj_q = Depthwise_conv(self.in_dim, self.out_dim, self.kernel, self.stride, self.padding)
        # self.proj_k = Depthwise_conv(self.in_dim, self.out_dim, self.kernel, self.stride, self.padding)
        # self.proj_v = Depthwise_conv(self.in_dim, self.out_dim, self.kernel, self.stride, self.padding)

        self.proj_qkv = Depthwise_conv(self.in_dim, self.out_dim * 3, self.kernel, self.stride, self.padding, H, W)

        self.softmax2d = nn.Softmax2d()
        self.proj = nn.Linear(out_dim, out_dim)
        self.norm = norm(in_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.transpose(1, 3)
        x = self.norm(x)
        x = x.transpose(1, 3)
        # qkv [B,heads_num,heads_dim,h,w]

        # q = self.proj_q(x).reshape(B, self.heads_num, self.heads_dim, H, W)
        # k = self.proj_k(x).reshape(B, self.heads_num, self.heads_dim, H, W)
        # v = self.proj_v(x).reshape(B, self.heads_num, self.heads_dim, H, W)

        qkv = self.proj_qkv(x).reshape(3, B, self.heads_num, self.heads_dim, H, W)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # atten_2d = q @k.transpose(-2, -1) * self.qk_scale
        atten_2d = torch.matmul(q, k.transpose(-2, -1)) * self.qk_scale
        atten_2d = atten_2d.reshape(B * self.heads_num, self.heads_dim, H, W)
        atten_2d = self.softmax2d(atten_2d)
        atten_2d = atten_2d.reshape(B, self.heads_num, self.heads_dim, H, W)
        x = torch.matmul(atten_2d, v).reshape(B, -1, H, W)
        x = (self.proj(x.transpose(1, 3))).transpose(1, 3)
        return x  # B,C,W,H


class MLP(BaseModule):
    def __init__(self, in_feature, out_feature=None, hidden_feature=None, nonLinear_func=nn.ReLU, drop=0.,
                 norm=nn.LayerNorm):
        super(MLP, self).__init__()
        self.drop = drop
        self.in_feature = in_feature
        self.out_feature = in_feature or out_feature
        self.hidden_feature = hidden_feature
        self.fc1 = nn.Linear(self.in_feature, self.hidden_feature)
        self.activation = nonLinear_func()
        self.fc2 = nn.Linear(self.hidden_feature, self.in_feature)
        self.drop = nn.Dropout(drop)
        self.norm = norm(in_feature)

    def forward(self, x):
        x = x.transpose(1, 3)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.transpose(1, 3)
        return x


class Block(BaseModule):
    def __init__(self,
                 in_dim, out_dim, H, W, kernel_size=7, padding=(3, 3), stride=1, head_num=12,
                 mlp_hidden_ratio=4,
                 drop=0.,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.padd = padding
        self.head_num = head_num
        self.stride = stride
        self.kernel = kernel_size

        self.norm1 = norm_layer(in_dim)
        self.num_head = head_num
        self.att_dim = in_dim
        self.after_att_dim = out_dim
        # self.attn = MutiHeadAttentnion_2d(padding=padding, stride=stride, head_num=head_num, in_dim=self.att_dim,
        #                                   out_dim=self.after_att_dim, kernel_size=kernel_size)
        self.attn = MutiHeadAttentnion_2d(head_num=self.head_num, in_dim=self.att_dim, out_dim=self.after_att_dim,
                                          padding=self.padd,
                                          stride=self.stride, H=H, W=W)

        self.mlp = MLP(in_feature=self.after_att_dim, hidden_feature=self.after_att_dim * mlp_hidden_ratio)
        self.norm2 = norm_layer(self.after_att_dim)

    def forward(self, x):
        x = x + self.attn((self.norm1(x.transpose(1, 3))).transpose(1, 3))
        x = x + self.mlp((self.norm2(x.transpose(1, 3))).transpose(1, 3))
        return x  # B,C,W,H


class DownSample2x(BaseModule):
    def __init__(self, in_channel, out_channel):
        super(DownSample2x, self).__init__()
        self.downsample = nn.Conv2d(in_channel, out_channel, stride=2, padding=1, kernel_size=3)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.downsample(x)
        x = self.activation(x)
        return x


class Stem(BaseModule):
    def __init__(self, kernelsize=4, strde=4, out_dim=96):
        super(Stem, self).__init__()
        self.proj1 = nn.Conv2d(3, out_channels=int(out_dim), stride=(strde, strde),
                               kernel_size=(kernelsize, kernelsize))
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.proj1(x)
        x = self.activation(x)
        return x


class Stage(BaseModule):
    def __init__(self, cin_dim, featuresize, block_num=3, dim_hidden_rate=4, down_sample_rate=2, is_last_stage=False,
                 num_heads=12):
        super(Stage, self).__init__()
        self.Blocks = nn.Sequential(
            *[Block(in_dim=cin_dim, out_dim=cin_dim, mlp_hidden_ratio=dim_hidden_rate, head_num=num_heads,
                    H=featuresize, W=featuresize
                    ) for i in range(block_num)])
        self.downsample = DownSample2x(in_channel=cin_dim, out_channel=cin_dim * down_sample_rate)
        self.last_stage = is_last_stage

    def forward(self, x):
        x = self.Blocks(x)
        if self.last_stage is not True:
            x = self.downsample(x)
        return x


# class cls_head(BaseBackbone):
#     def __init__(self, class_num, feature_num=768):
#         super(cls_head, self).__init__()
#         self.class_num = class_num
#         self.feature_num = feature_num
#         self.fc = nn.Linear(feature_num, class_num)
#         self.GAP = nn.AdaptiveAvgPool2d((1, 1))
#
#     def forward(self, x):
#         x = x.transpose(1, 3)
#         x = self.fc(x)
#         x = x.transpose(1, 3)
#         x = self.GAP(x)
#         x = torch.squeeze(x)
#         # x = x.max(1)[0]
#         return x

@BACKBONES.register_module()
class vision_2dtransformer(BaseBackbone):
    def __init__(self, stem_dim=96, stage_num=4, num_heads=12, block_per_stage=[2, 2, 6, 2],
                 dim_change_rate=[1, 2, 4, 8],
                 state=[False, False, False, True], picturesize=224):
        super(vision_2dtransformer, self).__init__()
        self.picsize = picturesize
        self.STEM = Stem(out_dim=stem_dim)
        self.stages = nn.Sequential(
            *[Stage(block_num=i, cin_dim=j * stem_dim, is_last_stage=k, num_heads=num_heads,
                    featuresize=self.picsize/4 / j) for i, j, k in#stem 下采样四倍
              zip(block_per_stage, dim_change_rate, state)])
        # self.head = cls_head(class_num=class_num)

    def forward(self, x):
        # x = self.head(self.stages(self.STEM(x)))
        x = self.stages(self.STEM(x))
        return x

#
# def vit2d_base_patch16_224(num_classes: int = 1000):
#     """
#     ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
#     weights ported from official Google JAX impl:
#     链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
#     """
#     model = vision_2dtransformer(
#         class_num=num_classes)
#     return model
