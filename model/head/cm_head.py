# -*- coding: utf-8 -*-
"""
@Time ： 21-11-27 下午10:52
@Auth ： Nathan
@File ：attn_head.py.py
@WF ： ...
"""
import torch
import torch.nn as nn
import math
class EcaLayer(nn.Module):

    def __init__(self, channels, gamma=2, b=1):
        super(EcaLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = x #self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class CrossModalHead(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CrossModalHead, self).__init__()

        self.attn1 = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.ReLU(inplace=True)
        )
        self.attn2 = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.ReLU(inplace=True)
        )
        self.attn_drop = nn.Dropout(0.25)
        # nn.Sequential(
        #     nn.Linear(channel, int(channel // reduction), bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(int(channel // reduction), channel, bias=False),
        #     nn.Sigmoid()
        # )

    def forward(self, img_glo, text_glo):

        img_glo = img_glo.squeeze(dim=-1).squeeze(dim=-1)
        text_glo = text_glo.squeeze(dim=-1).squeeze(dim=-1)

        # img_glo = self.attn1(img_glo)
        # text_glo = self.attn2(text_glo)

        # if self.training:
        #     img_glo = self.attn_drop(img_glo)
        #     text_glo = self.attn_drop(text_glo)
        # attn = torch.softmax(torch.matmul(text_glo, img_glo), 1)  # 用Transformer里KQV那套范式为例
        # img_glo = torch.matmul(attn, img_glo)
        # print(attn.shape, img_glo.shape)
        # x
        return img_glo, text_glo