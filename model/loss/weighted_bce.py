# -*- coding: utf-8 -*-
"""
@Time ： 21-11-24 下午8:19
@Auth ： Nathan
@File ：weighted_bce.py
@WF ： ...
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
EPS = 1e-8
#69808
class Weighted_BCELoss(nn.Module):
    """
        Weighted_BCELoss was proposed in "Multi-attribute learning for pedestrian attribute recognition in surveillance scenarios"[13].
    """
    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(Weighted_BCELoss, self).__init__()
        self.weights = None
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.func = nn.BCEWithLogitsLoss()
        # self.weights = torch.Tensor([30703/69808, 26345/69808, 12760/69808]).cuda()

    def forward(self, output, target):
        # output = torch.sigmoid(output)
        # if self.weights is not None:
        #     cur_weights = torch.exp(target + (1 - target * 2) * self.weights)
        #     loss = cur_weights * (target * torch.log(output + EPS)) + ((1 - target) * torch.log(1 - output + EPS))
        # else:
        #     loss = target * torch.log(output + EPS) + (1 - target) * torch.log(1 - output + EPS)
        # nn.BCEWithLogitsLoss()
        # return torch.neg(torch.mean(loss))
        return self.func(output, target.float())