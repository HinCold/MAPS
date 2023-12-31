# -*- coding: utf-8 -*-
"""
@Time ： 21-12-3 下午2:38
@Auth ： Nathan
@File ：asmr.py
@WF ： ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
class ASMR(nn.Module):
    def __init__(self, s=50.0, m=0.30, class_num=11003, in_features=2048,bias=False):
        super(ASMR, self).__init__()

        self.epsilon = 1e-8
        self.criterion = nn.MSELoss() # size_average = True

    def forward(self, input_img, input_text, caption, labels):
        caption = caption.float()
        cosine_text = F.linear(F.normalize(input_text), F.normalize(input_text))
        # print('cosine_text', cosine_text)
        mean_sim = torch.mean(cosine_text)
        # print('shape ', caption.shape, input_text.shape, caption.shape)
        dist = torch.zeros((caption.shape[0], caption.shape[0]))
        # print('--------------------')
        # way 1:use two loops
        # for i in range(caption.shape[0]):
        #     for j in range(caption.shape[0]):
        #         dist[i, j] = torch.sum((caption[i, :] - caption[j, :]) ** 2)
        for i in range(caption.shape[0]):
            dist[i, :] = torch.sum((caption - caption[i, :]) ** 2, axis=1)
        dist = dist.to('cuda')
        #cosine_img = F.linear(caption, caption)
        cosine_img = torch.sigmoid(dist)
        # print('cosine_caption+mean_sim', cosine_img+mean_sim)
        # C
        loss = self.criterion(cosine_text, cosine_img+mean_sim)
        return loss
