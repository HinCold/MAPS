# -*- coding: utf-8 -*-
"""
@Time ： 21-10-12 上午9:39
@Auth ： Nathan
@File ：id_loss.py
@WF ： ...
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class Id_Loss(nn.Module):

    def __init__(self, criterion):
        super(Id_Loss, self).__init__()
        self.criterion = criterion

    def calculate_IdLoss(self, score_img, score_text, label):

        label = label.view(label.size(0))

        Lipt_local = 0
        Ltpi_local = 0


        Lipt_local += self.criterion(score_img, label)
        Ltpi_local += self.criterion(score_text, label)

        loss = (Lipt_local + Ltpi_local)

        return loss

    def forward(self, score_img, score_text, label):
        # print('image_embedding_local, text_embedding_local', image_embedding_local.shape, text_embedding_local.shape)
        loss = self.calculate_IdLoss(score_img, score_text, label)

        return loss

