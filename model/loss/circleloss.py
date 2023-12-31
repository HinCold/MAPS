# -*- coding: utf-8 -*-
"""
@Time ： 21-10-18 下午4:10
@Auth ： Nathan
@File ：circleloss.py
@WF ： ...
"""
import torch
import torch.nn as nn
class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()


    def forward(self,feat,label) -> Tensor:

        feat=normalize(feat,axis=-1)
        sp, sn = convert_label_to_similarity(feat, label)
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss
