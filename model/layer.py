# -*- coding: utf-8 -*-
"""
@Time ： 21-11-1 上午10:53
@Auth ： Nathan
@File ：layer.py
@WF ： ...
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
import math


class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


def init_params(x):
    if x is None:
        return

    for m in x.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1, 0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.normal_(m.weight, 1, 0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


##########
# Building blocks for omni-scale feature learning
##########
class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(self, in_channels, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # self.global_avgpool = AdaptiveGeM2d()
        # self.fc = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True, padding=0)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates, kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
            # self.gate_activation = Hsigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError("Unknown gate activation: {}".format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        # x = self.fc(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = AdaptiveGeM2d()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight


def adaptive_gem2d(x, output_size=(1, 1), p=3, eps=1e-6):
    return F.adaptive_avg_pool2d(x.clamp(min=eps).pow(p), output_size).pow(1. / p)


class AdaptiveGeM2d(nn.Module):

    def __init__(self, output_size=(1, 1), p=3.0, eps=1e-6, freeze_p=True):
        super(AdaptiveGeM2d, self).__init__()
        self.output_size = output_size
        self.p = p if freeze_p else Parameter(torch.ones(1) * p)
        self.eps = eps
        self.freeze_p = freeze_p

    def forward(self, x):
        # return LF.gem(x, p=self.p, eps=self.eps)
        return adaptive_gem2d(x, self.output_size, p=self.p, eps=self.eps)

    def __repr__(self):
        if isinstance(self.p, float):
            p = self.p
        else:
            p = self.p.data.tolist()[0]
        return self.__class__.__name__ + \
               '(' + 'output_size=' + '{}'.format(self.output_size) + \
               ',' + 'p=' + '{:.4f}'.format(p) + \
               ', ' + 'eps=' + str(self.eps) + \
               ', ' + 'freeze_p=' + str(self.freeze_p) + \
               ')'


class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)


class conv(nn.Module):

    def __init__(self, input_dim, output_dim, relu=False, BN=False):
        super(conv, self).__init__()

        block = []
        block += [nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)]

        if BN:
            block += [nn.BatchNorm2d(output_dim)]
        if relu:
            block += [nn.LeakyReLU(0.25, inplace=True)]

        self.block = nn.Sequential(*block)
        self.block.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.block(x)
        x = x.squeeze(3).squeeze(2)
        return x


class NonLocalNet(nn.Module):
    def __init__(self, opt, dim_cut=8):
        super(NonLocalNet, self).__init__()
        self.opt = opt

        up_dim_conv = []
        part_sim_conv = []
        cur_sim_conv = []
        conv_local_att = []
        for i in range(opt.part):
            up_dim_conv.append(conv(opt.feature_length//dim_cut, 1024, relu=True, BN=True))
            part_sim_conv.append(conv(opt.feature_length, opt.feature_length // dim_cut, relu=True, BN=False))
            cur_sim_conv.append(conv(opt.feature_length, opt.feature_length // dim_cut, relu=True, BN=False))
            conv_local_att.append(conv(opt.feature_length, 512))

        self.up_dim_conv = nn.Sequential(*up_dim_conv)
        self.part_sim_conv = nn.Sequential(*part_sim_conv)
        self.cur_sim_conv = nn.Sequential(*cur_sim_conv)
        self.conv_local_att = nn.Sequential(*conv_local_att)

        self.zero_eye = (torch.eye(opt.part, opt.part) * -1e6).unsqueeze(0).to(opt.device)

        self.lambda_softmax = 1

    def forward(self, embedding):
        embedding = embedding.unsqueeze(3)
        embedding_part_sim = []
        embedding_cur_sim = []

        for i in range(self.opt.part):
            embedding_i = embedding[:, :, i, :].unsqueeze(2)

            embedding_part_sim_i = self.part_sim_conv[i](embedding_i).unsqueeze(2)
            embedding_part_sim.append(embedding_part_sim_i)

            embedding_cur_sim_i = self.cur_sim_conv[i](embedding_i).unsqueeze(2)
            embedding_cur_sim.append(embedding_cur_sim_i)

        embedding_part_sim = torch.cat(embedding_part_sim, dim=2)
        embedding_cur_sim = torch.cat(embedding_cur_sim, dim=2)

        embedding_part_sim_norm = l2norm(embedding_part_sim, dim=1)  # N*D*n
        embedding_cur_sim_norm = l2norm(embedding_cur_sim, dim=1)  # N*D*n
        self_att = torch.bmm(embedding_part_sim_norm.transpose(1, 2), embedding_cur_sim_norm)  # N*n*n
        self_att = self_att + self.zero_eye.repeat(self_att.size(0), 1, 1)
        self_att = F.softmax(self_att * self.lambda_softmax, dim=1)  # .transpose(1, 2).contiguous()
        embedding_att = torch.bmm(embedding_part_sim_norm, self_att).unsqueeze(3)

        embedding_att_up_dim = []
        for i in range(self.opt.part):
            embedding_att_up_dim_i = embedding_att[:, :, i, :].unsqueeze(2)
            embedding_att_up_dim_i = self.up_dim_conv[i](embedding_att_up_dim_i).unsqueeze(2)
            embedding_att_up_dim.append(embedding_att_up_dim_i)
        embedding_att_up_dim = torch.cat(embedding_att_up_dim, dim=2).unsqueeze(3)

        embedding_att = embedding + embedding_att_up_dim

        embedding_local_att = []
        for i in range(self.opt.part):
            embedding_att_i = embedding_att[:, :, i, :].unsqueeze(2)
            embedding_att_i = self.conv_local_att[i](embedding_att_i).unsqueeze(2)
            embedding_local_att.append(embedding_att_i)

        embedding_local_att = torch.cat(embedding_local_att, 2)

        return embedding_local_att.squeeze()
