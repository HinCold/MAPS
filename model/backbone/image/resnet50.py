# -*- coding: utf-8 -*-
"""
@Time ： 21-10-7 上午11:15
@Auth ： Nathan
@File ：resnet50.py
@WF ： ...
"""
from torchvision import models
import torch
from torch.nn import init
import torch.nn as nn
 
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

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
        return x

class ResNet50(nn.Module):
    def __init__(self, num_class, cfg):
        super(ResNet50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        resnet50.layer4[0].downsample[0].stride = (1, 1)
        resnet50.layer4[0].conv2.stride = (1, 1)
        self.base1 = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,  # 256 64 32
        )
        self.base2 = nn.Sequential(
            resnet50.layer2,  # 512 32 16
        )
        self.base3 = nn.Sequential(
            resnet50.layer3,  # 1024 16 8
        )
        self.base4 = nn.Sequential(
            resnet50.layer4  # 2048 16 8
        )
        # self.base5 = conv(cfg.MODEL.FEARTURE_DIM, 512)
        #
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        # self.attn = SELayer(2048)

        self.conv1x1_512 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            #nn.AdaptiveMaxPool2d((1, 1))
            # self.act
        )
        self.conv1x1_1024 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            #nn.AdaptiveMaxPool2d((1, 1))
            # self.act
        )
        self.conv1x1_2048 = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2048),
            #nn.AdaptiveMaxPool2d((1, 1))
            # self.act
        )


    def forward(self, x):
        x1 = self.base1(x)
        img_embeds_512 = self.base2(x1)
        img_embeds_1024 = self.base3(img_embeds_512)
        img_embeds_2048 = self.base4(img_embeds_1024)

        img_embeds_512 = self.conv1x1_512(img_embeds_512)
        img_embeds_1024 = self.conv1x1_1024(img_embeds_1024)
        img_embeds_2048 = self.conv1x1_2048(img_embeds_2048) #

        img_embeds_512 = self.pool(img_embeds_512)
        img_embeds_1024 = self.pool(img_embeds_1024)
        img_embeds_2048 = self.pool(img_embeds_2048) #.squeeze(dim=-1).squeeze(dim=-1)

        return img_embeds_2048, img_embeds_1024, img_embeds_512
