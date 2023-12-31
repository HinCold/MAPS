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
# from model.layer import SELayer
from torchvision.models import ResNet
from torch.hub import load_state_dict_from_url
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

class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
 
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def se_resnet50(num_classes=5, pretrained=True):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(
            "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))
    return model
 
class SENet50(nn.Module):
    def __init__(self, num_class, cfg):
        super(SENet50, self).__init__()
        resnet50 = se_resnet50(1000)
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

        # self.conv1x1_512 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(512),
        #     #nn.AdaptiveMaxPool2d((1, 1))
        #     # self.act
        # )
        # self.conv1x1_1024 = nn.Sequential(
        #     nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(1024),
        #     #nn.AdaptiveMaxPool2d((1, 1))
        #     # self.act
        # )
        # self.conv1x1_2048 = nn.Sequential(
        #     nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(2048),
        #     #nn.AdaptiveMaxPool2d((1, 1))
        #     # self.act
        # )


    def forward(self, x):
        x1 = self.base1(x)
        img_embeds_512 = self.base2(x1)
        img_embeds_1024 = self.base3(img_embeds_512)
        img_embeds_2048 = self.base4(img_embeds_1024)

        # img_embeds_512 = self.conv1x1_512(img_embeds_512)
        # img_embeds_1024 = self.conv1x1_1024(img_embeds_1024)
        # img_embeds_2048 = self.conv1x1_2048(img_embeds_2048) #

        img_embeds_512 = self.pool(img_embeds_512)
        img_embeds_1024 = self.pool(img_embeds_1024)
        img_embeds_2048 = self.pool(img_embeds_2048) #.squeeze(dim=-1).squeeze(dim=-1)

        return img_embeds_2048, img_embeds_1024, img_embeds_512
