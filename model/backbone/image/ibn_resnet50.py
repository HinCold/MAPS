# -*- coding: utf-8 -*-

from torch.nn import init
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import warnings
from torch.nn import Parameter
# from model.layer import AdaptiveGeM2d
model_urls = {
    'resnet50_ibn': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


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


class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None):
        super(Bottleneck_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class ResNet_IBN(nn.Module):

    def __init__(self, last_stride, block, layers, num_classes=1000):
        scale = 64
        self.inplanes = scale
        super(ResNet_IBN, self).__init__()
        self.conv1 = nn.Conv2d(3, scale, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(scale)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, scale, layers[0])
        self.layer2 = self._make_layer(block, scale * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, scale * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, scale * 8, layers[3], stride=last_stride)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(scale * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        ibn = True
        if planes == 512:
            ibn = False
        layers.append(block(self.inplanes, planes, ibn, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])


def resnet50_ibn_a(last_stride=1, pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(last_stride, Bottleneck_IBN, [3, 4, 6, 3], **kwargs)
    state_dict = torch.load('/home/lh/project/SeqNet/resnet50_ibn_a-d9d0bb7b.pth')
    if pretrained:
        print('loaded resnet50_ibn_a-d9d0bb7b.pth')
        model.load_state_dict(state_dict)
    return model

class IBNResNet50(nn.Module):
    def __init__(self, num_class, cfg):
        super(IBNResNet50, self).__init__()
        ibnresnet50 = resnet50_ibn_a()
        self.base1 = nn.Sequential(
            ibnresnet50.conv1,
            ibnresnet50.bn1,
            ibnresnet50.relu,
            ibnresnet50.maxpool,
            ibnresnet50.layer1,  # 256 64 32
        )
        self.base2 = nn.Sequential(
            ibnresnet50.layer2,  # 512 32 16
        )
        self.base3 = nn.Sequential(
            ibnresnet50.layer3,  # 1024 16 8
        )
        self.base4 = nn.Sequential(
            ibnresnet50.layer4  # 2048 16 8
        )
        # self.base5 = conv(cfg.MODEL.FEARTURE_DIM, 512)
        #
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveMaxPool2d((1, 1)) #AdaptiveGeM2d() #nn.AdaptiveMaxPool2d((1, 1))
        size = 1
        # self.conv1x1_512 = nn.Conv2d(512, 512, kernel_size=size, stride=1,
        #                        padding=0, bias=False)
        # self.conv1x1_1024 = nn.Conv2d(1024, 1024, kernel_size=size, stride=1,
        #                        padding=0, bias=False)
        # self.conv1x1_2048 = nn.Conv2d(2048, 2048, kernel_size=size, stride=1,
        #                        padding=0, bias=False)

        self.conv1x1_512 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            # nn.AdaptiveMaxPool2d((1, 1))
            # self.act
        )
        self.conv1x1_1024 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            # nn.AdaptiveMaxPool2d((1, 1))
            # self.act
        )
        self.conv1x1_2048 = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            # nn.AdaptiveMaxPool2d((1, 1))
            # self.act
        )

        self.sigmoid = nn.Sigmoid()

    def img_swm512(self, img_embeds_512):
        img_embeds_512_ = self.conv1x1_512(img_embeds_512)
        img_embeds_512_ = img_embeds_512+img_embeds_512_
        img_embeds_512 = self.act(img_embeds_512_)
        # img_embeds_512 = self.sigmoid(img_embeds_512_) * img_embeds_512
        return img_embeds_512

    def img_swm1024(self, img_embeds_1024):
        img_embeds_1024_ = self.conv1x1_1024(img_embeds_1024)
        img_embeds_1024_ = img_embeds_1024+img_embeds_1024_
        img_embeds_1024 = self.act(img_embeds_1024_)
        # img_embeds_1024 = self.sigmoid(img_embeds_1024_) * img_embeds_1024
        return img_embeds_1024

    def img_swm2048(self, img_embeds_2048):
        img_embeds_2048_ = self.conv1x1_2048(img_embeds_2048)
        img_embeds_2048_ = img_embeds_2048+img_embeds_2048_
        img_embeds_2048 = self.act(img_embeds_2048_)
        # img_embeds_2048 = self.sigmoid(img_embeds_2048_) * img_embeds_2048
        return img_embeds_2048

    def forward(self, x):
        x1 = self.base1(x)
        img_embeds_512 = self.base2(x1)
        img_embeds_1024 = self.base3(img_embeds_512)
        img_embeds_2048 = self.base4(img_embeds_1024)

        # img_embeds_512 = self.conv1x1_512(img_embeds_512)
        # img_embeds_1024 = self.conv1x1_1024(img_embeds_1024)
        # img_embeds_2048 = self.conv1x1_2048(img_embeds_2048) #
        img_embeds_512 = self.img_swm512(img_embeds_512)
        img_embeds_1024 = self.img_swm1024(img_embeds_1024)
        img_embeds_2048 = self.img_swm2048(img_embeds_2048)

        img_embeds_512 = self.pool(img_embeds_512)
        img_embeds_1024 = self.pool(img_embeds_1024)
        img_embeds_2048 = self.pool(img_embeds_2048) #.squeeze(dim=-1).squeeze(dim=-1)

        # img_embeds_all = torch.cat((img_embeds_2048, img_embeds_1024, img_embeds_512), dim=1).squeeze(dim=-1).squeeze(dim=-1)
        # return img_embeds_all
        return img_embeds_2048, img_embeds_1024, img_embeds_512

if __name__ == "__main__":
    #from torchvision.models import resnet50, resnet101, resnet152, resnext101_32x8d

    # model = resnet50_ibn_a(1, pretrained=False)
    # print('model', model)
    #
    # from torchstat import stat
    #
    # stat(model, (3, 1500, 900))
    import collections
    import random
    la = torch.tensor([1, 2, 1,3,5,2,6])
    a = torch.tensor([[1, 5, 3, 4], [2, 3, 4, 5], [1, 2, 3, 6],
                      [6, 5, 3, 4], [6, 3, 4, 5], [6, 2, 3, 6], [1, 2, 3, 7]])

    batch_centers = collections.defaultdict(list)
    index_list = []
    for i, l in enumerate(la):
        l = l.item()
        batch_centers[l].append(i)
        index_list.append(i)
    print('index', index_list)
    for index, features in batch_centers.items():
        print(index, features)
        s1 = random.choice(features)
        s2 = random.choice(features)
        index_list[s2], index_list[s1] = index_list[s1], index_list[s2]
        print(s1, s2)
    print(index_list)
    a = a[index_list]
    # b = torch.transpose(a, dim0=0, dim1=2)
    print(a)
    # print(b.shape)