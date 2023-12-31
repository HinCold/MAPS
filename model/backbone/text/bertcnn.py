# -*- coding: utf-8 -*-
"""
@Time ： 21-10-7 上午11:15
@Auth ： Nathan
@File ：bert.py
@WF ： ...
"""
import torch
from torch.nn import init
import torch.nn as nn
import transformers as ppb
import torchvision
import torchvision.transforms as transforms
# from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

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

def conv1x2(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """1x2 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 2), stride=stride,
                     padding=(0,1), groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BertCNN(nn.Module):
    def __init__(self, feature_size, num_classes):
        super(BertCNN, self).__init__()
        # BERT

        model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        print('BERT', feature_size)
        self.text_embed = model_class.from_pretrained(pretrained_weights)
        self.text_embed.eval()
        for p in self.text_embed.parameters():
            p.requires_grad = False

        self.feature_size = feature_size

        # CNN
        self.in_planes = 768
        self.num_classes = num_classes

        # self.block1 = nn.Sequential(
        #     conv1x1(self.in_planes, feature_size // 2),
        #     nn.BatchNorm2d(feature_size // 2),
        #     nn.ReLU(inplace=True)
        # )
        # self.block2 = nn.Sequential(
        #     conv1x2(feature_size // 2, feature_size),
        #     nn.BatchNorm2d(feature_size)
        # )
        #
        # self.block1.apply(weights_init_kaiming)
        # self.block2.apply(weights_init_kaiming)

        # self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Linear(feature_size, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)


    def forward(self, text, mask):
        with torch.no_grad():
            txt = self.text_embed(text, attention_mask=mask)
            txt = txt[0]
            # txt = self.drop(txt)
            x = txt

        x = self.pool(x).squeeze(dim=-1)
        print('x', x.shape)
        c

        cls_score = self.classifier(x)
        if self.training:
            return cls_score, x
        else:
            return x