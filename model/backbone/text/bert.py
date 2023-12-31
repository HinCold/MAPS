# -*- coding: utf-8 -*-

import torch
from torch.nn import init
import transformers as ppb
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings


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
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0)


def conv1x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """1x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), stride=stride,
                     padding=(0, 1), groups=groups, bias=False, dilation=dilation)


def conv1x2(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """1x2 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 2), stride=stride,
                     padding=(0, 1), groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bert(nn.Module):
    def __init__(self, cfg):  # nn.LayerNorm
        super(Bert, self).__init__()
        self.in_planes = 768
        self.out_planes = cfg.MODEL.BERT_FEARTURE_DIM  # BERT_FEARTURE_DIM FINAL_FEARTURE_DIM
        self.out_planes2 = cfg.MODEL.FINAL_FEARTURE_DIM
        # self.token_type_embeddings = nn.Embedding(2, hidden_size)
        # self.token_type_embeddings.apply(init_weights)
        model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        # print('BERT', feature_size)
        self.text_embed = model_class.from_pretrained(pretrained_weights)
        self.text_embed.eval()
        for p in self.text_embed.parameters():
            p.requires_grad = False
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        # self.norm_layer = nn.BatchNorm2d(self.out_planes)
        self.act = nn.GELU()  # nn.LeakyReLU #nn.GELU() #nn.ReLU(inplace=True) #nn.GELU()
        self.act1 = nn.ReLU(inplace=True)
        self.block1x1 = nn.Sequential(
            conv1x1(self.in_planes, self.out_planes //2),
            nn.BatchNorm2d(self.out_planes //2),
            self.act
        )

        self.block1x2 = nn.Sequential(
            conv1x2(self.out_planes //2, self.out_planes), #self.out_planes // 2
            nn.BatchNorm2d(self.out_planes),
            self.act
        )

        self.block1x3 = nn.Sequential(
            conv1x3(self.out_planes //2, self.out_planes // 4),
            nn.BatchNorm2d(self.out_planes // 4),
            self.act
        )

        self.conv1x1_512 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(512),
            # nn.AdaptiveMaxPool2d((1, 1))
            # self.act
        )
        self.conv1x1_1024 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(1024),
            # nn.AdaptiveMaxPool2d((1, 1))
            # self.act
        )
        self.conv1x1_2048 = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(2048),
            # nn.AdaptiveMaxPool2d((1, 1))
            # self.act
        )
        # self.conv1x1_1536 = nn.Conv2d(1536, 1536, kernel_size=1, stride=1,
        #                              padding=0, bias=False)
        # self.conv1x1_512 = nn.Conv2d(512, 512, kernel_size=1, stride=1,
        #                              padding=0, bias=False)
        # self.conv1x1_1024 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1,
        #                               padding=0, bias=False)
        # self.conv1x1_2048 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1,
        #                               padding=0, bias=False)

        self.conv1x1_512.apply(weights_init_kaiming)
        self.conv1x1_1024.apply(weights_init_kaiming)
        # self.conv1x1_1536.apply(weights_init_kaiming)
        self.conv1x1_2048.apply(weights_init_kaiming)

        self.block1x1.apply(weights_init_kaiming)
        self.block1x2.apply(weights_init_kaiming)
        self.block1x3.apply(weights_init_kaiming)
        self.eps = 0.0001
        self.sigmoid = nn.Sigmoid()

    def text_swm1x1(self, text_embeds1x1):
        text_embeds1x1_ = self.conv1x1_1024(text_embeds1x1)
        text_embeds1x1_ = text_embeds1x1 + text_embeds1x1_
        text_embeds1x1 = self.act(text_embeds1x1_)
        # text_embeds1x1 = self.sigmoid(text_embeds1x1_) * text_embeds1x1
        return text_embeds1x1

    def text_swm1x2(self, text_embeds1x2):
        text_embeds1x2_ = self.conv1x1_2048(text_embeds1x2)
        text_embeds1x2_ = text_embeds1x2_ + text_embeds1x2
        text_embeds1x2 = self.act(text_embeds1x2_)
        # text_embeds1x2 = self.sigmoid(text_embeds1x2_) * text_embeds1x2
        return text_embeds1x2

    def text_swm1x3(self, text_embeds1x3):
        text_embeds1x3_ = self.conv1x1_512(text_embeds1x3)
        text_embeds1x3_ = text_embeds1x3_ + text_embeds1x3
        text_embeds1x3 = self.act(text_embeds1x3_)
        # text_embeds1x3 = self.sigmoid(text_embeds1x3_) * text_embeds1x3
        return text_embeds1x3

        # sfm = nn.

    def forward(self, text_ids, text_masks):
        with torch.no_grad():
            text_embeds = self.text_embed(text_ids, attention_mask=text_masks)
            text_embeds = text_embeds[0]
            text_embeds = text_embeds.unsqueeze(1)
            text_embeds = text_embeds.permute(0, 3, 1, 2)

        text_embeds1x1 = self.block1x1(text_embeds)
        text_embeds1x2 = self.block1x2(text_embeds1x1)
        text_embeds1x3 = self.block1x3(text_embeds1x1)

        text_embeds1x1 = self.text_swm1x1(text_embeds1x1)
        text_embeds1x2 = self.text_swm1x2(text_embeds1x2)
        text_embeds1x3 = self.text_swm1x3(text_embeds1x3)

        text_embeds1x3 = self.pool(text_embeds1x3)
        text_embeds1x1 = self.pool(text_embeds1x1)
        text_embeds1x2 = self.pool(text_embeds1x2)  # .squeeze(dim=-1).squeeze(dim=-1)

        # return None, None, text_embeds1x1
        return text_embeds1x3, text_embeds1x2, text_embeds1x1
