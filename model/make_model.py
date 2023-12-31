# -*- coding: utf-8 -*-

import collections
import random
random.seed(42)
import copy
from functools import partial
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import torch
import torch.nn as nn
import copy
from model.backbone.image import ResNet50, RMGLModel, IBNResNet50, SENet50, VIT, ResNeXt
from model.backbone.text.bert import Bert
from model.loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
import transformers as ppb
from model.backbone.text.text_feature_extract import TextExtract
from torchvision import models

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

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
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class build_resnet(nn.Module):
    def __init__(self, cfg, num_class):
        super(build_resnet, self).__init__()
        embed_size = 3584
        self.id = False
        self.AlPHA = cfg.MODEL.AlPHA
        self.cfa = cfg.MODEL.CFA
        self.fusion = cfg.MODEL.FUSION
        if cfg.MODEL.NAME == 'ibn':
            print('===========building ibn-ResNet50 ===========')
            self.im_backbone = IBNResNet50(num_class, cfg)
        elif cfg.MODEL.NAME == 'SE':
            print('===========building SENet ===========')
            self.im_backbone = SENet50(num_class, cfg)
        elif cfg.MODEL.NAME == 'VIT':
            print('===========building VIT ===========')
            self.im_backbone = VIT(num_class,  cfg)
        elif cfg.MODEL.NAME == 'Next':
            print('===========building RESNext ===========')
            self.im_backbone = ResNeXt(num_class,  cfg)
        else :
            print('===========building ResNet50 ===========')
            self.im_backbone = ResNet50(num_class, cfg)
        self.text_backbone = Bert(cfg)
        self.num_classes = num_class
        
        # self.conv1 = nn.Conv1d(3584*2, 3584, kernel_size=1, padding=0, bias=False)
        self.conv2 = nn.Conv1d(3584, 1, kernel_size=1, padding=0, bias=False)
        self.conv3 = nn.Conv1d(3584, 1, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()


    def rerange(self, img_emb, text_emb, label):

        batch_centers = collections.defaultdict(list)
        index_list = []
        for i, l in enumerate(label):
            l = l.item()
            batch_centers[l].append(i)
            index_list.append(i)

        img_extra_list = []
        text_extra_list = []
        label_list = []

        # AlPHA = self.AlPHA
        # Betal = 0.3
        for index, ids in batch_centers.items():
            # print(index, features)
            index = torch.tensor(index).long().cuda()
            s1 = random.choice(ids)
            s2 = random.choice(ids)
            if s1 == s2:
                continue
            else:
                img_s1 = img_emb[s1]
                img_s2 = img_emb[s2]
                text_s2 = text_emb[s2]
                text_s1 = text_emb[s1]
                imgcfa_s1 = img_s1.view(1,-1,1)
                imgcfa_s2 = img_s2.view(1,-1,1)
                # img_cfa = imgcfa_s1+imgcfa_s2 #torch.cat((imgcfa_s1, imgcfa_s2), dim=1)
                # y = self.avg_pool(img_cfa)

                # y1 = self.conv2(imgcfa_s1)
                # y2 = self.conv3(imgcfa_s2)
                # Multi-scale information fusion
                AlPHA = 1 #self.sigmoid(y1)
                Belta = 1 #self.sigmoid(y2)
                img_s1new = (AlPHA * img_s1 + Belta * img_s2).view(-1)
                img_s2new = (AlPHA * img_s2 + Belta * img_s1).view(-1)
                # img_s1new = (AlPHA * img_s1 + (1.0 - AlPHA) * img_s2).view(-1)
                # img_s2new = (AlPHA * img_s2 + (1.0 - AlPHA) * img_s1).view(-1)
                # print(img_s1new.shape, img_s2new.shape)
                # text_s1new = Betal * text_s1 + (1.0 - Betal) * text_s2
                # text_s2new = Betal * text_s2 + (1.0 - Betal) * text_s1

                img_extra_list.append(img_s1new)
                img_extra_list.append(img_s2new)
                text_extra_list.append(text_s1)
                text_extra_list.append(text_s2)
                label_list.append(index)
                label_list.append(index)

            # index_list[s2], index_list[s1] = index_list[s1], index_list[s2]
        if len(img_extra_list) != 0:
            img_extra_tensor = torch.stack(img_extra_list, dim=0)
            text_extra_tensor = torch.stack(text_extra_list, dim=0)
            label_extra_tensor = torch.stack(label_list, dim=0)
            img_emb = torch.cat([img_emb, img_extra_tensor], dim=0)
            text_emb = torch.cat([text_emb, text_extra_tensor], dim=0)
            label = torch.cat([label, label_extra_tensor], dim=0)

        return img_emb, text_emb, label

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']

        new_dict = {k: v for k, v in param_dict.items() if k in self.state_dict().keys()}
        for i in new_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))



    def forward(self, img, text, mask, label=None, label_attr=None):
 
        text_1x3, text_1x2, text_1x1 = self.text_backbone(text, mask)  # batch x F_DIM , batch x MAX_LENGTH x F_DIM
        img_lay4, img_lay3, img_lay2 = self.im_backbone(img)  # batch x F_DIM, batch x A x F_DIM

        if self.fusion:
            text_glo = torch.cat((text_1x2, text_1x1, text_1x3), dim=1)
            img_glo = torch.cat((img_lay4, img_lay3, img_lay2), dim=1)
        else:
            text_glo = text_1x1
            img_glo = img_lay4

        img_glo = img_glo.squeeze(dim=-1).squeeze(dim=-1)
        text_glo = text_glo.squeeze(dim=-1).squeeze(dim=-1)


        if self.training:
            if  self.cfa:
                img_glo, text_glo, label = self.rerange(img_glo, text_glo, label)
            return img_glo, text_glo, label

        return img_glo, text_glo


 
def make_model(cfg, num_class):
    if cfg.MODEL.NAME == 'SSAN':
        print('===========building SSAN-TextImgPersonReidNet ===========')
        # model = TextImgPersonReidNet(cfg).to(cfg.device)
    else:
        print('===========building MAPS ===========')
        model = build_resnet(cfg, num_class)

    return model
