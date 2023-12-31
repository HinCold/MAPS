import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import collections
import numpy as np
class CFA(nn.Module):
    def __init__(self, s=50.0, m=0.20, class_num=11003, in_features=2048,bias=False):
        super(CFA, self).__init__()

        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.epsilon = 1e-8
        self.criterion = nn.CrossEntropyLoss() # nn.NLLLoss(reduction="sum") #nn.CrossEntropyLoss()

        # self.weight = nn.Parameter(torch.FloatTensor(class_num, in_features)).cuda()
        # nn.init.xavier_uniform_(self.weight)


    def forward(self, input_img, input_text, labels):
        # print('1', input_img.shape, input_text.shape)
        img_batch_clsfeats = collections.defaultdict(list)
        text_batch_clsfeats = collections.defaultdict(list)
        new_labels = []
        new_img = []
        new_text = []
        for i, t, l in zip(input_img, input_text, labels):
            l = l.item()
            t = t.cpu()
            i = i.cpu()
            img_batch_clsfeats[l].append(i)
            text_batch_clsfeats[l].append(t)
        # print(len(img_batch_clsfeats))
        for index, img_fea in img_batch_clsfeats.items():
            text_fea = text_batch_clsfeats[index]
            if len(img_fea) > 1:
                text_fea = torch.stack(text_fea, dim=0) #torch.from_numpy(np.array(text_batch_clsfeats[index]))
                img_fea = torch.stack(img_fea, dim=0)
                # print(img_fea, text_fea)
                text_fea = torch.mean(text_fea, dim=0)
                img_fea = torch.mean(img_fea, dim=0)

                new_labels.append(index)
                new_img.append(img_fea)
                new_text.append(text_fea)
            else:
                # print(img_fea[0].shape)
                new_labels.append(index)
                new_img.append(img_fea[0])
                new_text.append(text_fea[0])
        # print(new_img[34])
        input_img = torch.stack(new_img, dim=0).cuda()
        input_text = torch.stack(new_text, dim=0).cuda()
        labels = torch.tensor(new_labels, dtype=torch.int64)
        # print('2',input_img.shape, input_text.shape)
        # c
        batch_size = input_img.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)
        labels_mask_norm = labels_mask_norm.cuda()
        cosine = F.linear(F.normalize(input_img), F.normalize(input_text))
        # cosine = torch.matmul(input_img, F.normalize(input_text))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        #TODO easy margin
        b = (cosine - self.mm).float()
        phi = torch.where(cosine > self.th, phi, b)  # drop to CosFace

        # print(i2t_loss)
        cos_theat_ = torch.exp(cosine * self.s)
        sum_cos_theat = torch.sum(torch.exp(cosine * self.s), dim=1, keepdim=True) - cos_theat_
        top = torch.exp(phi * self.s)
        thetas = torch.log(top / (top + sum_cos_theat))
        # i2t_loss = F.softmax(i2t_loss, dim=1)
        thetas_pred = F.softmax(thetas, dim=1)
        #
        i2t_loss = thetas_pred * (F.log_softmax(thetas, dim=1) - torch.log(labels_mask_norm + self.epsilon))

        cosine = F.linear(F.normalize(input_text), F.normalize(input_img))

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        #TODO easy margin
        b = (cosine - self.mm).float()
        phi = torch.where(cosine > self.th, phi, b)  # drop to CosFace

        cos_theat_ = torch.exp(cosine * self.s)
        sum_cos_theat = torch.sum(torch.exp(cosine * self.s), dim=1, keepdim=True) - cos_theat_
        top = torch.exp(phi * self.s)
        thetas = torch.log(top / (top + sum_cos_theat))
        thetas_pred = F.softmax(thetas, dim=1)
        t2i_loss = thetas_pred * (F.log_softmax(thetas, dim=1) - torch.log(labels_mask_norm + self.epsilon))

        output = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        return output
