import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class ArcFace(nn.Module):
    def __init__(self, s=50.0, m=0.20, class_num=11003, in_features=2048,bias=False):
        super(ArcFace, self).__init__()

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

        batch_size = input_img.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

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
        # t2i_loss = F.softmax(t2i_loss, dim=1)
        thetas_pred = F.softmax(thetas, dim=1)
        t2i_loss = thetas_pred * (F.log_softmax(thetas, dim=1) - torch.log(labels_mask_norm + self.epsilon))

        # # arc_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
        # t2i_loss = (labels_mask_norm * phi) + ((1.0 - labels_mask_norm) * cosine)
        # t2i_loss *= self.s

        output = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
        #
        # refine_loss = i2t_loss + t2i_loss
        # # print(refine_loss, refine_loss.shape)
        # cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
        #
        # c
        # batch_size = len(thetas)
        # thetas[range(batch_size), label] = phi[range(batch_size), label]
        #
        # thetas *= self.s

        # output = arc_loss #torch.mean(torch.sum(arc_loss, dim=1)) #self.criterion(thetas, label)
        # output = output.half()
        # print(output)

        return output

class ArcFace2(nn.Module):
    def __init__(self, s=30.0, m=0.20, class_num=11003, in_features=2048,bias=False):
        super(ArcFace, self).__init__()

        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.epsilon = 1e-8
        self.criterion = nn.CrossEntropyLoss() #nn.CrossEntropyLoss()

        self.weight1 = nn.Parameter(torch.FloatTensor(in_features, class_num)).cuda()
        nn.init.xavier_uniform_(self.weight1)
        self.weight2 = nn.Parameter(torch.FloatTensor(in_features, class_num)).cuda()
        nn.init.xavier_uniform_(self.weight2)

    def forward(self, image_embeddings, text_embeddings, labels):
        labels = labels.view(labels.size(0)).long()
        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        batch_size = image_embeddings.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

        image_proj_text = torch.matmul(
            torch.matmul(image_embeddings, text_norm.t()),
            text_norm)
        text_proj_image = torch.matmul(
            torch.matmul(text_embeddings, image_norm.t()),
            image_norm)
        # print(image_proj_text.shape, text_proj_image.shape)
        # # text_proj_image = torch.matmul(text_embeddings, image_norm.t())

        cosine = F.linear(F.normalize(image_proj_text), F.normalize(text_proj_image))
        # print(cosine.shape)
        # cosine = torch.matmul(input_img, F.normalize(input_text))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        # TODO easy margin
        b = (cosine - self.mm).float()
        phi = torch.where(cosine > self.th, phi, b)  # drop to CosFace

        #
        cos_theat_ = torch.exp(cosine * self.s)
        sum_cos_theat = torch.sum(torch.exp(cosine * self.s), dim=1, keepdim=True) - cos_theat_
        top = torch.exp(phi * self.s)
        thetas = torch.log(top / (top + sum_cos_theat))
        # print(cosine.shape)
        thetas_pred = F.softmax(thetas, dim=1)
        #
        i2t_loss = thetas_pred * (F.log_softmax(thetas, dim=1) - torch.log(labels_mask_norm + self.epsilon))

        # i2t_loss = self.criterion(thetas, labels)

        # print(text_proj_image.shape, text_embeddings.shape)
        cosine = F.linear(F.normalize(text_proj_image), F.normalize(image_proj_text))

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        # TODO easy margin
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

