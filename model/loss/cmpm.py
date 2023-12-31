# -*- coding: utf-8 -*-
"""
@Time ： 21-10-7 上午11:14
@Auth ： Nathan
@File ：cmpm.py
@WF ： ...
"""
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class CMPMLoss(nn.Module):
    def __init__(self, epsilon, num_classes, feature_size):
        super(CMPMLoss, self).__init__()
        # self.mse = nn.MSELoss()
        self.epsilon = epsilon
        self.num_classes = num_classes

        self.W = Parameter(torch.randn(feature_size, self.num_classes))
        self.init_weight()


    def init_weight(self):
        nn.init.xavier_uniform_(self.W.data, gain=1)

    def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels):
        """
        Cross-Modal Projection Matching Loss(CMPM)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
            i2t_loss: cmpm loss for image projected to text
            t2i_loss: cmpm loss for text projected to image
            pos_avg_sim: average cosine-similarity for positive pairs
            neg_avg_sim: averate cosine-similarity for negative pairs
        """

        batch_size = image_embeddings.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)

        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        image_proj_text = torch.matmul(image_embeddings, text_norm.t())
        text_proj_image = torch.matmul(text_embeddings, image_norm.t())

        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

        i2t_pred = F.softmax(image_proj_text, dim=1)

        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + self.epsilon))
        t2i_pred = F.softmax(text_proj_image, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + self.epsilon))

        refine_loss = i2t_loss + t2i_loss
        # print(refine_loss, refine_loss.shape)
        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        return cmpm_loss, refine_loss

    def forward(self, img_f4, txt_f11, labels):
        loss = self.compute_cmpm_loss(img_f4, txt_f11, labels)
        return loss