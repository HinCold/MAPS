# -*- coding: utf-8 -*-
"""
@Time ： 21-10-12 上午10:59
@Auth ： Nathan
@File ：RMGL.py
@WF ： ...
"""
from torchvision.models import resnet50
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


class ABP(nn.Module):
    def __init__(self, ns):
        super(ABP, self).__init__()
        self.ns = ns

    def make_I(self, x):
        height, width = x.shape[2], x.shape[3]
        a = x.view(x.shape[0], x.shape[1], -1)
        value, index = torch.max(a, 2)
        new_value = value.unsqueeze(-1).repeat(1, 1, height * width)
        mask = a == new_value
        I = torch.zeros_like(a)
        I[mask] = 1
        I = I.view(x.shape[0], x.shape[1], height, width)
        return I

    def make_H(self, x):
        I = self.make_I(x)
        height = x.shape[2]
        batch = x.shape[0]
        H = torch.zeros([batch, height]).to(x.device)
        for i in range(height):
            H[:, i] = I[:, :, :i, :].sum(dim=(1, 2, 3))
        return H

    def make_hk(self, x):
        H = self.make_H(x)
        C = x.shape[1]
        hks = torch.zeros(H.shape[0], self.ns + 1)
        hks[:, self.ns] = x.shape[2]

        for i in range(H.shape[0]):
            k = 1
            for j in range(1, H.shape[1]):
                # print('i,j' , i, j)
                if k == self.ns or j + 1>= H.shape[1]:
                    break
                if H[i, j] <= int(k * C / self.ns) and H[i, j + 1] > int(k * C / self.ns):
                    hks[i, k] = j
                    k += 1
        return hks

    def forward(self, x):
        #         print(x.shape)
        hk = self.make_hk(x)
        #         print(hk)
        F = x.sum(dim=(2, 3)) / x.shape[-1]
        hk_sub = torch.zeros(x.shape[0], self.ns, 1)
        for i in range(1, self.ns + 1):
            hk_sub[:, i - 1, 0] = hk[:, i] - hk[:, i - 1]
        hk_sub = hk_sub.to(x.device)
        F = F.unsqueeze(1)
        F = F.repeat(1, self.ns, 1)
        F = F / hk_sub

        F = F.view(F.shape[0], -1)
        return F

class Local_n_branch(nn.Module):
    def __init__(self, backbone, n, num_class):
        super(Local_n_branch, self).__init__()
        self.n = n
        m = 1
        # print('n', n)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.poola = nn.AdaptiveMaxPool2d((1, 1))
        # self.poolb = ABP(n)

        self.poolb = nn.AdaptiveMaxPool2d(1)
        self.reduction_linear = nn.Linear(2048 * (m+1), 256 * (m+1))
        self.dropout = nn.Dropout()
        self.bn = nn.BatchNorm1d(256 * (m+1))
        self.bn.bias.requires_grad_(False)
        self.fc = nn.Linear(256 * (m+1), num_class, bias=False)
        self.bn.apply(weights_init_kaiming)
        self.fc.apply(weights_init_classifier)

    def path_a_forward(self, x):
        gf = self.layer2(x)
        gf = self.layer3(gf)
        gf = self.layer4(gf)
        gf = self.poola(gf)
        return gf

    def path_b_forward(self, model, x):
        height = x.shape[2]

        x0 = x[:, :, :int(height / 2), :]
        x1 = x[:, :, int(height / 2):, :]
        x0 = model(x0)
        x1 = model(x1)
        x = torch.cat([x0, x1], dim=2)
        return x

    def forward(self, x):
        gf = self.path_a_forward(x)
        gf = gf.squeeze(dim=-1).squeeze(dim=-1)  #.view(gf.shape[0], -1)

        batch = x.shape[0]
        height = x.shape[2]
        inputs = []
        stride = int(height / self.n)
        for i in range(self.n):
            inputs.append(x[:, :, i * stride:(i + 1) * stride, :])
        lf = torch.cat(inputs, dim=0).cuda()
        lf = self.path_b_forward(self.layer2, lf)
        lf = self.path_b_forward(self.layer3, lf)
        lf = self.path_b_forward(self.layer4, lf)
        outputs = []
        for i in range(self.n):
            outputs.append(lf[i * batch:(i + 1) * batch])
        lf = torch.cat(outputs, dim=2).cuda()

        lf = self.poolb(lf).squeeze(dim=-1).squeeze(dim=-1)

        feature = torch.cat([gf, lf], dim=1)
        # print('shape', feature.shape,gf.shape, lf.shape, self.n)

        global_feature = self.reduction_linear(feature)
        feature = self.bn(global_feature)
        logit = self.fc(feature)
        return global_feature, feature, logit


class RMGLModel(nn.Module):
    def __init__(self, out_dim, num_class):
        super(RMGLModel, self).__init__()
        backbone = resnet50(True)
        self.share_backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1
        )
        self.local_2_branch = Local_n_branch(backbone, 2, num_class)
        self.local_3_branch = Local_n_branch(backbone, 3, num_class)
        self.classifier = nn.Linear(out_dim, num_class, bias=False)
        self.classifier.apply(weights_init_classifier)
        # self.cls_criterion = CrossEntropyLabelSmooth(num_class)
        # self.circle_criterion = CircleLoss(m=0.6, gamma=80)
        # self.pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x, labels=None):
        # print('1 x.shape', x.shape)
        x = self.share_backbone(x)
        # print('2 x.shape', x.shape)
        global_feat0, feat0, logit0 = self.local_2_branch(x)
        global_feat1, feat1, logit1 = self.local_3_branch(x)
        # print('shape 2', global_feat0.shape, feat0.shape)
        # print('shape 3', global_feat1.shape, feat1.shape)
        loss = 0
        # if self.training:
        #     loss = loss + self.cls_criterion(logit0, labels)
        #
        #     loss = loss + self.cls_criterion(logit1, labels)
        #
        #     loss = loss + self.circle_criterion(global_feat0, labels)
        #
        #     loss = loss + self.circle_criterion(global_feat1, labels)
        glboal_feat = torch.cat([global_feat0, global_feat1], dim=1)
        feat = torch.cat([feat0, feat1], dim=1)
        # print('feat', feat.shape)
        # feat = self.pool(feat).squeeze(dim=-1).squeeze(dim=-1)
        cls_score = self.classifier(feat)
        if self.training:
            return cls_score, feat, feat0, feat1
        else:
            return feat

if __name__ == '__main__':
    # 0.99 MB
    x = torch.randn(2, 3, 300, 300).cuda()
    net = RMGLModel(11003).cuda()
    net.eval()
    print(net(x)['feature'].shape)
    # from torchsummary import summary
    # summary(net, (3, 300, 300))
