import torch
import torch.nn as nn
import torchvision
from resnest.torch import resnest50


class ResNeXt(nn.Module):
    def __init__(self, num_class, cfg):
        super(ResNeXt,self).__init__()
        resnet50 = resnest50(pretrained=True)
        # resnet50.layer4[0].downsample[0].stride = (1, 1)
        # resnet50.layer4[0].conv2.stride = (1, 1)
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


if __name__ =='__main__':
    # get list of models
    # torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)

    # load pretrained models, using ResNeSt-50 as an example
    net = resnest50(pretrained=True)
    model = net.cuda()
    print(model.layer4)

    x = torch.randn(2, 3, 384, 128).cuda()

    out = model(input)
    print(out.shape)