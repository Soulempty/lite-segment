# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from .backbone.resnet import ResNet18, ResNet50
from torchvision import models
import torch.nn.functional as F

def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            
class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=8):
        super(segmenthead, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor
        init_weight(self)
    def forward(self, x):
        
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,size=[height, width],mode='bilinear',align_corners=False)
        return out

class FCN(nn.Module):
    def __init__(self, num_classes,scale_factor=4):
        print('FCNNet')
        super().__init__()
        self.scale_factor = scale_factor
        self.pretrained_net = ResNet18()
        self.final_layer = segmenthead(512,128,num_classes)

    def forward(self, x):
        x = self.pretrained_net(x)[-1]
        height = x.shape[-2] * self.scale_factor
        width = x.shape[-1] * self.scale_factor
        upsample = F.interpolate(x,size=[height, width],mode='bilinear',align_corners=False)
        score = self.final_layer(upsample)
        return [score]




