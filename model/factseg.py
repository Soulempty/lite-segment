from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import models
from .backbone.resnet import ResNet18, ResNet50
import torch.nn.functional as F

class FPN(nn.Module):
    
    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=256):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(len(in_channels)):
            l_conv = nn.Conv2d(in_channels[i], out_channels, 1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)

    def forward(self, inputs):
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            x = inputs[i]
            y = lateral_conv(x) 
            laterals.append(y)

        used_levels = len(laterals) 
        for i in range(used_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, mode="nearest")
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_levels)]

        return outs


class AssymetricDecoder(nn.Module):
    def __init__(self,
                 in_channels=[64,128,256,512],
                 mid_channels=128,
                 out_channels=64,
                 in_feat_output_strides=(4, 8, 16, 32),
                 out_feat_output_stride=4):
        super(AssymetricDecoder, self).__init__()
        self.fpn = FPN(in_channels,mid_channels)
        self.blocks = nn.ModuleList()
        for in_feat_os in in_feat_output_strides:
            num_upsample = int(np.log2(in_feat_os) - np.log2(out_feat_output_stride))
            num_layers = max(1,num_upsample)

            self.blocks.append(nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(mid_channels if idx == 0 else out_channels, out_channels, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.UpsamplingBilinear2d(scale_factor=2) if num_upsample != 0 else nn.Identity(),
                ) for idx in range(num_layers)]))

    def forward(self, inputs):
        feat_list = self.fpn(inputs)
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(feat_list[idx])
            inner_feat_list.append(decoder_feat)

        out_feat = sum(inner_feat_list) / 4.
        return out_feat
    
class FactSeg(nn.Module):
    def __init__(self, num_classes):
        super(FactSeg, self).__init__()
        self.encoder = ResNet18()
        self.fg_decoder = AssymetricDecoder()
        self.bi_decoder = AssymetricDecoder()
        
        self.fg_cls = nn.Conv2d(64, num_classes, 1)
        self.bi_cls = nn.Conv2d(64, 1, 1)

    def forward(self, x):
       
        feat_list = self.encoder(x)
        fg_out = self.fg_decoder(feat_list)
        bi_out = self.bi_decoder(feat_list)

        fg_pred = self.fg_cls(fg_out)
        bi_pred = self.bi_cls(bi_out)

        fg_pred = F.interpolate(fg_pred, scale_factor=4.0, mode='bilinear',align_corners=True)
        bi_pred = F.interpolate(bi_pred, scale_factor=4.0, mode='bilinear',align_corners=True)

        binary_prob = torch.sigmoid(bi_pred)
        cls_prob = torch.softmax(fg_pred, dim=1)
        prob = cls_prob.clone()
        prob[:, 0, :, :] = cls_prob[:, 0, :, :] * (1- binary_prob).squeeze(dim=1)
        prob[:, 1:, :, :] = cls_prob[:, 1:, :, :] * binary_prob
        Z = torch.sum(prob, dim=1,keepdim=True)
        prob = prob.div_(Z)
        return prob


if __name__ == "__main__":
    img = torch.randn([1,3,512,512])
    model = FactSeg(4)
    output = model(img)
    print('output size:',output.size())

       




