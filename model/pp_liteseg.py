from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ResNet18,STDCNet1,STDCNet2
from torchsummary import summary

class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, bias=False, with_act=True):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)     
        self.with_act = with_act
        if self.with_act:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn(self.conv(x))
        if self.with_act:
            out = self.relu(out)
        return out
    
class PPLiteSeg(nn.Module):
    def __init__(self,
                 num_classes, 
                 backbone_indices=[2, 3, 4],
                 backbone_out_chs=[256,512,1024],
                 cm_bin_sizes=[1, 2, 4],
                 cm_out_ch=128,
                 uafm_out_chs=[32, 64, 128],
                 fpn_inter_chs=[32, 64, 64],
                 resize_mode='bilinear',   
                 pretrained=True):
        super().__init__()
        
        self.backbone = STDCNet1(pretrained=True) # f2,f4,f8,f16,f32
        self.backbone_indices = backbone_indices  # [..., x16_id, x32_id]
        self.backbone_out_chs = backbone_out_chs
        self.ppfpn = PPFPN(backbone_out_chs, uafm_out_chs, cm_bin_sizes, cm_out_ch, resize_mode)

        self.seg_heads = nn.ModuleList()  
        for in_ch, mid_ch in zip(uafm_out_chs, fpn_inter_chs):
            self.seg_heads.append(SegHead(in_ch, mid_ch, num_classes))

        self.pretrained = pretrained

    def forward(self, x):
        size = x.size()[2:]
        
        feats = self.backbone(x)  # [x2, x4, x8, x16, x32]
        feats_selected = [feats[i] for i in self.backbone_indices]
        feats_head = self.ppfpn(feats_selected)  # [..., x8, x16, x32]
        logit_list = []
        if self.training:
            for x, seg_head in zip(feats_head, self.seg_heads):
                x = seg_head(x)
                x = F.interpolate(x, size, mode='bilinear', align_corners=False)
                logit_list.append(x)
        else:
            x = self.seg_heads[0](feats_head[0])
            x = F.interpolate(x, size, mode='bilinear', align_corners=False)
            logit_list = [x]

        return logit_list

class PPFPN(nn.Module):

    def __init__(self, backbone_out_chs=[256,512,1024], uafm_out_chs=[32, 64, 128], cm_bin_sizes=[1, 2, 4], cm_out_ch=128, resize_mode='bilinear'):
        super().__init__()

        self.cm = PPContextModule(backbone_out_chs[-1], cm_out_ch, cm_out_ch,cm_bin_sizes)
        self.uafm_list = nn.ModuleList()  
        for i in range(len(backbone_out_chs)): # [256,512,1024] 
            low_chs = backbone_out_chs[i]
            high_ch = cm_out_ch if i == len(backbone_out_chs) - 1 else uafm_out_chs[i + 1]
            out_ch = uafm_out_chs[i]
            arm = UAFM(low_chs, high_ch, out_ch, ksize=3, resize_mode=resize_mode)
            self.uafm_list.append(arm)

    def forward(self, in_feat_list): # [1/8,1/16,1/32]
      
        high_feat = self.cm(in_feat_list[-1])
        out_feat_list = []

        for i in reversed(range(len(in_feat_list))): # 210 []
            low_feat = in_feat_list[i]
            arm = self.uafm_list[i]
            high_feat = arm(low_feat, high_feat)
            out_feat_list.insert(0, high_feat)

        return out_feat_list


class UAFM(nn.Module):
    """
    The Unified Attention Fusion Module.
   
    """

    def __init__(self, low_ch, high_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()
        self.conv_x = ConvX(low_ch, high_ch, ksize)
        self.conv_atten = nn.Sequential(
            ConvX(4, 2, 3),
            ConvX(2, 1, 3,with_act=False))
        self.conv_out = ConvX(high_ch, out_ch, 3)
        self.resize_mode = resize_mode

    def channel_reduce(self,x,y):
        mean_x = torch.mean(x, axis=1, keepdim=True)
        max_x = torch.max(x, axis=1, keepdim=True)[0]
        mean_y = torch.mean(y, axis=1, keepdim=True)
        max_y = torch.max(y, axis=1, keepdim=True)[0]
        out = torch.cat([mean_x,max_x,mean_y,max_y],1)
        return out
    
    def forward(self, low_feature, high_feature):
        x = self.conv_x(low_feature)
        y = F.interpolate(high_feature, x.size()[2:], mode=self.resize_mode)
        atten_feature = self.channel_reduce(x,y)
        atten = F.sigmoid(self.conv_atten(atten_feature))
        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out
           
class PPContextModule(nn.Module):
    """
    Simple Context module.

    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 bin_sizes,
                 align_corners=False):
        super().__init__()

        self.stages = nn.ModuleList([self._make_stage(in_channels, inter_channels, size)for size in bin_sizes])

        self.conv_out = ConvX(inter_channels,out_channels,3)
        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):

        prior = nn.AdaptiveAvgPool2d(output_size=size)
        conv = ConvX(in_channels, out_channels, 1)
        return nn.Sequential(prior, conv)

    def forward(self, input):
        out = None
        height, width = input.size()[2:]

        for stage in self.stages:
            x = stage(input)
            x = F.interpolate(x,size=[height, width],mode='bilinear',align_corners=self.align_corners)
            if out is None:
                out = x
            else:
                out += x
        out = self.conv_out(out)
        return out
    
class SegHead(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = ConvX(in_chan,mid_chan,3)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x
    
if __name__ == "__main__":
    img = torch.randn([1,3,512,512]).cuda()
    model = PPLiteSeg(num_classes=19).cuda()
    model.eval()
    out = model(img)
    print(out[0].size())

