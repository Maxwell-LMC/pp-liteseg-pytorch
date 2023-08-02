import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenetv2 import Conv2dNormActivation as ConvBNReLU
from torch.ao.nn.intrinsic import ConvBn2d




###Implement SPPM and UAFM modules###

class SPPM(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 bin_sizes,
                 align_corners=False):
        super().__init__()

        self.stages = nn.ModuleList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])

        self.conv_out = ConvBNReLU(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1)

        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=size)
        conv = ConvBNReLU(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        return nn.Sequential(prior, conv)

    def forward(self, input):
        out = None
        input_shape = (input.shape)[2:]

        for stage in self.stages:
            x = stage(input)
            x = F.interpolate(
                x,
                size=input_shape,
                mode='bilinear',
                align_corners=self.align_corners)
            if out is None:
                out = x
            else:
                out += x

        out = self.conv_out(out)
        return out
    
class UAFM(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 feature_low_shape,
                 AttentionModule = 'SAM'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_low_shape = feature_low_shape
        self.upsample = nn.Upsample(size=self.feature_low_shape, mode='bilinear')

        self.conv_xy_atten = nn.Sequential(
            ConvBNReLU(
                4, 2, kernel_size=3, padding=1, bias_attr=False),
            ConvBn2d(
                2, 1, kernel_size=3, padding=1, bias_attr=False)
        )

        if AttentionModule == 'SAM':
            self.AttentionModule = self.SAM
        elif AttentionModule == 'CAM':
            self.AttentionModule = self.CAM
        else:
            raise "Attention module not found"
        

    def SAM(self, feature_up, feature_low):
        feature_cat = torch.cat(
            (torch.mean(feature_up, dim=1, keepdim=True),
             torch.max(feature_up, dim=1, keepdim=True),
             torch.mean(feature_low, dim=1, keepdim=True),
             torch.max(feature_low, dim=1, keepdim=True)
            ),
            dim=1
        )
        alpha = nn.Sigmoid(self.conv_xy_atten(feature_cat))

        return alpha

    def CAM(self):
        pass

    def forward(self, feature_high, feature_low):
        feature_up = self.upsample(feature_high)
        alpha = self.AttentionModule(feature_up, feature_low)
        feature_out = feature_up * alpha + feature_low * (1 - alpha)
        return feature_out

    
