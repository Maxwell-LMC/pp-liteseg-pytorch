import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenetv2 import Conv2dNormActivation as ConvBNReLU





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
                 feature_high_channels,
                 feature_low_channels,
                 feature_high_shape,
                 feature_low_shape,
                 out_channels,
                 AttentionModule='SAM'):
        super().__init__()

        self.upsample = nn.Upsample(size=feature_low_shape, mode='bilinear')
        if AttentionModule == 'SAM':
            self.AttentionModule = self.SAM()
        elif AttentionModule == 'CAM':
            self.AttentionModule = self.CAM()
        else:
            raise "Attention module not found"
        

    def SAM(self):
        pass

    def CAM(self):
        pass

    def forward(self, feature_high, feature_low):
        feature_up = self.upsample(feature_high)
        alpha = self.AttentionModule(feature_high, feature_low)
        feature_out = feature_up * alpha + feature_low * (1 - alpha)
        return feature_out

    
