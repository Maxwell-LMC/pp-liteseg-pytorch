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
                 feature_low_channels,
                 feature_high_channels,
                 out_channels,
                 ksize=3,
                 resize_mode='bilinear',
                 AttentionModule='SAM'):
        super().__init__()

        self.conv_x = ConvBNReLU(
            feature_low_channels, feature_high_channels, kernel_size=ksize, padding=ksize // 2, bias=False)
        self.conv_out = ConvBNReLU(
            feature_high_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.resize_mode = resize_mode

        self.conv_xy_atten_SAM = nn.Sequential(
            ConvBNReLU(
                4, 2, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=1)
        )

        self.conv_xy_atten_CAM = nn.Sequential(
            nn.Conv2d(4 * feature_high_channels,
                feature_high_channels // 2,
                kernel_size=1,
                bias=False),
            nn.BatchNorm2d(num_features= feature_high_channels // 2),
            nn.LeakyReLU(),    
            nn.Conv2d(feature_high_channels // 2, feature_high_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_high_channels)
        )

        if AttentionModule == 'SAM':
            self.AttentionModule = self.SAM
        elif AttentionModule == 'CAM':
            self.AttentionModule = self.CAM
        else:
            raise "Attention module not found"
        
    def check(self, x, y):
        assert x.ndim == 4 and y.ndim == 4
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        assert x_h >= y_h and x_w >= y_w

    def SAM(self, feature_up, feature_low):
        feature_cat = torch.cat(
            (torch.mean(feature_up, dim=1, keepdim=True),
             torch.max(feature_up, dim=1, keepdim=True).values,
             torch.mean(feature_low, dim=1, keepdim=True),
             torch.max(feature_low, dim=1, keepdim=True).values
            ),
            dim=1
        )
        alpha = torch.sigmoid(self.conv_xy_atten_SAM(feature_cat))
        return alpha

    def CAM(self, feature_up, feature_low):
        avgpool = nn.AdaptiveAvgPool2d((feature_low.shape)[2:])
        maxpool = nn.AdaptiveMaxPool2d((feature_low.shape)[2:])
        feature_cat = torch.cat(
            (avgpool(feature_up),
             maxpool(feature_up),
             avgpool(feature_low),
             maxpool(feature_low)),
             dim=1
        )

        alpha = torch.sigmoid(self.conv_xy_atten_CAM(feature_cat))
        return alpha

    def forward(self, feature_low, feature_high):
        self.check(feature_low, feature_high)
        feature_up = F.interpolate(feature_high, (feature_low.shape)[2:], mode=self.resize_mode)
        feature_low = self.conv_x(feature_low)
        alpha = self.AttentionModule(feature_up, feature_low)
        feature_out = feature_up * alpha + feature_low * (1 - alpha)
        feature_out = self.conv_out(feature_out)
        return feature_out
    
class SegHead(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = ConvBNReLU(
            in_chan,
            mid_chan,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv_out = nn.Conv2d(
            mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

class FLD(nn.Module):
    def __init__(self, encoder_out_ch, decoder_ch, resize_mode='bilinear', AttentionModule='SAM'):
        super().__init__()



        #initialise UAFMs
        self.UAFM1 = UAFM(feature_low_channels=encoder_out_ch[-2],
                          feature_high_channels=decoder_ch[-1],
                          out_channels=decoder_ch[-2],
                          resize_mode=resize_mode,
                          AttentionModule=AttentionModule)

        self.UAFM2 = UAFM(feature_low_channels=encoder_out_ch[-3],
                          feature_high_channels=decoder_ch[-2],
                          out_channels=decoder_ch[-3],
                          resize_mode=resize_mode,
                          AttentionModule=AttentionModule)
        
    def forward(self, feature_after_SPPM, feature_low_one, feature_low_two):
        feature_after_UAFM1 = self.UAFM1(feature_low_one, feature_after_SPPM)
        feature_after_UAFM2 = self.UAFM2(feature_low_two, feature_after_UAFM1)
        feat_list = [feature_after_UAFM2, feature_after_UAFM1, feature_after_SPPM]
        return feat_list


