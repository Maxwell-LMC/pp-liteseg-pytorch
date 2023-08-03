import torch
import torch.nn as nn
from module import SPPM
from module import FLD


#using stdc1
class liteseg(nn.Module):
    def __init__(self,
                 num_classes, 
                 encoder, 
                 encoder_layers=[2,3,4],
                 encoder_out_ch=[256,512,1024],
                 attention_module="SAM",
                 sppm_bin_sizes=[1,2,4],
                 sppm_out_ch=128,
                 decoder_ch=[32,64,128],
                 seg_head_mid_ch = 64,
                 resize_mode='bilinear',
                 pretrained=None):
        super().__init__()

        #initialise constants
        self.num_classes = num_classes
        self.encoder_layers = encoder_layers
        self.encoder_out_ch = encoder_out_ch
        self.attention_module = attention_module
        self.sppm_bin_sizes = sppm_bin_sizes
        self.sppm_out_ch = sppm_out_ch
        self.decoder_ch = decoder_ch
        self.seg_head_mid_ch = seg_head_mid_ch
        self.resize_mode = resize_mode
        self.pretrained = pretrained

        #initialise encoder, and freezing it
        self.encoder = encoder()
        for p in self.encoder.parameters():
            p.require_grad = False

        #initialise SPPM
        self.SPPM = SPPM(in_channels=self.encoder_out_ch[-1],
                         inter_channels=self.decoder_ch[-1],
                         out_channels=self.decoder_ch[-1],
                         bin_sizes=self.sppm_bin_sizes)

        #initialise FLD
        self.FLD = FLD(feature_low_ch1=self.encoder_out_ch[-2],
                       feature_high_ch1=self.decoder_ch[-1],
                       feature_low_ch2=self.encoder_out_ch[-3],
                       feature_high_ch2=self.decoder_ch[-2],
                       out_ch1=self.decoder_ch[-2],
                       out_ch2=self.decoder_ch[-3],
                       seg_head_mid_ch=self.seg_head_mid_ch,
                       num_classes=self.num_classes)
        
    def forward(self, feature):
        original_shape = (feature.shape)[2:]
        encoded_feature = self.encoder(feature)
        feature_after_SPPM = self.SPPM(encoded_feature[self.encoder_layers[-1]])
        feature_after_FLD = self.FLD(feature_after_SPPM,
                                     encoded_feature[self.encoder_layers[-2]],
                                     encoded_feature[self.encoder_layers[-3]])
        reshaped_prediction = torch.nn.functional.interpolate(feature_after_FLD,size=original_shape, mode='bilinear', align_corners=False)
        return reshaped_prediction



        
