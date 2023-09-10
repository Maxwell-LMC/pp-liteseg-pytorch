import torch
import torch.nn as nn
from module import SPPM
from module import FLD
from module import SegHead
import torch.nn.functional as F


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
                 seg_head_mid_chs = [32,64,64],
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
        self.seg_head_mid_chs = seg_head_mid_chs
        self.resize_mode = resize_mode
        self.pretrained = pretrained

        #initialise encoder, and freezing it
        self.encoder = encoder()
        for p in self.encoder.parameters():
            p.require_grad = True

        #initialise SPPM
        self.SPPM = SPPM(in_channels=self.encoder_out_ch[-1],
                         inter_channels=self.decoder_ch[-1],
                         out_channels=self.decoder_ch[-1],
                         bin_sizes=self.sppm_bin_sizes)

        #initialise FLD
        self.FLD = FLD(encoder_out_ch = self.encoder_out_ch,
                       decoder_ch = self.decoder_ch)
        
        #initialise segheads
        self.segheads = nn.ModuleList()
        assert len(decoder_ch) == len(seg_head_mid_chs)
        for in_ch, mid_ch in zip(decoder_ch, seg_head_mid_chs):
            self.segheads.append(SegHead(in_ch, mid_ch, num_classes))
        
    def forward(self, feature):
        original_shape = (feature.shape)[2:]
        encoded_feature = self.encoder(feature)
        feature_after_SPPM = self.SPPM(encoded_feature[self.encoder_layers[-1]])
        feat_list = self.FLD(feature_after_SPPM,
                            encoded_feature[self.encoder_layers[-2]],
                            encoded_feature[self.encoder_layers[-3]])
        if self.training:
            logit_list = []

            for x, seg_head in zip(feat_list, self.segheads):
                x = seg_head(x)
                logit_list.append(x)
            
            logit_list = [
                F.interpolate(
                    x, original_shape, mode='bilinear', align_corners=False)
                for x in logit_list
            ]
        else:
            x = self.segheads[0](feat_list[0])
            x = F.interpolate(x, original_shape, mode='bilinear', align_corners=False)
            logit_list = [x]


        return logit_list



        
