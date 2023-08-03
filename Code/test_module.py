import module as m
import torch

# UAFM1 = m.UAFM(feature_low_channels=4,
#                feature_high_channels=8,
#                out_channels=4)

SPPM1 = m.SPPM(in_channels=1024, inter_channels=128, out_channels=128, bin_sizes=[1,2,4])

SPPM1.eval()
with torch.inference_mode():
    x = torch.randn((1,1024,16,32))
    y = SPPM1(x)
    print(y.shape)

UAFM1 = m.UAFM(feature_low_channels=512, feature_high_channels=128, out_channels=64, AttentionModule="SAM")

UAFM1.eval()
with torch.inference_mode():
    x = torch.randn((1, 512, 32, 64))
    y = torch.randn((1, 128, 16, 32))
    z = UAFM1(x, y)
    print(z.shape)