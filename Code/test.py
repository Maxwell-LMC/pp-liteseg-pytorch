import torch
import STDC.nets.stdcnet as stdcnet

stdc2 = stdcnet.STDCNet813(pretrain_model="Code/STDC/pretrained/STDCNet813M_73.91.tar")
stdc2.eval()
with torch.inference_mode():
    x = torch.randn(1,3,512,1024)
    y = stdc2(x)
    
    for i in range(len(y)):
        print(y[i].size())

