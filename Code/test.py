import torch
from PIL import Image
import STDC.nets.stdcnet as stdcnet
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

raw_image = Image.open('Code/Test_image/city.jpg')
transform = transforms.Compose([
    transforms.PILToTensor()
])

T = transform(raw_image)

T = T.unsqueeze(0)
T = torch.nn.functional.interpolate(T,size=(512,1024), mode='bilinear')
T = T.squeeze(0)

plt.imshow(T.permute(1,2,0))
plt.show()


stdc2 = stdcnet.STDCNet813(pretrain_model="Code/STDC/pretrained/STDCNet813M_73.91.tar")

stdc2.eval()
with torch.inference_mode():
    x = T.unsqueeze(dim=0).float()
    y = stdc2(x)
    
    for i in range(len(y)):
        print(y[i].shape)
        encoded_img = y[i].squeeze(dim=0).permute(1,2,0)
        plt.imshow(encoded_img[:,:,:3])
        plt.show()



