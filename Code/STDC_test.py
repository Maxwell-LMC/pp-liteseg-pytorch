import torch
from PIL import Image
import STDC.nets.stdcnet as stdcnet
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from STDC1_pytorch_raw.model import Model


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

stdc1 = Model()

stdc1.eval()
with torch.inference_mode():
    x = T.unsqueeze(dim=0).float()
    y = stdc1(x)
    
    for i in range(len(y)):
        print(y[i].shape)
        encoded_img = y[i].squeeze(dim=0).permute(1,2,0)
        print(encoded_img.dtype)
        plt.imshow(encoded_img[:,:,:3])
        plt.show()
torch.save(stdc1.state_dict(), "STDC1_pytorch.pth")



