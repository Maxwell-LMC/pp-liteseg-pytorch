import pytorch_liteseg
from STDC1_pytorch_raw.model import STDC1
import torch
import torch.onnx
import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread
from convert_labels import mapping

device = "cuda" if torch.cuda.is_available() else "cpu"

loss_list = np.load(file="Code/Trained_data/loss_list_1.npy")
iter_list = np.linspace(1, len(loss_list), num=len(loss_list))
plt.scatter(x=iter_list, y=loss_list, marker="o", s=2)
plt.ylabel("loss")
plt.xlabel("iterations")
plt.yticks(np.linspace(0,3,31))
plt.show()

img_size = (720,960,3)

model = pytorch_liteseg.liteseg(num_classes=11, encoder=STDC1)
model.load_state_dict(torch.load(f="Code/Trained_data/state_dict_1.pth", map_location=torch.device(device)))
img_path = "CamVid/test/0006R0_f02850.png"
img = imread(img_path)
img = np.array(img, dtype=np.uint8)
img = np.transpose(img, axes=(2,0,1))
img_tensor = torch.from_numpy(img)
img_tensor = img_tensor.unsqueeze(dim = 0)
print(img_tensor.shape)
model.eval()
with torch.inference_mode():
    pred = model(img_tensor.float())
    pred = pred.squeeze(dim=0)
    pred_argmax = torch.argmax(pred, dim=0, keepdim=True)
    segmented_img = pred_argmax.squeeze(dim=0)

reverse_mapping = {}
for key, value in mapping.items():
    reverse_mapping[value] = key

converted_seg = np.zeros(shape=img_size)
for i in range(img_size[0]):
    for j in range(img_size[1]):
        for k in range(img_size[2]):
            converted_seg[i,j,k] = reverse_mapping[int(segmented_img[i,j])][k]

print(converted_seg.shape)

plt.imshow(converted_seg.astype(int))
plt.show()
