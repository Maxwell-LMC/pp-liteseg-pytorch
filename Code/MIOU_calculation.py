import numpy as np
import torch
from torchmetrics import JaccardIndex
import pytorch_liteseg
from STDC1_pytorch_raw.model import STDC1
import matplotlib.pyplot as plt
from dataloader import camvidLoader
from torch.utils import data
from convert_labels import mapping
import os

CLASS_NUM = 11
IMG_SIZE = (720,960,3)

device = "cuda" if torch.cuda.is_available() else "cpu"

# local_path = "CamVid"
local_path = os.path.dirname(os.getcwd())

model = pytorch_liteseg.liteseg(num_classes=11, encoder=STDC1)
model.load_state_dict(torch.load(f=local_path + "/Code/Trained_data/state_dict_1.pth", map_location=torch.device(device)))
jaccard = JaccardIndex(num_classes=CLASS_NUM, task="multiclass")


dst = camvidLoader(root=local_path + "/CamVid", train_size=232, split="test", is_transform=True, augmentations=None, img_norm=True, convert_label_class=True)
bs = 1
testloader = data.DataLoader(dst, batch_size=bs, shuffle=False)

total_iou = 0

reverse_mapping = {}
for key, value in mapping.items():
    reverse_mapping[value] = key

model.eval()
with torch.inference_mode():
    for batch_num, batch in enumerate(testloader):
        img, lbl_true = batch
        lbl_true = lbl_true[0]
        lbl_pred = model(img.float())
        lbl_pred = lbl_pred.squeeze(dim=0)
        pred_argmax = torch.argmax(lbl_pred, dim=0, keepdim=True)
        segmented_img = pred_argmax.squeeze(dim=0)
        iou = jaccard(segmented_img, lbl_true)
        print(iou)
        total_iou += iou
        # converted_seg = np.zeros(shape=IMG_SIZE)
        # converted_lbl = np.zeros(shape=IMG_SIZE)
        # for i in range(IMG_SIZE[0]):
        #     for j in range(IMG_SIZE[1]):
        #         for k in range(IMG_SIZE[2]):
        #             converted_lbl[i,j,k] = reverse_mapping[int(lbl_true[i,j])][k]
        #             converted_seg[i,j,k] = reverse_mapping[int(segmented_img[i,j])][k]

        # plt.imshow(converted_lbl.astype(int))
        # plt.show()
        # plt.imshow(converted_seg.astype(int))
        # plt.show()

mean_iou = total_iou / 232
print("MIOU is:")
print(mean_iou)