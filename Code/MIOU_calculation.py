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
import albumentations as A
from albumentations import augmentations as aug

CLASS_NUM = 11
IMG_SIZE = (720,960,3)
TRAIN_SIZE = 369
TEST_SIZE = 232

device = "cuda" if torch.cuda.is_available() else "cpu"

# local_path = "CamVid"
local_path = os.path.dirname(os.getcwd())



model = pytorch_liteseg.liteseg(num_classes=CLASS_NUM, encoder=STDC1)
model.load_state_dict(torch.load(f=local_path + "/Code/Trained_data/state_dict_12.pth", map_location=torch.device(device)))
jaccard = JaccardIndex(num_classes=CLASS_NUM, task="multiclass", ignore_index=255)


augmentations = A.Compose([A.HorizontalFlip(p=0.5),
                           A.VerticalFlip(p=0.5),
                           aug.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue = 0, always_apply=False, p=0.5),
                           aug.crops.RandomResizedCrop(height=720, width=960, scale=(0.5,2.5), p=0.5)])

dst = camvidLoader(root=local_path + "/CamVid", train_size=TEST_SIZE, split="test", augmentations=None, img_norm=True, convert_label_class=True,
                   use_grouped_classes=True)
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
        lbl_pred = model(img.float())[0]
        lbl_pred = lbl_pred.squeeze(dim=0)
        pred_argmax = torch.argmax(lbl_pred, dim=0, keepdim=True)
        segmented_img = pred_argmax.squeeze(dim=0)
        iou = jaccard(segmented_img, lbl_true)
        print(iou)
        total_iou += iou

mean_iou = total_iou / TEST_SIZE
print("MIOU is:")
print(mean_iou)