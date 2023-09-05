from dataloader import camvidLoader
import torch
from torch.utils import data
import matplotlib.pyplot as plt
from pytorch_liteseg import liteseg
from STDC1_pytorch_raw.model import STDC1
import os
import numpy as np
from utils.OHEMloss import OhemCrossEntropyLoss
import albumentations as A
from albumentations import augmentations as aug


#initial setup
torch.seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"
epoch_num = 168


#initialise model
model = liteseg(num_classes=11, encoder=STDC1).to(device)
print("MODEL LOADED")
print(next(model.parameters()).device)

#load data from Camvid

# FOR LINUX
local_path = os.path.dirname(os.getcwd()) + "/CamVid"

# FOR WINDOWS
# local_path = "Camvid"

print(local_path)

augmentations = A.Compose([A.HorizontalFlip(p=0.5),
                           A.VerticalFlip(p=0.5),
                           aug.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue = 0, always_apply=False, p=0.5),
                           aug.crops.RandomResizedCrop(height=720, width=960, scale=(0.5,2.5), p=0.5)])
dst = camvidLoader(root=local_path, train_size=366, split="train", augmentations=augmentations, img_norm=True, convert_label_class=True)
bs = 6
trainloader = data.DataLoader(dst, batch_size=bs, shuffle=True)
dst_test = camvidLoader(root=local_path, train_size=100, split="test", augmentations=None, img_norm=True, convert_label_class=True)
testloader = data.DataLoader(dst_test, batch_size=1, shuffle=False)

#set loss function and optimiser
base_lr = 0.01
warm_up_lr = 1e-5
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
OHEM_loss_fn = OhemCrossEntropyLoss(min_kept=250000, ignore_index=255)
optimiser = torch.optim.SGD(params=model.parameters(), lr=warm_up_lr, momentum=0.9, weight_decay=5e-4)

n = 0
train_loss_list = []
test_loss_list = []


for epoch in range(epoch_num):
    model.train()
    if epoch == 4:
        for g in optimiser.param_groups:
            g["lr"] = base_lr
            print('changed lr to {}'.format(g["lr"]))
    for batch_num, batch in enumerate(trainloader):
        optimiser.zero_grad()
        imgs, lbl = batch
        imgs = imgs.to(device)
        lbl = lbl.to(device)
        lbl_pred = model(imgs.float())
        # print(type(lbl))
        # print(type(lbl_pred))
        # print(lbl)
        # print(lbl_pred.shape)
        loss = OHEM_loss_fn(lbl_pred, lbl.long())
        loss.backward()
        optimiser.step()
        print('train loss is:{}'.format(loss.item()))
        train_loss_list.append(loss.item())
    if epoch >= 4:
        lr = base_lr * (1-(epoch-3)/(epoch_num-3)) ** 0.9
        for g in optimiser.param_groups:
            g["lr"] = lr
    
    print('testing model...')
    # test loop
    model.eval()
    with torch.inference_mode():
        test_loss = 0
        for batch_num, batch in enumerate(testloader):
            img, lbl_true = batch
            img = img.to(device)
            lbl_true = lbl_true.to(device)
            lbl_pred = model(img.float())
            loss = loss_fn(lbl_pred, lbl_true.long())
            test_loss += loss.item()
        test_loss /= 100
        print('test loss after epoch {} is:{}'.format(epoch+1, test_loss))
        test_loss_list.append(test_loss)
        
train_loss_list = np.asarray(train_loss_list)
test_loss_list = np.asarray(test_loss_list)

np.save(file="/home/paperspace/Documents/Code/Trained_data/train_loss_list_8.npy", arr = train_loss_list)
np.save(file="/home/paperspace/Documents/Code/Trained_data/test_loss_list_8.npy", arr = test_loss_list)
torch.save(model.state_dict(), "/home/paperspace/Documents/Code/Trained_data/state_dict_8.pth")
    




