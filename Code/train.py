from dataloader import camvidLoader
import torch
from torch.utils import data
import utils.augmentations as aug
import matplotlib.pyplot as plt
from pytorch_liteseg import liteseg
from STDC1_pytorch_raw.model import STDC1
import os
import numpy as np


#initial setup
torch.seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"
epoch_num = 30


#initialise model
model = liteseg(num_classes=11, encoder=STDC1).to(device)
print("MODEL LOADED")
print(next(model.parameters()).device)


#load data from Camvid
local_path = os.path.dirname(os.getcwd()) + "/CamVid"
print(local_path)

# local_path = "Camvid"
#augmentations = aug.Compose([aug.RandomRotate(10), aug.RandomHorizontallyFlip(0.5)])
dst = camvidLoader(root=local_path, train_size=368, split="train", is_transform=True, augmentations=None, img_norm=True, convert_label_class=True)
bs = 8
trainloader = data.DataLoader(dst, batch_size=bs, shuffle=True)



#set loss function and optimiser
base_lr = 0.01
loss_fn = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(params=model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)

n = 0
loss_list = []

for epoch in range(epoch_num):
    if epoch == 0:
        for g in optimiser.param_groups:
            g["lr"] = base_lr / 4
    if epoch == 1:
        for g in optimiser.param_groups:
            g["lr"] = base_lr
    for batch_num, batch in enumerate(trainloader):
        optimiser.zero_grad()
        imgs, lbl = batch
        imgs = imgs.to(device)
        lbl = lbl.to(device)
        lbl_pred = model(imgs.float())
        loss = loss_fn(lbl_pred, lbl.long())
        loss.backward()
        optimiser.step()
        print(loss.item())
        loss_list.append(loss.item())
    lr = base_lr * (1-epoch/epoch_num) 
    for g in optimiser.param_groups:
        g["lr"] = lr
    
loss_list = np.asarray(loss_list)

np.save(file="/home/paperspace/Documents/Code/loss_list_1.npy", arr = loss_list)
torch.save(model.state_dict(), "/home/paperspace/Documents/Code/state_dict_1.pth")
    




