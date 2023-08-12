from dataloader import camvidLoader
import torch
from torch.utils import data
import utils.augmentations as aug
import matplotlib.pyplot as plt
from pytorch_liteseg import liteseg
from STDC1_pytorch_raw.model import STDC1

torch.seed = 42
model = liteseg(num_classes=11, encoder=STDC1)
print("MODEL LOADED")
local_path = "CamVid"
# augmentations = aug.Compose([aug.RandomRotate(10), aug.RandomHorizontallyFlip(0.5)])
dst = camvidLoader(root=local_path, train_size=50, split="train", is_transform=True, augmentations=None, img_norm=True)
bs = 2
trainloader = data.DataLoader(dst, batch_size=bs, shuffle=True)

loss_fn = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(params=model.parameters(), lr=0.01)

#test training (one epoch)
for batch_num, batch in enumerate(trainloader):
    optimiser.zero_grad()
    imgs, lbl = batch
    lbl_pred = model(imgs.float())
    print(lbl_pred.shape)
    loss = loss_fn(lbl_pred, lbl.long())
    loss.backward()
    optimiser.step()
    print(loss.item())
    