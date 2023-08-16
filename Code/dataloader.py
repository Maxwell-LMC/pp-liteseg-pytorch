import os
import collections
import torch
import numpy as np
from imageio.v2 import imread
import matplotlib.pyplot as plt
from torch.utils import data
from utils.augmentations import Compose, RandomHorizontallyFlip, RandomRotate


class camvidLoader(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=[720,960],
        augmentations=None,
        img_norm=True,
        train_size = 100,
        n_classes = 11,
        convert_label_class = True
    ):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.mean = np.array([0,0,0])
        self.n_classes = n_classes
        self.files = collections.defaultdict(list)
        self.train_size = train_size
        self.convert_label_class = convert_label_class

        for split in ["train", "test", "val"]:
            file_list = os.listdir(root + "/" + split)
            self.files[split] = file_list[:self.train_size]

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + "/" + self.split + "/" + img_name
        img = imread(img_path)
        img = np.array(img, dtype=np.uint8)

        if self.convert_label_class:
            lbl_path = self.root + "/" + self.split + "_labels/" + img_name[:-4] + "_L.npy"
            lbl = np.load(lbl_path)
        else:
            lbl_path = self.root + "/" + self.split + "_labels/" + img_name[:-4] + "_L.png"
            lbl = imread(lbl_path)
            lbl = np.array(lbl, dtype=np.uint8)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        if self.img_norm:
            #PERFORM NORMALISATION
            pass
        img = np.transpose(img, axes=(2,0,1))
        return img, lbl

if __name__ == "__main__":
    local_path = "CamVid"
    augmentations = Compose([RandomRotate(10), RandomHorizontallyFlip(0.5)])
    dst = camvidLoader(root=local_path, is_transform=True, augmentations=None, img_norm=False)
    bs = 2
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples
        # imgs = np.transpose(imgs, [0, 1,2,3])
        # f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            print(type(imgs[j]), type(labels[j]))
            # print(imgs[j], labels[j])
            # print(imgs[j].shape, labels[j].shape)
            # axarr[j][0].imshow(imgs[j])
            # axarr[j][1].imshow(labels[j])
        # plt.show()
        break