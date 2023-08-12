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
    ):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.mean = np.array([0,0,0])
        self.n_classes = 11
        self.files = collections.defaultdict(list)

        for split in ["train", "test", "val"]:
            file_list = os.listdir(root + "/" + split)
            self.files[split] = file_list

        Unlabelled = (0, 0, 0)
        Sky = (128, 128, 128)
        Building = (128, 0, 0)
        Pole = (192, 192, 128)
        Road = (128, 64, 128)
        Pavement = (0, 0, 192)
        Tree = (128, 128, 0)
        Fence = (64, 64, 128)
        Car = (64, 0, 128)
        Pedestrian = (64, 64, 0)
        Bicyclist = (0, 128, 192)

        mapping = {
            Unlabelled:0,
            Sky:1,
            Building:2,
            Pole:3,
            Road:4,
            Pavement:5,
            Tree:6,
            Fence:7,
            Car:8,
            Pedestrian:9,
            Bicyclist:10,
        }

        self.label_mapping = collections.defaultdict(lambda: 0)

        for key, value in mapping.items():
            self.label_mapping[key] = value
    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + "/" + self.split + "/" + img_name
        lbl_path = self.root + "/" + self.split + "_labels/" + img_name[:-4] + "_L.png"

        img = imread(img_path)
        img = np.array(img, dtype=np.uint8)

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
        lbl = self.classify(lbl)
        img = np.transpose(img, axes=(2,0,1))
        lbl = torch.from_numpy(lbl).float()
        img = torch.from_numpy(img).float()
        return img, lbl

    def classify(self, lbl):
        masked_lbl = np.zeros(shape=(self.img_size[0],self.img_size[1]))
        for i in range(len(lbl)):
            for j in range(len(lbl[i])):
                masked_lbl[i,j] = self.label_mapping[tuple(lbl[i,j])]
        return masked_lbl

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