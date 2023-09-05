import os
import collections
import torch
import numpy as np
from imageio.v2 import imread
import matplotlib.pyplot as plt
from torch.utils import data
from utils.augmentations import Compose, RandomHorizontallyFlip, RandomRotate
from torchvision import transforms
import albumentations as A
import albumentations.augmentations as aug
class camvidLoader(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        img_size=[720,960],
        augmentations=None,
        img_norm=True,
        train_size = 100,
        n_classes = 12,
        convert_label_class = True,
        use_grouped_classes = True
    ):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.mean = np.array([0,0,0])
        self.n_classes = n_classes
        self.files = collections.defaultdict(list)
        self.train_size = train_size
        self.convert_label_class = convert_label_class
        self.use_grouped_classes = use_grouped_classes

        for split in ["train", "test", "val"]:
            file_list = os.listdir(root + "/" + split)
            self.files[split] = file_list[:self.train_size]

        self.norm = A.Compose([aug.transforms.Normalize(mean=[0.464, 0.475, 0.480],
                                                        std=[0.307, 0.287, 0.290],
                                                        max_pixel_value=255.0,
                                                        p=1)])
    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + "/" + self.split + "/" + img_name
        img = imread(img_path)
        img = np.array(img, dtype=np.uint8)

        if self.convert_label_class:
            if self.use_grouped_classes:
                lbl_path = self.root + "/CamVidGray/" + img_name[:-4] + "_L.png"
                lbl = imread(lbl_path)
                lbl = np.array(lbl, dtype=np.uint8)
            else:
                lbl_path = self.root + "/" + self.split + "_labels/" + img_name[:-4] + "_L.npy"
                lbl = np.load(lbl_path)
        else:
            lbl_path = self.root + "/" + self.split + "_labels/" + img_name[:-4] + "_L.png"
            lbl = imread(lbl_path)
            lbl = np.array(lbl, dtype=np.uint8)

        if self.augmentations is not None:
            augmented = self.augmentations(image=img, mask=lbl)
            img = augmented['image']
            lbl = augmented['mask']

        if self.img_norm:
            normalised = self.norm(image=img, mask=lbl)
            img = normalised['image']
            lbl = normalised['mask']
        
        img = np.transpose(img, axes=(2,0,1))

        return img, lbl

def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_batch = 0
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_size, height, width, channel = images.shape
        num_batch += batch_size
        mean += images.float().mean(axis=(0, 1, 2))
        std += images.float().std(axis=(0, 1, 2))

    mean /= num_batch
    std /= num_batch

    return mean, std

if __name__ == "__main__":
    local_path = "CamVid"
    augmentations = A.Compose([A.RandomCrop(width=960, height=720),
                               A.HorizontalFlip(p=0.5),
                               A.RandomBrightnessContrast(p=0.2)])
    dst = camvidLoader(root=local_path, convert_label_class=False,augmentations=augmentations, img_norm=True)
    bs = 2
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples
        print(data_samples)
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            print(labels)
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(labels[j])
        plt.show()
        break

    # batch_size = 1
    # loader = data.DataLoader(dst, batch_size=batch_size, shuffle=True)
    # mean, std = get_mean_std(loader)
    # print(f"mean: {mean}, std: {std}")