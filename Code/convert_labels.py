import collections
import os
import numpy as np
from imageio.v2 import imread

img_size = (720,960)

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
Animal = (64, 128, 64)
Archway = (192, 0, 128)
Child = (192,128,64)
LaneMkgsDriv = (128, 0, 192)
LaneMkgsNonDriv= (192, 0, 64)


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

label_mapping = collections.defaultdict(lambda: 0)

# root = "Camvid"

# root = os.path.dirname(os.getcwd()) + "/CamVid"

# for key, value in mapping.items():
#     label_mapping[key] = value

# for split in ["val"]:
#     file_list = os.listdir(root + "/" + split)
#     label_list = os.listdir(root + "/" + split + "_labels")
#     for img_name in file_list:
#         s = img_name[:-4] + "_L.npy"
#         if s in label_list:
#             continue
#         lbl_path = root + "/" + split + "_labels/" + img_name[:-4] + "_L.png"
#         lbl = imread(lbl_path)
#         lbl = np.array(lbl, dtype=np.uint8)
#         convert = np.zeros(shape=img_size)
#         for i in range(img_size[0]):
#             for j in range(img_size[1]):
#                 convert[i,j] = label_mapping[tuple(lbl[i,j])]
#         np.save(file=(root + "/" + split + "_labels/" + img_name[:-4] + "_L.npy"), arr=convert)
#         print(lbl_path)
