__author__ = 'fabian'
import numpy as np
import sys
import os.path as path
from os import mkdir
import matplotlib.pyplot as plt
sys.path.append("../")
from dataset_utility import extract_brain_region

folders_in = ["/media/fabian/My Book/datasets/BraTS/2015/train/HGG_npy/", "/media/fabian/My Book/datasets/BraTS/2015/train/LGG_npy/", "/media/fabian/My Book/datasets/BraTS/2015/test_npy/"]
folders_out = ["/media/fabian/My Book/datasets/BraTS/2015/train/HGG_npy/brain_only", "/media/fabian/My Book/datasets/BraTS/2015/train/LGG_npy/brain_only", "/media/fabian/My Book/datasets/BraTS/2015/test_npy/brain_only"]

for folder, folder_out in zip(folders_in, folders_out):
    if not path.isdir(folder_out):
        mkdir(folder_out)

    for id in np.arange(999):
        if not path.isfile(path.join(folder, "%03.0d_Flair_ws.npy"%id)):
            continue
        if not path.isfile(path.join(folder, "%03.0d_T1_bc_ws.npy"%id)):
            continue
        if not path.isfile(path.join(folder, "%03.0d_T1c_bc_ws.npy"%id)):
            continue
        if not path.isfile(path.join(folder, "%03.0d_T2_bc_ws.npy"%id)):
            continue
        if not path.isfile(path.join(folder, "%03.0d_segmentation.npy"%id)):
            continue
        t1_img = np.load(path.join(folder, "%03.0d_T1_bc_ws.npy"%id))
        t1c_img = np.load(path.join(folder, "%03.0d_T1c_bc_ws.npy"%id))
        t2_img = np.load(path.join(folder, "%03.0d_T2_bc_ws.npy"%id))
        t2_img_no_bc = np.load(path.join(folder, "%03.0d_T2_ws.npy"%id))
        flair_img = np.load(path.join(folder, "%03.0d_Flair_ws.npy"%id))
        seg_img = np.load(path.join(folder, "%03.0d_segmentation.npy"%id))
        seg_img += 1
        seg_img[t1_img == t1_img[0,0,0]] = 0
        t1_img = extract_brain_region(t1_img, seg_img, 0)
        t1c_img = extract_brain_region(t1c_img, seg_img, 0)
        t2_img = extract_brain_region(t2_img, seg_img, 0)
        t2_img_no_bc = extract_brain_region(t2_img_no_bc, seg_img, 0)
        flair_img = extract_brain_region(flair_img, seg_img, 0)
        seg_img = extract_brain_region(seg_img, seg_img, 0)
        np.save(path.join(folder_out, "%03.0d_Flair_ws.npy"%id), flair_img)
        np.save(path.join(folder_out, "%03.0d_segmentation.npy"%id), seg_img)
        np.save(path.join(folder_out, "%03.0d_T2_bc_ws.npy"%id), t2_img)
        np.save(path.join(folder_out, "%03.0d_T2_ws.npy"%id), t2_img_no_bc)
        np.save(path.join(folder_out, "%03.0d_T1c_bc_ws.npy"%id), t1c_img)
        np.save(path.join(folder_out, "%03.0d_T1_bc_ws.npy"%id), t1_img)

