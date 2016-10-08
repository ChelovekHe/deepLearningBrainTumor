__author__ = 'fabian'
import numpy as np
import sys
import os.path as path
from os import mkdir
import matplotlib.pyplot as plt
sys.path.append("../")
from dataset_utility import extract_brain_region

folder = "/media/fabian/My Book/datasets/BraTS/2014/train/HGG_npy/"
folder_out = "/media/fabian/My Book/datasets/BraTS/2014/train/HGG_npy/brain_only"

if not path.isdir(folder_out):
    mkdir(folder_out)

all_shapes_HGG = []
max_values_HGG = []
min_values_HGG = []
for id in np.arange(300):
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
    flair_img = np.load(path.join(folder, "%03.0d_Flair_ws.npy"%id))
    seg_img = np.load(path.join(folder, "%03.0d_segmentation.npy"%id))
    seg_img += 1
    seg_img[t1_img == t1_img[0,0,0]] = 0
    t1_img = extract_brain_region(t1_img, seg_img, 0)
    t1c_img = extract_brain_region(t1c_img, seg_img, 0)
    t2_img = extract_brain_region(t2_img, seg_img, 0)
    flair_img = extract_brain_region(flair_img, seg_img, 0)
    seg_img = extract_brain_region(seg_img, seg_img, 0)
    np.save(path.join(folder_out, "%03.0d_Flair_ws.npy"%id), flair_img)
    np.save(path.join(folder_out, "%03.0d_segmentation.npy"%id), seg_img)
    np.save(path.join(folder_out, "%03.0d_T2_bc_ws.npy"%id), t2_img)
    np.save(path.join(folder_out, "%03.0d_T1c_bc_ws.npy"%id), t1c_img)
    np.save(path.join(folder_out, "%03.0d_T1_bc_ws.npy"%id), t1_img)
    all_shapes_HGG.append(seg_img.shape)
    max_values_HGG.append([np.max(t1_img), np.max(t1c_img), np.max(t2_img), np.max(flair_img)])
    min_values_HGG.append([np.min(t1_img), np.min(t1c_img), np.min(t2_img), np.min(flair_img)])
    np.save(path.join(folder_out, "min_values"), min_values_HGG)
    np.save(path.join(folder_out, "max_values"), max_values_HGG)
    np.save(path.join(folder_out, "all_shapes"), all_shapes_HGG)


folder = "/media/fabian/My Book/datasets/BraTS/2014/train/LGG_npy/"
folder_out = "/media/fabian/My Book/datasets/BraTS/2014/train/LGG_npy/brain_only"

if not path.isdir(folder_out):
    mkdir(folder_out)

all_shapes_LGG = []
max_values_LGG = []
min_values_LGG = []
for id in np.arange(300):
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
    flair_img = np.load(path.join(folder, "%03.0d_Flair_ws.npy"%id))
    seg_img = np.load(path.join(folder, "%03.0d_segmentation.npy"%id))
    seg_img += 1
    seg_img[t1_img == t1_img[0,0,0]] = 0
    t1_img = extract_brain_region(t1_img, seg_img, 0)
    t1c_img = extract_brain_region(t1c_img, seg_img, 0)
    t2_img = extract_brain_region(t2_img, seg_img, 0)
    flair_img = extract_brain_region(flair_img, seg_img, 0)
    seg_img = extract_brain_region(seg_img, seg_img, 0)
    np.save(path.join(folder_out, "%03.0d_Flair_ws.npy"%id), flair_img)
    np.save(path.join(folder_out, "%03.0d_segmentation.npy"%id), seg_img)
    np.save(path.join(folder_out, "%03.0d_T2_bc_ws.npy"%id), t2_img)
    np.save(path.join(folder_out, "%03.0d_T1c_bc_ws.npy"%id), t1c_img)
    np.save(path.join(folder_out, "%03.0d_T1_bc_ws.npy"%id), t1_img)
    all_shapes_LGG.append(seg_img.shape)
    max_values_LGG.append([np.max(t1_img), np.max(t1c_img), np.max(t2_img), np.max(flair_img)])
    min_values_LGG.append([np.min(t1_img), np.min(t1c_img), np.min(t2_img), np.min(flair_img)])
    np.save(path.join(folder_out, "min_values"), min_values_LGG)
    np.save(path.join(folder_out, "max_values"), max_values_LGG)
    np.save(path.join(folder_out, "all_shapes"), all_shapes_LGG)

folder = "/media/fabian/My Book/datasets/BraTS/2014/test_npy/"
folder_out = "/media/fabian/My Book/datasets/BraTS/2014/test_npy/brain_only"

if not path.isdir(folder_out):
    mkdir(folder_out)

all_shapes_test = []
max_values_test = []
min_values_test = []
for id in np.arange(300):
    if not path.isfile(path.join(folder, "%03.0d_Flair_ws.npy"%id)):
        continue
    if not path.isfile(path.join(folder, "%03.0d_T1_bc_ws.npy"%id)):
        continue
    if not path.isfile(path.join(folder, "%03.0d_T1c_bc_ws.npy"%id)):
        continue
    if not path.isfile(path.join(folder, "%03.0d_T2_bc_ws.npy"%id)):
        continue
    t1_img = np.load(path.join(folder, "%03.0d_T1_bc_ws.npy"%id))
    t1c_img = np.load(path.join(folder, "%03.0d_T1c_bc_ws.npy"%id))
    t2_img = np.load(path.join(folder, "%03.0d_T2_bc_ws.npy"%id))
    flair_img = np.load(path.join(folder, "%03.0d_Flair_ws.npy"%id))
    mask = np.zeros(t1_img.shape)
    mask[t1_img != t1_img[0,0,0]] = 1
    t1_img = extract_brain_region(t1_img, mask, 0)
    t1c_img = extract_brain_region(t1c_img, mask, 0)
    t2_img = extract_brain_region(t2_img, mask, 0)
    flair_img = extract_brain_region(flair_img, mask, 0)
    np.save(path.join(folder_out, "%03.0d_Flair_ws.npy"%id), flair_img)
    np.save(path.join(folder_out, "%03.0d_T2_bc_ws.npy"%id), t2_img)
    np.save(path.join(folder_out, "%03.0d_T1c_bc_ws.npy"%id), t1c_img)
    np.save(path.join(folder_out, "%03.0d_T1_bc_ws.npy"%id), t1_img)
    all_shapes_test.append(t1_img.shape)
    max_values_test.append([np.max(t1_img), np.max(t1c_img), np.max(t2_img), np.max(flair_img)])
    min_values_test.append([np.min(t1_img), np.min(t1c_img), np.min(t2_img), np.min(flair_img)])
    np.save(path.join(folder_out, "min_values"), min_values_test)
    np.save(path.join(folder_out, "max_values"), max_values_test)
    np.save(path.join(folder_out, "all_shapes"), all_shapes_test)

