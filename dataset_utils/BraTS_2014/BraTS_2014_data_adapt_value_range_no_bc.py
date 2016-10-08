__author__ = 'fabian'
import numpy as np
import os.path as path
from os import mkdir

folders = ["/media/fabian/My Book/datasets/BraTS/2014/train/HGG/npy/brain_only/", "/media/fabian/My Book/datasets/BraTS/2014/train/LGG/npy/brain_only/", "/media/fabian/My Book/datasets/BraTS/2014/test/npy/brain_only/"]
folders_out = ["/media/fabian/My Book/datasets/BraTS/2014/train/HGG/npy/brain_only/adapted_value_range_no_bc", "/media/fabian/My Book/datasets/BraTS/2014/train/LGG/npy/brain_only/adapted_value_range_no_bc", "/media/fabian/My Book/datasets/BraTS/2014/test/npy/brain_only/adapted_value_range_no_bc"]
folders_out2 = ["/media/fabian/My Book/datasets/BraTS/2014/train/HGG/npy/brain_only/original_value_range_no_bc", "/media/fabian/My Book/datasets/BraTS/2014/train/LGG/npy/brain_only/original_value_range_no_bc", "/media/fabian/My Book/datasets/BraTS/2014/test/npy/brain_only/original_value_range_no_bc"]

percentile_target = [-10., 10.]

for folder, folder_out, folder_out2 in zip(folders, folders_out, folders_out2):
    if not path.isdir(folder_out):
        mkdir(folder_out)
    if not path.isdir(folder_out2):
        mkdir(folder_out2)

    for id in range(300):
        if not path.isfile(path.join(folder, "%03.0d_Flair_ws.npy"%id)):
            continue
        if not path.isfile(path.join(folder, "%03.0d_T1_ws.npy"%id)):
            continue
        if not path.isfile(path.join(folder, "%03.0d_T1c_ws.npy"%id)):
            continue
        if not path.isfile(path.join(folder, "%03.0d_T2_ws.npy"%id)):
            continue
        if not path.isfile(path.join(folder, "%03.0d_segmentation.npy"%id)):
            continue
        t1_img = np.load(path.join(folder, "%03.0d_T1_ws.npy"%id))
        t1c_img = np.load(path.join(folder, "%03.0d_T1c_ws.npy"%id))
        t2_img = np.load(path.join(folder, "%03.0d_T2_ws.npy"%id))
        flair_img = np.load(path.join(folder, "%03.0d_Flair_ws.npy"%id))
        seg_img = np.load(path.join(folder, "%03.0d_segmentation.npy"%id))

        all_data = np.zeros((5, t1_img.shape[0], t1_img.shape[1], t1_img.shape[2]), dtype=np.float32)
        all_data[0] = t1_img
        all_data[1] = t1c_img
        all_data[2] = t2_img
        all_data[3] = flair_img
        all_data[4] = seg_img
        np.save(path.join(folder_out2, "%03.0d.npy"%id), all_data)

        percentile_actual = [np.percentile(t1c_img[t1c_img!=t1c_img[0,0,0]].ravel(), 35), np.percentile(t1c_img[t1c_img!=t1c_img[0,0,0]].ravel(), 65)]
        t1c_img -= percentile_actual[0]
        t1c_img *= (percentile_target[1] - percentile_target[0]) / (percentile_actual[1] - percentile_actual[0])

        percentile_actual = [np.percentile(t1_img[t1_img!=t1_img[0,0,0]].ravel(), 35), np.percentile(t1_img[t1_img!=t1_img[0,0,0]].ravel(), 65)]
        t1_img -= percentile_actual[0]
        t1_img *= (percentile_target[1] - percentile_target[0]) / (percentile_actual[1] - percentile_actual[0])

        percentile_actual = [np.percentile(t2_img[t2_img!=t2_img[0,0,0]].ravel(), 35), np.percentile(t2_img[t2_img!=t2_img[0,0,0]].ravel(), 65)]
        t2_img -= percentile_actual[0]
        t2_img *= (percentile_target[1] - percentile_target[0]) / (percentile_actual[1] - percentile_actual[0])

        percentile_actual = [np.percentile(flair_img[flair_img!=flair_img[0,0,0]].ravel(), 35), np.percentile(flair_img[flair_img!=flair_img[0,0,0]].ravel(), 65)]
        flair_img -= percentile_actual[0]
        flair_img *= (percentile_target[1] - percentile_target[0]) / (percentile_actual[1] - percentile_actual[0])

        all_data = np.zeros((5, t1_img.shape[0], t1_img.shape[1], t1_img.shape[2]), dtype=np.float32)
        all_data[0] = t1_img
        all_data[1] = t1c_img
        all_data[2] = t2_img
        all_data[3] = flair_img
        all_data[4] = seg_img
        np.save(path.join(folder_out, "%03.0d.npy"%id), all_data)





import matplotlib.pyplot as plt
for folder_out in folders_out+folders_out2:
    for id in range(300):
        if not path.isfile(path.join(folder_out, "%03.0d.npy"%id)):
            continue
        data_all = np.load(path.join(folder_out, "%03.0d.npy"%id))
        t1_img = data_all[0]
        t1c_img = data_all[1]
        t2_img = data_all[2]
        flair_img = data_all[3]
        seg_img = data_all[4]
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 4, 1)
        plt.hist(t1_img[t1_img!=t1_img[0,0,0]], bins=100)
        plt.xlim((-200, 200))
        plt.subplot(1, 4, 2)
        plt.hist(t1c_img[t1c_img!=t1c_img[0,0,0]], bins=100)
        plt.xlim((-200, 200))
        plt.subplot(1, 4, 3)
        plt.hist(t2_img[t2_img!=t2_img[0,0,0]], bins=100)
        plt.xlim((-200, 200))
        plt.subplot(1, 4, 4)
        plt.hist(flair_img[flair_img!=flair_img[0,0,0]], bins=100)
        plt.xlim((-200, 200))
        plt.savefig(path.join(folder_out, "%03.0d_hist.png"%id))
        plt.close()




