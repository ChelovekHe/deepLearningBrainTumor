__author__ = 'fabian'
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../generators/")
from data_generators import load_all_patients_BraTS_2014_HGG
import os
from matplotlib.colors import ListedColormap

all_patients = load_all_patients_BraTS_2014_HGG()
out_dir = "/media/fabian/DeepLearningData/datasets/BraTS_2014/all_data_as_png"
cmap = ListedColormap([(0,0,0), (0,0,1), (0,1,0), (1,0,0), (1,1,0), (0.7, 0.7, 0.3)])

for k in all_patients.keys():
    this_out_dir = os.path.join(out_dir, "%03.0d"%k)
    if not os.path.isdir(this_out_dir):
        os.mkdir(this_out_dir)
    num_z_slices = all_patients[k]["t1"].shape[0]
    for slice_id in xrange(num_z_slices):
        seg_slice = np.array(all_patients[k]["seg"][slice_id]).astype(np.int32)
        seg_slice[0, 0:6] = [0,1,2,3,4,5]
        plt.figure(figsize=(26, 6))
        plt.subplot(1, 6, 1)
        plt.imshow(all_patients[k]["t1"][slice_id], cmap="gray")
        plt.title("t1")
        plt.subplot(1, 6, 2)
        plt.imshow(all_patients[k]["t1km"][slice_id], cmap="gray")
        plt.title("t1km")
        plt.subplot(1, 6, 3)
        plt.imshow(all_patients[k]["t2"][slice_id], cmap="gray")
        plt.title("t2")
        plt.subplot(1, 6, 4)
        plt.imshow(all_patients[k]["t2_no_bc"][slice_id], cmap="gray")
        plt.title("t2_no_bc")
        plt.subplot(1, 6, 5)
        plt.imshow(all_patients[k]["flair"][slice_id], cmap="gray")
        plt.title("flair_no_bc")
        plt.subplot(1, 6, 6)
        plt.imshow(all_patients[k]["seg"][slice_id], cmap=cmap)
        plt.title("seg_gt")
        plt.savefig(os.path.join(this_out_dir, "%03.0d_%03.0d.png"%(k, slice_id)))
        plt.close()