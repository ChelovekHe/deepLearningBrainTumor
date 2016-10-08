__author__ = 'fabian'

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from subprocess import call
import os.path as path
from multiprocessing import Pool
import os

# run convert_BraTS_2014_to_npy and bc_BraTS_2014_data first
def convert_patient_to_npy_star(args):
    return convert_patient_to_npy(*args)

def convert_patient_to_npy(folder, out_folder, id):
    print id
    if not path.isfile(path.join(folder, "%03.0d_T1_ws.nii.gz"%id)):
        return
    if not path.isfile(path.join(folder, "%03.0d_T1c_ws.nii.gz"%id)):
        return
    if not path.isfile(path.join(folder, "%03.0d_T2_ws.nii.gz"%id)):
        return
    if not path.isfile(path.join(folder, "%03.0d_Flair_ws.nii.gz"%id)):
        return
    if not path.isfile(path.join(folder, "%03.0d_segmentation.nii.gz"%id)):
        return
    t1_img = sitk.GetArrayFromImage(sitk.ReadImage(path.join(folder, "%03.0d_T1_ws.nii.gz"%id))).astype(np.float32)
    t1c_img = sitk.GetArrayFromImage(sitk.ReadImage(path.join(folder, "%03.0d_T1c_ws.nii.gz"%id))).astype(np.float32)
    t2_img = sitk.GetArrayFromImage(sitk.ReadImage(path.join(folder, "%03.0d_T2_ws.nii.gz"%id))).astype(np.float32)
    flair_img = sitk.GetArrayFromImage(sitk.ReadImage(path.join(folder, "%03.0d_Flair_ws.nii.gz"%id))).astype(np.float32)
    seg_img = sitk.GetArrayFromImage(sitk.ReadImage(path.join(folder, "%03.0d_segmentation.nii.gz"%id))).astype(np.float32)
    assert t1_img.shape == t1c_img.shape == t2_img.shape == flair_img.shape == seg_img.shape
    np.save(path.join(out_folder, "%03.0d_T1_ws.npy"%id), t1_img)
    np.save(path.join(out_folder, "%03.0d_T1c_ws.npy"%id), t1c_img)
    np.save(path.join(out_folder, "%03.0d_T2_ws.npy"%id), t2_img)
    np.save(path.join(out_folder, "%03.0d_Flair_ws.npy"%id), flair_img)
    np.save(path.join(out_folder, "%03.0d_segmentation.npy"%id), seg_img)

# there are 251 [0, 250] patients in HGG
folders = ["/media/fabian/My Book/datasets/BraTS/2014/train/HGG/", "/media/fabian/My Book/datasets/BraTS/2014/train/LGG/", "/media/fabian/My Book/datasets/BraTS/2014/test/"]
out_folders = ["/media/fabian/My Book/datasets/BraTS/2014/train/HGG/npy", "/media/fabian/My Book/datasets/BraTS/2014/train/LGG/npy", "/media/fabian/My Book/datasets/BraTS/2014/test/npy"]
for folder, out_folder in zip(folders, out_folders):
    if not path.isdir(out_folder):
        os.mkdir(out_folder)
    print folder
    pool = Pool(8)
    pool.map(convert_patient_to_npy_star, zip([folder]*255, [out_folder]*255, np.arange(255)))
    pool.close()
    pool.join()

