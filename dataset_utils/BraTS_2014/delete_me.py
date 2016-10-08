__author__ = 'fabian'
import numpy as np
import os
import sys
import SimpleITK as sitk
import matplotlib.pyplot as plt

def save_BraTS_Data_as_nifti():
    # what an ugly block of code
    name_id_correspondence = []
    folder_src = "/media/fabian/My Book/datasets/BraTS/2014/"
    folder_src_train = os.path.join(folder_src, "train")
    folder_src_train_hgg = os.path.join(folder_src_train, "HGG")
    ctr = 0
    for folder in os.listdir(folder_src_train_hgg):
        if os.path.isdir(os.path.join(folder_src_train_hgg, folder)):
            for folder2 in os.listdir(os.path.join(folder_src_train_hgg, folder)):
                if os.path.isdir(os.path.join(folder_src_train_hgg, folder, folder2)):
                    modality = None
                    if folder2.find("T1c") == -1:
                        if folder2.find("T1") != -1:
                            modality = "T1"
                        if folder2.find("Flair") != -1:
                            modality= "Flair"
                        if folder2.find("T2") != -1:
                            modality = "T2"
                    else:
                        modality = "T1c"
                    if modality is None:
                        continue
                    for file in os.listdir(os.path.join(folder_src_train_hgg, folder, folder2)):
                        if file[-4:] == ".mha":
                            print os.path.join(folder_src_train_hgg, folder, folder2, file)
                            # sitk.WriteImage(sitk.ReadImage(os.path.join(folder_src_train_hgg, folder, folder2, file)), os.path.join(folder_src_train_hgg) + "/%03.0d_%s.nii.gz" % (ctr, modality))
                    name_id_correspondence.append([ctr, folder])
                    ctr += 1
                elif os.path.join(folder_src_train_hgg, folder, folder2)[-4:] == ".mha":
                    print os.path.join(folder_src_train_hgg, folder, folder2)
                    # sitk.WriteImage(sitk.ReadImage(os.path.join(folder_src_train_hgg, folder, folder2)), os.path.join(folder_src_train_hgg) + "/%03.0d_segmentation.nii.gz" % ctr)
                    name_id_correspondence.append([ctr, folder])
                    ctr += 1
