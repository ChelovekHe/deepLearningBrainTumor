__author__ = 'fabian'
import numpy as np
import os
import sys
import SimpleITK as sitk
import matplotlib.pyplot as plt

def save_BraTS_Data_as_nifti():
    # what an ugly block of code
    folder_src = "/media/fabian/My Book/datasets/BraTS/2014/"
    folder_src_train = os.path.join(folder_src, "train")
    folder_src_test = os.path.join(folder_src, "test")
    folder_src_train_hgg = os.path.join(folder_src_train, "HGG")
    folder_src_train_lgg = os.path.join(folder_src_train, "LGG")
    for folder in os.listdir(folder_src_train_hgg):
        if os.path.isdir(os.path.join(folder_src_train_hgg, folder)):
            for folder2 in os.listdir(os.path.join(folder_src_train_hgg, folder)):
                if os.path.isdir(os.path.join(folder_src_train_hgg, folder, folder2)):
                    for file in os.listdir(os.path.join(folder_src_train_hgg, folder, folder2)):
                        if file[-4:] == ".mha":
                            print os.path.join(folder_src_train_hgg, folder, folder2, file)
                            sitk.WriteImage(sitk.ReadImage(os.path.join(folder_src_train_hgg, folder, folder2, file)), os.path.join(folder_src_train_hgg, folder, folder2, file)[:-4] + ".nii.gz")
                elif os.path.join(folder_src_train_hgg, folder, folder2)[-4:] == ".mha":
                    print os.path.join(folder_src_train_hgg, folder, folder2)
                    sitk.WriteImage(sitk.ReadImage(os.path.join(folder_src_train_hgg, folder, folder2)), os.path.join(folder_src_train_hgg, folder, folder2)[:-4]+ ".nii.gz")
    for folder in os.listdir(folder_src_train_lgg):
        if os.path.isdir(os.path.join(folder_src_train_lgg, folder)):
            for folder2 in os.listdir(os.path.join(folder_src_train_lgg, folder)):
                if os.path.isdir(os.path.join(folder_src_train_lgg, folder, folder2)):
                    for file in os.listdir(os.path.join(folder_src_train_lgg, folder, folder2)):
                        if file[-4:] == ".mha":
                            print os.path.join(folder_src_train_lgg, folder, folder2, file)
                            sitk.WriteImage(sitk.ReadImage(os.path.join(folder_src_train_lgg, folder, folder2, file)), os.path.join(folder_src_train_lgg, folder, folder2, file)[:-4] + ".nii.gz")
                elif os.path.join(folder_src_train_lgg, folder, folder2)[-4:] == ".mha":
                    print os.path.join(folder_src_train_lgg, folder, folder2)
                    sitk.WriteImage(sitk.ReadImage(os.path.join(folder_src_train_lgg, folder, folder2)), os.path.join(folder_src_train_lgg, folder, folder2)[:-4]+ ".nii.gz")
    for folder in os.listdir(folder_src_test):
        if os.path.isdir(os.path.join(folder_src_test, folder)):
            for folder2 in os.listdir(os.path.join(folder_src_test, folder)):
                if os.path.isdir(os.path.join(folder_src_test, folder, folder2)):
                    for file in os.listdir(os.path.join(folder_src_test, folder, folder2)):
                        if file[-4:] == ".mha":
                            print os.path.join(folder_src_test, folder, folder2, file)
                            sitk.WriteImage(sitk.ReadImage(os.path.join(folder_src_test, folder, folder2, file)), os.path.join(folder_src_test, folder, folder2, file)[:-4] + ".nii.gz")
                elif os.path.join(folder_src_test, folder, folder2)[-4:] == ".mha":
                    print os.path.join(folder_src_test, folder, folder2)
                    sitk.WriteImage(sitk.ReadImage(os.path.join(folder_src_test, folder, folder2)), os.path.join(folder_src_test, folder, folder2)[:-4]+ ".nii.gz")
