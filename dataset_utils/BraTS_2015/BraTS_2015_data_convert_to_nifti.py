__author__ = 'fabian'
import numpy as np
import os
import sys
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cPickle

def save_BraTS_Data_as_nifti():
    folder_src = "/media/fabian/My Book/datasets/BraTS/2015/"
    folder_src_train = os.path.join(folder_src, "train")
    folder_src_test = os.path.join(folder_src, "test")
    folder_src_test_out = os.path.join(folder_src, "test_nifti")
    folder_src_train_hgg = os.path.join(folder_src_train, "HGG")
    folder_src_train_hgg_out = os.path.join(folder_src_train, "HGG_nifti")
    folder_src_train_lgg = os.path.join(folder_src_train, "LGG")
    folder_src_train_lgg_out = os.path.join(folder_src_train, "LGG_nifti")
    input_dirs = [folder_src_train_hgg, folder_src_train_lgg, folder_src_test]
    output_dirs = [folder_src_train_hgg_out, folder_src_train_lgg_out, folder_src_test_out]
    for data_dir, out_dir in zip(input_dirs, output_dirs):
        ctr = 0
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        name_id_conversion = []
        for folder in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, folder)):
                for folder2 in os.listdir(os.path.join(data_dir, folder)):
                    if os.path.isdir(os.path.join(data_dir, folder, folder2)):
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
                        for file in os.listdir(os.path.join(data_dir, folder, folder2)):
                            if file[-4:] == ".mha":
                                print os.path.join(data_dir, folder, folder2, file)
                                sitk.WriteImage(sitk.GetImageFromArray(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_dir, folder, folder2, file))).astype(np.float32)), os.path.join(out_dir) + "/%03.0d_%s.nii.gz" % (ctr, modality))
                    elif os.path.join(data_dir, folder, folder2)[-4:] == ".mha":
                        print os.path.join(data_dir, folder, folder2)
                        sitk.WriteImage(sitk.ReadImage(os.path.join(data_dir, folder, folder2)), os.path.join(out_dir) + "/%03.0d_segmentation.nii.gz" % ctr)
                name_id_conversion.append([ctr, folder])
                ctr += 1
        with open(os.path.join(out_dir, "id_name_conversion.txt"), 'w') as f:
            for datum in name_id_conversion:
                f.write("%03.0d:\t%s\n" % (datum[0], datum[1]))

if __name__ == "__main__":
    save_BraTS_Data_as_nifti()