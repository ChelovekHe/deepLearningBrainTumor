__author__ = 'fabian'

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from subprocess import call
import os.path as path
from multiprocessing import Pool


def run_ws_star(args):
    return run_ws(*args)

def run_ws(t1_img, input, output):
    cmd = ["Rscript", "/media/fabian/My Book/datasets/BraTS/whitestripe_hybrid_noMask.R", t1_img, input, output]
    call(cmd)

# run convert_BraTS_2014_to_npy and bc_BraTS_2014_data first

# there are 251 [0, 250] patients in HGG
folders = ["/media/fabian/My Book/datasets/BraTS/2015/train/HGG_nifti/", "/media/fabian/My Book/datasets/BraTS/2015/train/LGG_nifti/", "/media/fabian/My Book/datasets/BraTS/2015/test_nifti/"]
for folder in folders[1:]:
    t1_files = []
    t1_files_output = []
    other_files_t1 = []
    other_files_other = []
    output_files = []
    for id in range(300):
        if not path.isfile(path.join(folder, "%03.0d_T1_bc.nii.gz"%id)):
            continue
        if not path.isfile(path.join(folder, "%03.0d_T1c_bc.nii.gz"%id)):
            continue
        if not path.isfile(path.join(folder, "%03.0d_T2_bc.nii.gz"%id)):
            continue
        if not path.isfile(path.join(folder, "%03.0d_Flair.nii.gz"%id)):
            continue
        other_files_other.append(path.join(folder, "%03.0d_T1c_bc.nii.gz"%id))
        other_files_other.append(path.join(folder, "%03.0d_T2_bc.nii.gz"%id))
        other_files_other.append(path.join(folder, "%03.0d_Flair.nii.gz"%id))
        other_files_other.append(path.join(folder, "%03.0d_T2.nii.gz"%id))
        output_files.append(path.join(folder, "%03.0d_T1c_bc_ws"%id))
        output_files.append(path.join(folder, "%03.0d_T2_bc_ws"%id))
        output_files.append(path.join(folder, "%03.0d_Flair_ws"%id))
        output_files.append(path.join(folder, "%03.0d_T2_ws"%id))

        t1_files.append(path.join(folder, "%03.0d_T1_bc.nii.gz"%id))
        t1_files_output.append(path.join(folder, "%03.0d_T1_bc_ws"%id))

        other_files_t1.append(path.join(folder, "%03.0d_T1_bc.nii.gz"%id))
        other_files_t1.append(path.join(folder, "%03.0d_T1_bc.nii.gz"%id))
        other_files_t1.append(path.join(folder, "%03.0d_T1_bc.nii.gz"%id))
        other_files_t1.append(path.join(folder, "%03.0d_T1_bc.nii.gz"%id))

    print "run pool"
    pool = Pool(4)
    pool.map(run_ws_star, zip(t1_files, t1_files, t1_files_output))
    pool.close()
    pool.join()

    print "run pool"
    pool = Pool(4)
    pool.map(run_ws_star, zip(other_files_t1, other_files_other, output_files))
    pool.close()
    pool.join()