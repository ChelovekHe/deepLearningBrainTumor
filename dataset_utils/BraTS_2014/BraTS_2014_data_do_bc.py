__author__ = 'fabian'

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from subprocess import call
import os.path as path
from multiprocessing import Pool


def run_bc_star(args):
    return run_bc(*args)

def run_bc(input, output):
    cmd = ["Rscript", "/media/fabian/My Book/datasets/BraTS/biascorr_noMask.R", input, output]
    call(cmd)

# run convert_BraTS_2014_to_npy first

# there are 251 [0, 250] patients in HGG
HGG_folder = "/media/fabian/My Book/datasets/BraTS/2014/train/HGG/"
input_files = []
output_files = []
for id in range(251):
    if not path.isfile(path.join(HGG_folder, "%03.0d_T1.nii.gz"%id)):
        continue
    if not path.isfile(path.join(HGG_folder, "%03.0d_T1c.nii.gz"%id)):
        continue
    if not path.isfile(path.join(HGG_folder, "%03.0d_T2.nii.gz"%id)):
        continue
    if not path.isfile(path.join(HGG_folder, "%03.0d_Flair.nii.gz"%id)):
        continue
    input_files.append(path.join(HGG_folder, "%03.0d_T1.nii.gz"%id))
    input_files.append(path.join(HGG_folder, "%03.0d_T1c.nii.gz"%id))
    input_files.append(path.join(HGG_folder, "%03.0d_T2.nii.gz"%id))
    input_files.append(path.join(HGG_folder, "%03.0d_Flair.nii.gz"%id))
    output_files.append(path.join(HGG_folder, "%03.0d_T1_bc"%id))
    output_files.append(path.join(HGG_folder, "%03.0d_T1c_bc"%id))
    output_files.append(path.join(HGG_folder, "%03.0d_T2_bc"%id))
    output_files.append(path.join(HGG_folder, "%03.0d_Flair_bc"%id))

print "run pool"
pool = Pool(8)
pool.map(run_bc_star, zip(input_files, output_files))
pool.close()
pool.join()

# there are 156 [0, 155] patients in HGG
LGG_folder = "/media/fabian/My Book/datasets/BraTS/2014/train/LGG/"
input_files = []
output_files = []
for id in range(156):
    if not path.isfile(path.join(LGG_folder, "%03.0d_T1.nii.gz"%id)):
        continue
    if not path.isfile(path.join(LGG_folder, "%03.0d_T1c.nii.gz"%id)):
        continue
    if not path.isfile(path.join(LGG_folder, "%03.0d_T2.nii.gz"%id)):
        continue
    if not path.isfile(path.join(LGG_folder, "%03.0d_Flair.nii.gz"%id)):
        continue
    input_files.append(path.join(LGG_folder, "%03.0d_T1.nii.gz"%id))
    input_files.append(path.join(LGG_folder, "%03.0d_T1c.nii.gz"%id))
    input_files.append(path.join(LGG_folder, "%03.0d_T2.nii.gz"%id))
    input_files.append(path.join(LGG_folder, "%03.0d_Flair.nii.gz"%id))
    output_files.append(path.join(LGG_folder, "%03.0d_T1_bc"%id))
    output_files.append(path.join(LGG_folder, "%03.0d_T1c_bc"%id))
    output_files.append(path.join(LGG_folder, "%03.0d_T2_bc"%id))
    output_files.append(path.join(LGG_folder, "%03.0d_Flair_bc"%id))

print "run pool"
pool = Pool(8)
pool.map(run_bc_star, zip(input_files, output_files))
pool.close()
pool.join()

test_folder = "/media/fabian/My Book/datasets/BraTS/2014/test"
input_files = []
output_files = []
for id in range(200):
    if not path.isfile(path.join(test_folder, "%03.0d_T1.nii.gz"%id)):
        continue
    if not path.isfile(path.join(test_folder, "%03.0d_T1c.nii.gz"%id)):
        continue
    if not path.isfile(path.join(test_folder, "%03.0d_T2.nii.gz"%id)):
        continue
    if not path.isfile(path.join(test_folder, "%03.0d_Flair.nii.gz"%id)):
        continue
    input_files.append(path.join(test_folder, "%03.0d_T1.nii.gz"%id))
    input_files.append(path.join(test_folder, "%03.0d_T1c.nii.gz"%id))
    input_files.append(path.join(test_folder, "%03.0d_T2.nii.gz"%id))
    input_files.append(path.join(test_folder, "%03.0d_Flair.nii.gz"%id))
    output_files.append(path.join(test_folder, "%03.0d_T1_bc"%id))
    output_files.append(path.join(test_folder, "%03.0d_T1c_bc"%id))
    output_files.append(path.join(test_folder, "%03.0d_T2_bc"%id))
    output_files.append(path.join(test_folder, "%03.0d_Flair_bc"%id))

print "run pool"
pool = Pool(8)
pool.map(run_bc_star, zip(input_files, output_files))
pool.close()
pool.join()

