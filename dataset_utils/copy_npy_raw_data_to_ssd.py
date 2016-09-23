__author__ = 'fabian'
import os
import numpy as np
import shutil

src_folder = "/media/fabian/My Book/datasets/Hirntumor_rawData/"
dest_folder = "/media/fabian/DeepLearningData/datasets/Hirntumor_raw_data"

for id in range(150):
    patient_folder = os.path.join(src_folder, "%03.0d" % id)
    if not os.path.isdir(patient_folder):
        continue
    if not os.path.isfile(os.path.join(patient_folder, "ADC_mutualinfo2_reg.npy")):
        continue
    if not os.path.isfile(os.path.join(patient_folder, "CBV_mutualinfo2_reg.npy")):
        continue
    if not os.path.isfile(os.path.join(patient_folder, "FLAIR_m2_bc_ws.npy")):
        continue
    if not os.path.isfile(os.path.join(patient_folder, "seg_all.npy")):
        continue
    if not os.path.isfile(os.path.join(patient_folder, "T1KM_m2_bc_ws.npy")):
        continue

    # shutil.copyfile(os.path.join(patient_folder, "ADC_mutualinfo2_reg.npy"), os.path.join(dest_folder, "patient_%03.0d_ADC_data.npy" % id))
    # shutil.copyfile(os.path.join(patient_folder, "CBV_mutualinfo2_reg.npy"), os.path.join(dest_folder, "patient_%03.0d_CBV_data.npy" % id))
    # shutil.copyfile(os.path.join(patient_folder, "FLAIR_m2_bc_ws.npy"), os.path.join(dest_folder, "patient_%03.0d_FLAIR_data.npy" % id))
    # shutil.copyfile(os.path.join(patient_folder, "seg_all.npy"), os.path.join(dest_folder, "patient_%03.0d_seg.npy" % id))
    # shutil.copyfile(os.path.join(patient_folder, "T1KM_m2_bc_ws.npy"), os.path.join(dest_folder, "patient_%03.0d_T1KM_data.npy" % id))
    shutil.copyfile(os.path.join(patient_folder, "T1KM_m2_bc_ws_downsampled128.npy"), os.path.join(dest_folder, "patient_%03.0d_T1KM_downsampled_128_data.npy" % id))
