__author__ = 'fabian'
import numpy as np
import skimage
import lmdb
import matplotlib.pyplot as plt
import os.path
import os
import SimpleITK as sitk
import cPickle
import IPython
from numpy import memmap
from skimage.transform import resize
from scipy.ndimage import map_coordinates
import threading
from multiprocessing import Pool
import sys
from dataset_utils import load_patient_resampled, extract_brain_region, pad_3d_image
from copy import deepcopy
from scipy.ndimage import map_coordinates
sys.path.append("experiments/code")

def map_coordinates_star(args):
    return map_coordinates(*args)

PATCH_SIZE = 256
slice_size = int(np.ceil(np.sqrt(2*PATCH_SIZE**2)))
# ensure slice_size is compatible with several (here 5) maxpool operations
slice_size += 32-slice_size%32

expected_n_samples = 70000

patient_markers = np.loadtxt("../../patient_markers.txt").astype(np.int32)

memmap_shape = (expected_n_samples, 25, slice_size, slice_size)
info_memmap_shape = (expected_n_samples, 4)

memmap_name = "patchSegmentation_allInOne_ws_t1km_flair_adc_cbv"

memmap_data = memmap("%s.memmap" % (memmap_name), dtype=np.float32, mode="w+", shape=memmap_shape)
memmap_gt = memmap("%s_info.memmap" % (memmap_name), dtype=np.float32, mode="w+", shape=info_memmap_shape)

def add_patch_to_memmap(x, y, z, t1km_img, flair_img, adc_img, cbv_img, seg_combined, slice_size, patient_id, patient_state, data_ctr):
    t1km_patch = t1km_img[z-2:z+3, x:x+slice_size, y:y+slice_size]
    flair_patch = flair_img[z-2:z+3, x:x+slice_size, y:y+slice_size]
    adc_patch = adc_img[z-2:z+3, x:x+slice_size, y:y+slice_size]
    cbv_patch = cbv_img[z-2:z+3, x:x+slice_size, y:y+slice_size]
    seg_patch = seg_combined[z-2:z+3, x:x+slice_size, y:y+slice_size]
    # no empty slices
    if len(np.unique(seg_patch[2])) == 1 and np.unique(seg_patch[2])[0] == 0:
        return data_ctr
    memmap_data[data_ctr, 0:5, :, :] = t1km_patch
    memmap_data[data_ctr, 5:10, :, :] = flair_patch
    memmap_data[data_ctr, 10:15, :, :] = adc_patch
    memmap_data[data_ctr, 15:20, :, :] = cbv_patch
    memmap_data[data_ctr, 20:25, :, :] = seg_patch
    memmap_gt[data_ctr, 1:4] = patient_state
    memmap_gt[data_ctr, 0] = patient_id
    data_ctr += 1
    return data_ctr

data_ctr = 0

delta = PATCH_SIZE/2.

class_frequencies = {}
for c in range(5):
    class_frequencies[c] = 0

for patient_id in xrange(150):
    print patient_id
    idx_in_patient_markers = np.where(patient_markers[:, 0] == int(patient_id))
    if len(idx_in_patient_markers[0]) == 0:
        print "patient not found in marker file"
        patient_state = np.array([-1, -1, -1])
    else:
        idx_in_patient_markers = idx_in_patient_markers[0][0]
        rtk2_state = mgmt_state = egfr_state = -1
        if patient_markers[idx_in_patient_markers][1] != -1:
            rtk2_state = patient_markers[idx_in_patient_markers][1]
        if patient_markers[idx_in_patient_markers][2] != -1:
            mgmt_state = patient_markers[idx_in_patient_markers][2]
        if patient_markers[idx_in_patient_markers][3] != -1:
            egfr_state = patient_markers[idx_in_patient_markers][3]
        patient_state = np.array([rtk2_state, mgmt_state, egfr_state])

    t1km_img, flair_img, adc_img, cbv_img, t1km_downsampled, seg_combined = load_patient_resampled(patient_id)
    if t1km_img is None:
        continue

    t1km_img = extract_brain_region(t1km_img, seg_combined)
    flair_img = extract_brain_region(flair_img, seg_combined)
    adc_img = extract_brain_region(adc_img, seg_combined)
    cbv_img = extract_brain_region(cbv_img, seg_combined)
    seg_combined = extract_brain_region(seg_combined, seg_combined)

    target_min_size = np.array([2, slice_size, slice_size])
    pad_x = max(slice_size - PATCH_SIZE, slice_size - t1km_img.shape[1])
    pad_y = max(slice_size - PATCH_SIZE, slice_size - t1km_img.shape[2])

    # make pad even
    pad_x += pad_x%2
    pad_y += pad_y%2

    t1km_img = pad_3d_image(t1km_img, np.array([2, pad_x, pad_y]))
    flair_img = pad_3d_image(flair_img, np.array([2, pad_x, pad_y]))
    adc_img = pad_3d_image(adc_img, np.array([2, pad_x, pad_y]))
    cbv_img = pad_3d_image(cbv_img, np.array([2, pad_x, pad_y]))
    seg_combined = pad_3d_image(seg_combined, np.array([2, pad_x, pad_y]))

    for c in range(5):
        class_frequencies[c] += np.sum(seg_combined == c)

    print "generating training examples..."
    # extract training samples all over the image
    for z in range(2, t1km_img.shape[0]-2):
        # print z
        y = 0
        while y < t1km_img.shape[2] - slice_size:
            x = 0
            while x < t1km_img.shape[1] - slice_size:
                data_ctr = add_patch_to_memmap(x, y, z, t1km_img, flair_img, adc_img, cbv_img, seg_combined, slice_size, patient_id, patient_state, data_ctr)
                x += delta
            x = t1km_img.shape[1] - slice_size
            data_ctr = add_patch_to_memmap(x, y, z, t1km_img, flair_img, adc_img, cbv_img, seg_combined, slice_size, patient_id, patient_state, data_ctr)
            y += delta
        y = t1km_img.shape[2] - slice_size
        x = 0
        while x < t1km_img.shape[1] - slice_size:
            data_ctr = add_patch_to_memmap(x, y, z, t1km_img, flair_img, adc_img, cbv_img, seg_combined, slice_size, patient_id, patient_state, data_ctr)
            x += delta
        x = t1km_img.shape[1] - slice_size
        data_ctr = add_patch_to_memmap(x, y, z, t1km_img, flair_img, adc_img, cbv_img, seg_combined, slice_size, patient_id, patient_state, data_ctr)

    # extract some more training examples around the tumor
    # find tumor center and size
    '''print "adding additional tumor data:"
    tumor_idx = np.where(seg_combined > 1)
    if len(tumor_idx[0]) > 0:
        tumor_center = (tumor_idx[0].mean(), tumor_idx[1].mean(), tumor_idx[2].mean())
        tumor_size = (float(tumor_idx[0].max() - tumor_idx[0].min()), float(tumor_idx[1].max() - tumor_idx[1].min()), float(tumor_idx[2].max() - tumor_idx[2].min()))
        coords = np.meshgrid(range(5), range(slice_size), range(slice_size))
        coords[0] = coords[0].transpose((1, 0, 2)) - 2
        coords[1] = coords[1].transpose((1, 0, 2)) - slice_size/2
        coords[2] = coords[2].transpose((1, 0, 2)) - slice_size/2
        for _ in xrange(20):
            center = list((np.random.normal(tumor_center[0], tumor_size[0]/5.), np.random.normal(tumor_center[1], tumor_size[1]/5.), np.random.normal(tumor_center[2], tumor_size[2]/5.)))
            coords2 = deepcopy(coords)
            coords2[0] += center[0]
            coords2[1] += center[1]
            coords2[2] += center[2]
            print center
            pool = Pool(5)
            t1km_patch, flair_patch, adc_patch, cbv_patch, seg_patch = pool.map(map_coordinates_star, [(t1km_img, coords2, None, 3, "nearest"),
                                                                                                       (flair_img, coords2, None, 3, "nearest"),
                                                                                                       (adc_img, coords2, None, 3, "nearest"),
                                                                                                       (cbv_img, coords2, None, 3, "nearest"),
                                                                                                       (seg_combined, coords2, None, 3, "nearest")
                                                                                                       ])

            pool.close()
            pool.join()

            memmap_data[data_ctr, 0:5, :, :] = t1km_patch
            memmap_data[data_ctr, 5:10, :, :] = flair_patch
            memmap_data[data_ctr, 10:15, :, :] = adc_patch
            memmap_data[data_ctr, 15:20, :, :] = cbv_patch
            memmap_data[data_ctr, 20:25, :, :] = seg_patch
            memmap_gt[data_ctr, 1:4] = patient_state
            memmap_gt[data_ctr, 0] = patient_id
            data_ctr += 1'''
    print "data_ctr: ", data_ctr
    print class_frequencies


my_dict = {
    "n_data" : data_ctr,
    "train_neg_shape": memmap_shape,
    "info_shape": info_memmap_shape,
    "class_frequencies": class_frequencies

}
with open("%s_properties.pkl" % (memmap_name), 'w') as f:
    cPickle.dump(my_dict, f)
