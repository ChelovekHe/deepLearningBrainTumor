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
sys.path.append("experiments/code")
from utils import create_random_rotation, create_matrix_rotation_y, create_matrix_rotation_z, create_matrix_rotation_x


PATCH_SIZE = 128
PERCENT_VAL = 0.15 # expected percentage of validation samples

def create_default_slice():
    slice = np.zeros((PATCH_SIZE**2, 3))
    ctr = 0
    for x in np.arange(-PATCH_SIZE//2, PATCH_SIZE//2):
        for y in np.arange(-PATCH_SIZE//2, PATCH_SIZE//2):
            slice[ctr] = [0, x, y]
            ctr += 1
    return slice


expected_n_samples = 300000

patient_markers = np.loadtxt("patient_markers.txt").astype(np.int32)

train_neg_shape = (expected_n_samples, 18, PATCH_SIZE, PATCH_SIZE)
train_pos_shape = (expected_n_samples*0.5, 18, PATCH_SIZE, PATCH_SIZE)
val_neg_shape = (expected_n_samples * PERCENT_VAL, 18, PATCH_SIZE, PATCH_SIZE)
val_pos_shape = (expected_n_samples * 0.5 * PERCENT_VAL, 18, PATCH_SIZE, PATCH_SIZE)

memmap_name = "patchClassification_ws_resampled_t1km_flair_adc_cbv_rot_PC"

train_neg_memmap = memmap("/home/fabian/datasets/Hirntumor_von_David/experiments/data/%s_train_neg.memmap" % (memmap_name), dtype=np.float32, mode="w+", shape=train_neg_shape)
train_pos_memmap = memmap("/home/fabian/datasets/Hirntumor_von_David/experiments/data/%s_train_pos.memmap" % (memmap_name), dtype=np.float32, mode="w+", shape=train_pos_shape)
val_neg_memmap = memmap("/home/fabian/datasets/Hirntumor_von_David/experiments/data/%s_val_neg.memmap" % (memmap_name), dtype=np.float32, mode="w+", shape=val_neg_shape)
val_pos_memmap = memmap("/home/fabian/datasets/Hirntumor_von_David/experiments/data/%s_val_pos.memmap" % (memmap_name), dtype=np.float32, mode="w+", shape=val_pos_shape)


n_negative_train = 0
n_negative_val = 0
n_positive_train = 0
n_positive_val = 0


path = "/home/fabian/datasets/Hirntumor_von_David/"
subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
subdirs.sort()
voxels_in_patch = PATCH_SIZE**2

default_slice_0 = create_default_slice()
default_slice_1 = np.dot(default_slice_0, create_matrix_rotation_y(-np.pi/2.))
default_slice_2 = np.dot(default_slice_0, create_matrix_rotation_z(-np.pi/2.))

def correct_nans(image):
    t1_image_corr = np.array(image)
    isnan_coords = np.where(np.isnan(image))
    if len(isnan_coords[0]) > 0:
        for coord in zip(isnan_coords[0], isnan_coords[1], isnan_coords[2]):
            coord = list(coord)
            region = image[coord[0]-5 : coord[0]+5, coord[1]-5 : coord[1]+5, coord[2]-5 : coord[2]+5]
            t1_image_corr[tuple(coord)] = np.max(region[~np.isnan(region)])
    return t1_image_corr

val_dirs = ['001', '002', '004', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016']
labels = []
valid_patient_ids = []
for curr_dir in subdirs:
    test = False
    if curr_dir in val_dirs:
        test = True
    patient_id = os.path.split(curr_dir)[-1]
    print patient_id
    from utils import load_patient_resampled
    t1km_image, flair_img, adc_img, cbv_img, t1km_downsampled, seg_combined = load_patient_resampled(int(patient_id))
    if t1km_image is None:
        continue

    idx_in_patient_markers = np.where(patient_markers[:, 0] == int(patient_id))[0][0]

    # find tumor center and size
    tumor_idx = np.where(seg_combined > 1)
    if len(tumor_idx[0]) == 0:
        continue
    tumor_center = (tumor_idx[0].mean(), tumor_idx[1].mean(), tumor_idx[2].mean())
    tumor_size = (float(tumor_idx[0].max() - tumor_idx[0].min()), float(tumor_idx[1].max() - tumor_idx[1].min()), float(tumor_idx[2].max() - tumor_idx[2].min()))

    # randomly sample 30 locations around the tumor center (gaussian distributed)
    print "generating training examples..."

    n_random_locations = 300
    n_random_rotations = 8

    from scipy import ndimage
    # we do not check for too much background because we erode the brain mask prior to generating training samples
    brain_mask = np.array(seg_combined>1).astype(int)
    brain_mask_eroded = ndimage.morphology.binary_erosion(brain_mask, iterations = 10)
    new_brain_idx = np.where(brain_mask_eroded == 1)

    tmp = float(128) / np.max(t1km_image.shape)
    new_shape_downsampled = (int(np.round(t1km_image.shape[0] * tmp)),
                             int(np.round(t1km_image.shape[1] * tmp)),
                             int(np.round(t1km_image.shape[2] * tmp))
                             )

    for i in xrange(n_random_locations):
        print i/float(n_random_locations)*100, " %"

        # make sure we get a bit more tumor samples (compared to uniform sampling over brain region)
        if np.random.random() < 0.25:
            center = (np.random.normal(tumor_center[0], tumor_size[0]/3.), np.random.normal(tumor_center[1], tumor_size[1]/3.), np.random.normal(tumor_center[2], tumor_size[2]/3.))
            center_downsampled = np.array(center).astype(float) / np.array(t1km_image.shape) * np.array(new_shape_downsampled)
        else:
            idx = np.random.choice(len(new_brain_idx[0]), 1)[0]
            center = (new_brain_idx[0][idx], new_brain_idx[1][idx], new_brain_idx[2][idx])
            center_downsampled = np.array(center).astype(float) / np.array(t1km_image.shape) * np.array(new_shape_downsampled)

        # just a helper function to get multiprocessing to work
        def tmp_create_rotated_train_data(ignore_me = None):
            # if i don't do this then each worker from the pool has the same random state -> same rotation = bad!
            np.random.seed()
            # randomly rotate coordinates using rotation matrix
            rotation_matrix = create_random_rotation()
            slice_coords_0 = np.dot(default_slice_0, rotation_matrix)

            # create orthogonal slices
            slice_coords_1 = np.dot(np.dot(default_slice_0, create_matrix_rotation_y(-np.pi/2.)), rotation_matrix)
            slice_coords_2 = np.dot(np.dot(default_slice_0, create_matrix_rotation_z(-np.pi/2.)), rotation_matrix)

            # offset coordinates to center
            slice_coords_0 += center
            slice_coords_1 += center
            slice_coords_2 += center

            slice_t1km_0 = map_coordinates(t1km_image, slice_coords_0.transpose(), None, 3, "nearest").reshape(PATCH_SIZE, PATCH_SIZE)
            slice_t1km_1 = map_coordinates(t1km_image, slice_coords_1.transpose(), None, 3, "nearest").reshape(PATCH_SIZE, PATCH_SIZE)
            slice_t1km_2 = map_coordinates(t1km_image, slice_coords_2.transpose(), None, 3, "nearest").reshape(PATCH_SIZE, PATCH_SIZE)
            slice_flair_0 = map_coordinates(flair_img, slice_coords_0.transpose(), None, 3, "nearest").reshape(PATCH_SIZE, PATCH_SIZE)
            slice_flair_1 = map_coordinates(flair_img, slice_coords_1.transpose(), None, 3, "nearest").reshape(PATCH_SIZE, PATCH_SIZE)
            slice_flair_2 = map_coordinates(flair_img, slice_coords_2.transpose(), None, 3, "nearest").reshape(PATCH_SIZE, PATCH_SIZE)
            slice_adc_0 = map_coordinates(adc_img, slice_coords_0.transpose(), None, 3, "nearest").reshape(PATCH_SIZE, PATCH_SIZE)
            slice_adc_1 = map_coordinates(adc_img, slice_coords_1.transpose(), None, 3, "nearest").reshape(PATCH_SIZE, PATCH_SIZE)
            slice_adc_2 = map_coordinates(adc_img, slice_coords_2.transpose(), None, 3, "nearest").reshape(PATCH_SIZE, PATCH_SIZE)
            slice_cbv_0 = map_coordinates(cbv_img, slice_coords_0.transpose(), None, 3, "nearest").reshape(PATCH_SIZE, PATCH_SIZE)
            slice_cbv_1 = map_coordinates(cbv_img, slice_coords_1.transpose(), None, 3, "nearest").reshape(PATCH_SIZE, PATCH_SIZE)
            slice_cbv_2 = map_coordinates(cbv_img, slice_coords_2.transpose(), None, 3, "nearest").reshape(PATCH_SIZE, PATCH_SIZE)
            slice_t1km_donwsampled_0 = map_coordinates(t1km_downsampled, (default_slice_0 + center_downsampled).transpose(), None, 3, "nearest").reshape(PATCH_SIZE, PATCH_SIZE)
            slice_t1km_donwsampled_1 = map_coordinates(t1km_downsampled, (default_slice_1 + center_downsampled).transpose(), None, 3, "nearest").reshape(PATCH_SIZE, PATCH_SIZE)
            slice_t1km_donwsampled_2 = map_coordinates(t1km_downsampled, (default_slice_2 + center_downsampled).transpose(), None, 3, "nearest").reshape(PATCH_SIZE, PATCH_SIZE)
            slice_seg_0 = np.round(map_coordinates(seg_combined.astype(float), slice_coords_0.transpose(), None, 3, "nearest").reshape(PATCH_SIZE, PATCH_SIZE)).astype(np.int32)
            slice_seg_1 = np.round(map_coordinates(seg_combined.astype(float), slice_coords_1.transpose(), None, 3, "nearest").reshape(PATCH_SIZE, PATCH_SIZE)).astype(np.int32)
            slice_seg_2 = np.round(map_coordinates(seg_combined.astype(float), slice_coords_2.transpose(), None, 3, "nearest").reshape(PATCH_SIZE, PATCH_SIZE)).astype(np.int32)
            slice_seg_0[slice_seg_0 < 0] = 0
            slice_seg_0[slice_seg_0 > 3] = 3
            slice_seg_1[slice_seg_1 < 0] = 0
            slice_seg_1[slice_seg_1 > 3] = 3
            slice_seg_2[slice_seg_2 < 0] = 0
            slice_seg_2[slice_seg_2 > 3] = 3

            return (slice_t1km_0,
                    slice_t1km_1,
                    slice_t1km_2,
                    slice_flair_0,
                    slice_flair_1,
                    slice_flair_2,
                    slice_adc_0,
                    slice_adc_1,
                    slice_adc_2,
                    slice_cbv_0,
                    slice_cbv_1,
                    slice_cbv_2,
                    slice_t1km_donwsampled_0,
                    slice_t1km_donwsampled_1,
                    slice_t1km_donwsampled_2,
                    slice_seg_0,
                    slice_seg_1,
                    slice_seg_2)

        pool = Pool(n_random_rotations)
        result = pool.map(tmp_create_rotated_train_data, [(), (), (), (), (), (), (), ()])
        pool.close()
        pool.join()

        # result = (tmp_create_rotated_train_data(), tmp_create_rotated_train_data())
        for res in result:
            perc_tumor = float(np.sum(res[-3]>1) + np.sum(res[-2]>1) + np.sum(res[-1]>1)) / (PATCH_SIZE**2 * 3.)
            if not test:
                if perc_tumor == 0:
                    train_neg_memmap[n_negative_train:n_negative_train+1] = res
                    n_negative_train += 1
                elif perc_tumor > 0.08:
                    train_pos_memmap[n_positive_train:n_positive_train+1] = res
                    n_positive_train += 1
            else:
                if perc_tumor == 0:
                    val_neg_memmap[n_negative_val:n_negative_val+1] = res
                    n_negative_val += 1
                elif perc_tumor > 0.08:
                    val_pos_memmap[n_positive_val:n_positive_val+1] = res
                    n_positive_val += 1


    print "train (total, pos, neg): ", n_negative_train + n_positive_train, n_positive_train, n_negative_train
    print "test (total, pos, neg): ", n_positive_val + n_negative_val, n_positive_val, n_negative_val



my_dict = {
    "train_total" : n_negative_train + n_positive_train,
    "train_pos": n_positive_train,
    "train_neg": n_negative_train,
    "val_total" : n_positive_val + n_negative_val,
    "val_pos": n_positive_val,
    "val_neg": n_negative_val,
    "train_neg_shape": train_neg_shape,
    "train_pos_shape": train_pos_shape,
    "val_neg_shape": val_neg_shape,
    "val_pos_shape": val_pos_shape

}
with open("%s_properties.pkl" % (memmap_name), 'w') as f:
    cPickle.dump(my_dict, f)
