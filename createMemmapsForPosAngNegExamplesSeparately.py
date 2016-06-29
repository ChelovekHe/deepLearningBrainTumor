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

PATCH_SIZE = 128
PERCENT_VAL = 0.15

# from lmdb experiments we know that we will have roughly 115k train and 21k test images. create memmaps of this
# size and save how many entries are actually used. that is not very elegant but who cares!?
# each entry has size 128**2 * 2 + 1 (patch + segmentation + label). we save everything as float and convert to int in
# batchgen
train_neg_memmap = memmap("patchClassification128_neg_train_2.memmap", dtype=np.float32, mode="w+", shape=(450000, 128*128*2))
train_pos_memmap = memmap("patchClassification128_pos_train_2.memmap", dtype=np.float32, mode="w+", shape=(450000 * 10000./126964., 128*128*2))
val_pos_memmap = memmap("patchClassification128_pos_val_2.memmap", dtype=np.float32, mode="w+", shape=(450000*PERCENT_VAL * 10000./126964, 128*128*2))
val_neg_memmap = memmap("patchClassification128_neg_val_2.memmap", dtype=np.float32, mode="w+", shape=(450000*PERCENT_VAL, 128*128*2))

path = "/home/fabian/datasets/Hirntumor_von_David/"
subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
# subdirs = [os.path.join(path, '017')]
subdirs.sort()
voxels_in_patch = PATCH_SIZE**2


n_positive_train = 0
n_negative_train = 0
n_positive_val = 0
n_negative_val = 0
img_mean = 0

val_dirs = ['001', '002', '004', '005', '006', '007', '008', '009', '011', '012']

for curr_dir in subdirs:
    test = False
    if curr_dir in val_dirs:
        test = True
    patient_id = os.path.split(curr_dir)[-1]
    print patient_id
    # check if image exists
    if not os.path.isfile(os.path.join(curr_dir, "T1_m2_bc.nii.gz")):
        continue
    if not os.path.isfile(os.path.join(curr_dir, "seg_ce.nii.gz")):
        continue
    if not os.path.isfile(os.path.join(curr_dir, "seg_edema.nii.gz")):
        continue
    if not os.path.isfile(os.path.join(curr_dir, "seg_necrosis_ce.nii.gz")):
        continue

    # load image
    t1_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "T1_m2_bc.nii.gz")))
    outside_value = t1_image[0,0,0]
    t1_image -= outside_value
    max_val = t1_image.max()
    if np.isnan(max_val):
        IPython.embed()
    t1_image /= max_val

    # create joint segmentation map (useful for later)
    seg_ce = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "seg_ce.nii.gz")))
    seg_edema = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "seg_edema.nii.gz")))
    seg_necrosis_ce = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "seg_necrosis_ce.nii.gz")))
    seg_combined = np.zeros(seg_ce.shape)

    # necrosis is 3, ce tumor is 2, edema is 1
    seg_combined[seg_necrosis_ce == 1] = 3
    seg_combined[seg_ce == 1] = 2
    seg_combined[seg_edema == 1] = 1

    # images are brain extracted, they have some funky value everywhere outside of the brain region. we only want
    # slices with enough brain voxels. We can ensure that by counting background pixels
    outside_value = t1_image[0, 0, 0]

    # iterate over patches
    for z in xrange(t1_image.shape[0]):
        y0 = 0
        # print z
        while (y0 + PATCH_SIZE) < t1_image.shape[2]:
            x0 = 0
            while (x0 + PATCH_SIZE) < t1_image.shape[1]:
                patch = t1_image[z, x0:x0+PATCH_SIZE, y0:y0+PATCH_SIZE]
                seg_patch = seg_combined[z, x0:x0+PATCH_SIZE, y0:y0+PATCH_SIZE]
                patch = patch[np.newaxis, :, :]
                seg_patch = seg_patch[np.newaxis, :, :]
                # discard patch if more than 50% are background
                if float(np.sum(patch == outside_value))/float(voxels_in_patch) > 0.5:
                    x0 += PATCH_SIZE
                    continue

                # find label of patch. We label a patch as tumor if it has more than 8% tumor pixels. if it has
                # between 0 and 10% it will be discarded (belongs to neither class)
                label = 0
                percent_of_tumor_voxels = float(np.sum(seg_patch > 1)) / float(voxels_in_patch)
                if percent_of_tumor_voxels > 0:
                    if percent_of_tumor_voxels < 0.08:
                        x0 += PATCH_SIZE
                        continue
                    label = 1
                str_id = "%s_z%i_y%i_x%i_patchSize%i"%(patient_id, z, y0, x0, PATCH_SIZE)

                img_mean += np.mean(patch)

                # test = np.random.rand() < PERCENT_VAL
                if test:
                    if label == 0:
                        val_neg_memmap[n_negative_val][:128**2] = patch.flatten()
                        val_neg_memmap[n_negative_val][128**2:128**2*2] = seg_patch.flatten()
                        n_negative_val += 1
                    elif label == 1:
                        val_pos_memmap[n_positive_val][:128**2] = patch.flatten()
                        val_pos_memmap[n_positive_val][128**2:128**2*2] = seg_patch.flatten()
                        n_positive_val += 1
                else:
                    if label == 0:
                        train_neg_memmap[n_negative_train][:128**2] = patch.flatten()
                        train_neg_memmap[n_negative_train][128**2:128**2*2] = seg_patch.flatten()
                        n_negative_train += 1
                    if label == 1:
                        train_pos_memmap[n_positive_train][:128**2] = patch.flatten()
                        train_pos_memmap[n_positive_train][128**2:128**2*2] = seg_patch.flatten()
                        n_positive_train += 1
                # plt.imsave("%s.jpg" % str_id, patch, cmap="gray")
                x0 += int(PATCH_SIZE / 5)
            y0 += int(PATCH_SIZE / 5)



print "train: ", n_negative_train + n_positive_train, n_positive_train, n_negative_train
print "test: ", n_positive_val + n_negative_val, n_positive_val, n_negative_val
print "image mean: ", img_mean / float(n_negative_train + n_positive_train + n_positive_val + n_negative_val)