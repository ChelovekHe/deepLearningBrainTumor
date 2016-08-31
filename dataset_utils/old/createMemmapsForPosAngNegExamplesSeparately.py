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
import threading
from  multiprocessing.dummy import Pool as ThreadPool

PATCH_SIZE = 128
PERCENT_VAL = 0.15

entries_per_datum = PATCH_SIZE**2 * 2
expected_pos_to_neg_ratio = 10000./126964.
expected_n_samples = 650000

train_neg_shape = (expected_n_samples, entries_per_datum)
train_pos_shape = (int(expected_n_samples * expected_pos_to_neg_ratio), entries_per_datum)
val_neg_shape = (int(expected_n_samples * PERCENT_VAL), entries_per_datum)
val_pos_shape = (int(expected_n_samples * PERCENT_VAL * expected_pos_to_neg_ratio), entries_per_datum)

train_neg_memmap = memmap("patchClassification_train_neg_2.memmap", dtype=np.float32, mode="w+", shape=train_neg_shape)
train_pos_memmap = memmap("patchClassification_train_pos_2.memmap", dtype=np.float32, mode="w+", shape=train_pos_shape)
val_neg_memmap = memmap("patchClassification_val_neg_2.memmap", dtype=np.float32, mode="w+", shape=val_neg_shape)
val_pos_memmap = memmap("patchClassification_val_pos_2.memmap", dtype=np.float32, mode="w+", shape=val_pos_shape)

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

val_dirs = ['001', '002', '004', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016']

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
    itk_img = sitk.ReadImage(os.path.join(curr_dir, "T1_m2_bc.nii.gz"))
    seg_ce = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "seg_ce.nii.gz"))).astype(np.int32)
    seg_edema = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "seg_edema.nii.gz"))).astype(np.int32)
    seg_necrosis_ce = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "seg_necrosis_ce.nii.gz"))).astype(np.int32)
    t1_image = sitk.GetArrayFromImage(itk_img)

    assert seg_ce.shape == t1_image.shape
    assert seg_edema.shape == t1_image.shape
    assert seg_necrosis_ce.shape == t1_image.shape

    xy_spacing = None
    z_spacing = None
    spacing = itk_img.GetSpacing()
    spacing_target_xy = 0.5
    spacing_target_z = 1.

    # weird sitk spacing stuff... -.-'
    if spacing[0] == spacing[1]:
        xy_spacing = spacing[0]
        z_spacing = spacing[2]
    else:
        xy_spacing = spacing[1]
        z_spacing = spacing[0]

    new_shape = (int(z_spacing/spacing_target_z*float(t1_image.shape[0])),
                 int(xy_spacing/spacing_target_xy*float(t1_image.shape[1])),
                 int(xy_spacing/spacing_target_xy*float(t1_image.shape[2])))

    # maxVal hack to get skimage to work
    maxVal = np.max(t1_image)
    # t1_image = resize(t1_image/maxVal, new_shape)
    def resize_star(args):
        return resize(*args)

    pool = ThreadPool(4)
    (t1_image_resized, seg_ce_resized, seg_edema_resized, seg_necrosis_ce_resized) = pool.map(resize_star, [(t1_image/maxVal, new_shape),
                                                                                                       (seg_ce, new_shape),
                                                                                                       (seg_edema, new_shape),
                                                                                                       (seg_necrosis_ce, new_shape)])
    pool.close()
    pool.join()

    t1_image_resized *= maxVal

    print t1_image_resized.shape
    outside_value = t1_image_resized[0,0,0]
    t1_image_resized -= outside_value
    # max_val = t1_image.max()
    # if np.isnan(max_val):
    #     IPython.embed()
    # t1_image /= max_val

    # std works better than max on MRT because of outliers. We should invest some time into better normalization
    # techniques
    img_std = t1_image_resized[t1_image_resized != 0].std()
    t1_image_resized /= img_std

    # create joint segmentation map (useful for later)
    seg_combined = np.zeros(new_shape)

    # necrosis is 3, ce tumor is 2, edema is 1
    seg_combined[seg_necrosis_ce == 1] = 3
    seg_combined[seg_ce == 1] = 2
    seg_combined[seg_edema == 1] = 1

    # images are brain extracted, they have some funky value everywhere outside of the brain region. we only want
    # slices with enough brain voxels. We can ensure that by counting background pixels
    outside_value = t1_image_resized[0, 0, 0]

    # iterate over patches
    for z in xrange(new_shape[0]):
        y0 = 0
        # print "z", z
        while (y0 + PATCH_SIZE) < new_shape[2]:
            x0 = 0
            # print "y0", y0
            while (x0 + PATCH_SIZE) < new_shape[1]:
                # print "x0", x0
                patch = t1_image_resized[z, x0:x0+PATCH_SIZE, y0:y0+PATCH_SIZE]
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
    print "current pos train: ", n_positive_train, " max allowed: ", train_pos_shape[0]
    print "current pos val: ", n_positive_val, " max allowed: ", val_pos_shape[0]
    print "current neg train: ", n_negative_train, " max allowed: ", train_neg_shape[0]
    print "current neg val: ", n_negative_val, " max allowed: ", val_neg_shape[0]


print "train: ", n_negative_train + n_positive_train, n_positive_train, n_negative_train
print "test: ", n_positive_val + n_negative_val, n_positive_val, n_negative_val
print "image mean: ", img_mean / float(n_negative_train + n_positive_train + n_positive_val + n_negative_val)


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
with open("patchClassification_memmap_properties_2.pkl", 'w') as f:
    cPickle.dump(my_dict, f)