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


lmdb_name = "lmdb_train128"
lmdb_name_test = "lmdb_test128"

map_size = 1e12

env = lmdb.open(lmdb_name, map_size=map_size)
env_test = lmdb.open(lmdb_name_test, map_size=map_size)

path = "/home/fabian/datasets/Hirntumor_von_David/"
subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
# subdirs = [os.path.join(path, '017')]
subdirs.sort()
PATCH_SIZE = 128
voxels_in_patch = PATCH_SIZE**2

n_positive = 0
n_total = 0

n_positive_test = 0
n_total_test = 0
img_mean = 0


with env.begin(write=True) as txn:
    with env_test.begin(write=True) as txn_test:
        for curr_dir in subdirs:
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

                        # find label of patch. We label a patch as tumor if it has more than 10% tumor pixels. if it has
                        # between 0 and 10% it will be discarded (belongs to neither class)
                        label = 0
                        percent_of_tumor_voxels = float(np.sum(seg_patch > 1)) / float(voxels_in_patch)
                        if percent_of_tumor_voxels > 0:
                            if percent_of_tumor_voxels < 0.10:
                                x0 += PATCH_SIZE
                                continue
                            label = 1
                        str_id = "%s_z%i_y%i_x%i_patchSize%i"%(patient_id, z, y0, x0, PATCH_SIZE)

                        img_mean += np.mean(patch)

                        test = np.random.rand() < 0.15
                        if test:
                            txn_test.put(str_id.encode('ascii'), cPickle.dumps((patch, seg_patch, label)))
                            n_total_test += 1
                            n_positive_test += label
                        else:
                            txn.put(str_id.encode('ascii'), cPickle.dumps((patch, seg_patch, label)))
                            n_total += 1
                            n_positive += label
                        # plt.imsave("%s.jpg" % str_id, patch, cmap="gray")
                        x0 += int(PATCH_SIZE / 3)
                    y0 += int(PATCH_SIZE / 3)



print "train: ", n_total, n_positive, n_total-n_positive
print "test: ", n_total_test, n_positive_test, n_total_test-n_positive_test
print "image mean: ", img_mean / float(n_total + n_total_test)