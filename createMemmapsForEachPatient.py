__author__ = 'fabian'

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
from multiprocessing.dummy import Pool as ThreadPool

PATCH_SIZE = 128
PERCENT_VAL = 0.15

entries_per_datum = PATCH_SIZE**2 * 2

path = "/home/fabian/datasets/Hirntumor_von_David/"
subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
# subdirs = [os.path.join(path, '017')]
subdirs.sort()
voxels_in_patch = PATCH_SIZE**2

valid_patient_dirs = []
val_dirs = ['001', '002', '004', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016']
labels = []
for curr_dir in subdirs:
    test = False
    if curr_dir in val_dirs:
        test = True
    patient_id = os.path.split(curr_dir)[-1]
    print patient_id
    # check if image exists
    if not os.path.isfile(os.path.join(curr_dir, "T1_m2_bc_ws.nii.gz")):
        continue
    if not os.path.isfile(os.path.join(curr_dir, "seg_ce.nii.gz")):
        continue
    if not os.path.isfile(os.path.join(curr_dir, "seg_edema.nii.gz")):
        continue
    if not os.path.isfile(os.path.join(curr_dir, "seg_necrosis_ce.nii.gz")):
        continue
    if not os.path.isfile(os.path.join(curr_dir, "T1KM_m2_bc_ws.nii.gz")):
        continue
    if not os.path.isfile(os.path.join(curr_dir, "FLAIR_m2_bc_ws.nii.gz")):
        continue
    if not os.path.isfile(os.path.join(curr_dir, "ADC_mutualinfo2_reg.nii.gz")):
        continue
    if not os.path.isfile(os.path.join(curr_dir, "CBV_mutualinfo2_reg.nii.gz")):
        continue

    valid_patient_dirs.append(curr_dir)

    # load image
    itk_img = sitk.ReadImage(os.path.join(curr_dir, "T1_m2_bc_ws.nii.gz"))
    t1km_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "T1KM_m2_bc_ws.nii.gz"))).astype(np.float)
    FLAIR_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "FLAIR_m2_bc_ws.nii.gz"))).astype(np.float)
    ADC_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "ADC_mutualinfo2_reg.nii.gz"))).astype(np.float)
    CBV_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "CBV_mutualinfo2_reg.nii.gz"))).astype(np.float)
    seg_ce = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "seg_ce.nii.gz"))).astype(np.float)
    seg_edema = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "seg_edema.nii.gz"))).astype(np.float)
    seg_necrosis_ce = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "seg_necrosis_ce.nii.gz"))).astype(np.float)
    brain_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "T1_mutualinfo2_bet_mask.nii.gz"))).astype(np.float)
    t1_image = sitk.GetArrayFromImage(itk_img).astype(np.float)

    assert seg_ce.shape == t1_image.shape
    assert seg_edema.shape == t1_image.shape
    assert seg_necrosis_ce.shape == t1_image.shape
    assert brain_mask.shape == t1_image.shape
    assert t1km_image.shape == t1_image.shape
    assert CBV_image.shape == t1_image.shape
    assert ADC_image.shape == t1_image.shape
    assert FLAIR_image.shape == t1_image.shape

    # correct nans
    t1_image_corr = np.array(t1_image)
    isnan_coords = np.where(np.isnan(t1_image))
    if len(isnan_coords[0]) > 0:
        for coord in zip(isnan_coords[0], isnan_coords[1], isnan_coords[2]):
            coord = list(coord)
            region = t1_image[coord[0]-5 : coord[0]+5, coord[1]-5 : coord[1]+5, coord[2]-5 : coord[2]+5]
            t1_image_corr[tuple(coord)] = np.max(region[~np.isnan(region)])
    t1_image = t1_image_corr

    spacing = np.array(itk_img.GetSpacing())[[2, 1, 0]]
    spacing_target = [1, 0.5, 0.5]

    new_shape = (int(spacing[0]/spacing_target[0]*float(t1_image.shape[0])),
                 int(spacing[1]/spacing_target[1]*float(t1_image.shape[1])),
                 int(spacing[2]/spacing_target[2]*float(t1_image.shape[2])))

    # t1_image = resize(t1_image/maxVal, new_shape)
    def resize_star(args):
        return resize(*args)

    outside_value_t1 = t1_image[0,0,0]
    outside_value_t1km = t1km_image[0,0,0]
    outside_value_flair = FLAIR_image[0,0,0]
    pool = ThreadPool(9)
    (t1_image, t1km_image, FLAIR_image, ADC_image, CBV_image, seg_ce, seg_edema, seg_necrosis_ce, brain_mask) \
        = pool.map(resize_star, [(t1_image, new_shape, 3, 'edge'),
                                 (t1km_image, new_shape, 3, 'edge'),
                                 (FLAIR_image, new_shape, 3, 'edge'),
                                 (ADC_image, new_shape, 3, 'edge'),
                                 (CBV_image, new_shape, 3, 'edge'),
                                 (seg_ce, new_shape, 1, 'edge'),
                                 (seg_edema, new_shape, 1, 'edge'),
                                 (seg_necrosis_ce, new_shape, 1, 'edge'),
                                 (brain_mask, new_shape, 1, 'edge')])
    pool.close()
    pool.join()

    t1_image = t1_image.astype(np.float32)
    t1km_image = t1km_image.astype(np.float32)
    FLAIR_image = FLAIR_image.astype(np.float32)
    ADC_image = ADC_image.astype(np.float32)
    CBV_image = CBV_image.astype(np.float32)
    seg_ce = seg_ce.astype(np.int32)
    seg_edema = seg_edema.astype(np.int32)
    seg_necrosis_ce = seg_necrosis_ce.astype(np.int32)
    brain_mask = np.round(brain_mask).astype(np.int32)

    t1_image[brain_mask == 0] = outside_value_t1
    t1km_image[brain_mask == 0] = outside_value_t1km
    FLAIR_image[brain_mask == 0] = outside_value_flair
    # outside_value = 0
    # t1_image -= outside_value

    # create joint segmentation map (useful for later)
    seg_combined = np.zeros(t1_image.shape, dtype=np.int32)

    # necrosis is 3, ce tumor is 2, edema is 1
    seg_combined[seg_necrosis_ce == 1] = 3
    seg_combined[seg_ce == 1] = 2
    seg_combined[seg_edema == 1] = 1

    # images are brain extracted, they have some funky value everywhere outside of the brain region. we only want
    # slices with enough brain voxels. We can ensure that by counting background pixels

    # extract interesting region of image to reduce disk space usage
    # we need the brain region with a little bit of buffer around it
    brainmask_brain_idx = np.where(brain_mask == 1)
    minZidx = int(np.round(np.max((0, brainmask_brain_idx[0].min()-PATCH_SIZE/5.))))
    maxZidx = int(np.round(np.min((brain_mask.shape[0], brainmask_brain_idx[0].max()+PATCH_SIZE/5.))))
    minXidx = int(np.round(np.max((0, brainmask_brain_idx[1].min()-PATCH_SIZE/5.))))
    maxXidx = int(np.round(np.min((brain_mask.shape[1], brainmask_brain_idx[1].max()+PATCH_SIZE/5.))))
    minYidx = int(np.round(np.max((0, brainmask_brain_idx[2].min()-PATCH_SIZE/5.))))
    maxYidx = int(np.round(np.min((brain_mask.shape[2], brainmask_brain_idx[2].max()+PATCH_SIZE/5.))))

    # create memmap of exactly the size we need (all modalities/segmentation into one single memmap per patient)
    required_shape = (maxZidx-minZidx, maxXidx-minXidx, maxYidx-minYidx)
    tmp = list(required_shape)
    tmp.insert(0, 6)
    memmap_shape = tuple(tmp)
    memmap_name = "patient_%s"%patient_id
    this_memmap = memmap("%s_data.memmap"%memmap_name, dtype=np.float32, mode="w+", shape=memmap_shape)

    # resize images
    resizer = (slice(minZidx, maxZidx), slice(minXidx, maxXidx), slice(minYidx, maxYidx))

    t1_image = t1_image[resizer]
    t1km_image = t1km_image[resizer]
    FLAIR_image = FLAIR_image[resizer]
    ADC_image = ADC_image[resizer]
    CBV_image = CBV_image[resizer]
    seg_combined = seg_combined[resizer]

    # write everything into memmap
    this_memmap[0] = t1_image
    this_memmap[1] = t1km_image
    this_memmap[2] = FLAIR_image
    this_memmap[3] = ADC_image
    this_memmap[4] = CBV_image
    this_memmap[5] = seg_combined.astype(np.float32)

    # store memmap shape etc
    my_dict = {
        "shape" : memmap_shape,
        "t1_idx": 0,
        "t1km_idx": 1,
        "flair_idx" : 2,
        "adc_idx": 3,
        "cbv_idx": 4,
        "seg_idx": 5,
    }

    with open("%s_properties.pkl" % memmap_name, 'w') as f:
        cPickle.dump(my_dict, f)