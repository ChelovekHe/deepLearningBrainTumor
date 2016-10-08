__author__ = 'fabian'
import numpy as np
import os
import SimpleITK as sitk
from multiprocessing import Pool as ThreadPool
from skimage.transform import resize


def create_matrix_rotation_x(angle, matrix = None):
    rotation_x = np.array([[1,              0,              0],
                           [0,              np.cos(angle),  -np.sin(angle)],
                           [0,              np.sin(angle),  np.cos(angle)]])
    if matrix is None:
        return rotation_x

    return np.dot(matrix, rotation_x)


def create_matrix_rotation_y(angle, matrix = None):
    rotation_y = np.array([[np.cos(angle),  0,              np.sin(angle)],
                           [0,              1,              0],
                           [-np.sin(angle), 0,              np.cos(angle)]])
    if matrix is None:
        return rotation_y

    return np.dot(matrix, rotation_y)


def create_matrix_rotation_z(angle, matrix = None):
    rotation_z = np.array([[np.cos(angle),  -np.sin(angle), 0],
                           [np.sin(angle),  np.cos(angle),  0],
                           [0,              0,              1]])
    if matrix is None:
        return rotation_z

    return np.dot(matrix, rotation_z)


def create_random_rotation():
    return create_matrix_rotation_x(np.random.uniform(0.0, 2*np.pi), create_matrix_rotation_y(np.random.uniform(0.0, 2*np.pi), create_matrix_rotation_z(np.random.uniform(0.0, 2*np.pi))))


def correct_nans(image):
    t1_image_corr = np.array(image)
    isnan_coords = np.where(np.isnan(image))
    if len(isnan_coords[0]) > 0:
        for coord in zip(isnan_coords[0], isnan_coords[1], isnan_coords[2]):
            coord = list(coord)
            region = image[coord[0]-5 : coord[0]+5, coord[1]-5 : coord[1]+5, coord[2]-5 : coord[2]+5]
            t1_image_corr[tuple(coord)] = np.max(region[~np.isnan(region)])
    return t1_image_corr


def load_patient(id, spacing_target=[1, 0.5, 0.5], seg_format='old'):
    assert seg_format in ['old', 'new']

    curr_dir = "/home/fabian/datasets/Hirntumor_von_David/%03.0f"%id
    if not os.path.isfile(os.path.join(curr_dir, "T1KM_m2_bc_ws.nii.gz")):
        return [None]*6
    if not os.path.isfile(os.path.join(curr_dir, "FLAIR_m2_bc_ws.nii.gz")):
        return [None]*6
    if not os.path.isfile(os.path.join(curr_dir, "seg_ce.nii.gz")):
        return [None]*6
    if not os.path.isfile(os.path.join(curr_dir, "seg_edema.nii.gz")):
        return [None]*6
    if not os.path.isfile(os.path.join(curr_dir, "seg_necrosis_ce.nii.gz")):
        return [None]*6
    if not os.path.isfile(os.path.join(curr_dir, "CBV_mutualinfo2_reg.nii.gz")):
        return [None]*6
    if not os.path.isfile(os.path.join(curr_dir, "ADC_mutualinfo2_reg.nii.gz")):
        return [None]*6

    # load image
    itk_img = sitk.ReadImage(os.path.join(curr_dir, "T1KM_m2_bc_ws.nii.gz"))
    flair_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "FLAIR_m2_bc_ws.nii.gz"))).astype(np.float)
    adc_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "ADC_mutualinfo2_reg.nii.gz"))).astype(np.float)
    cbv_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "CBV_mutualinfo2_reg.nii.gz"))).astype(np.float)
    seg_ce = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "seg_ce.nii.gz"))).astype(np.float)
    seg_edema = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "seg_edema.nii.gz"))).astype(np.float)
    seg_necrosis_ce = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "seg_necrosis_ce.nii.gz"))).astype(np.float)
    brain_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "T1_mutualinfo2_bet_mask.nii.gz"))).astype(np.float)
    t1km_img = sitk.GetArrayFromImage(itk_img).astype(np.float)

    assert seg_ce.shape == t1km_img.shape
    assert seg_edema.shape == t1km_img.shape
    assert seg_necrosis_ce.shape == t1km_img.shape
    assert brain_mask.shape == t1km_img.shape
    assert flair_img.shape == t1km_img.shape
    assert adc_img.shape == t1km_img.shape
    assert cbv_img.shape == t1km_img.shape

    # correct nans
    t1km_img = correct_nans(t1km_img)
    flair_img = correct_nans(flair_img)
    cbv_img = correct_nans(cbv_img)
    adc_img = correct_nans(adc_img)

    spacing = np.array(itk_img.GetSpacing())[[2, 1, 0]]

    new_shape = (int(spacing[0]/spacing_target[0]*float(t1km_img.shape[0])),
                 int(spacing[1]/spacing_target[1]*float(t1km_img.shape[1])),
                 int(spacing[2]/spacing_target[2]*float(t1km_img.shape[2])))

    tmp = float(128) / np.max(new_shape)
    new_shape_downsampled = (int(np.round(new_shape[0] * tmp)),
                             int(np.round(new_shape[1] * tmp)),
                             int(np.round(new_shape[2] * tmp))
                             )

    # t1_image = resize(t1_image/maxVal, new_shape)

    outside_value_t1km = t1km_img[0,0,0]
    outside_value_flair = flair_img[0,0,0]
    outside_value_adc = adc_img[0,0,0] # not really necessary, outside is 0
    outside_value_cbv = cbv_img[0,0,0] # not really necessary, outside is 0
    print "reshaping images..."
    pool = ThreadPool(9)
    (t1km_img, flair_img, adc_img, cbv_img, seg_ce, seg_edema, seg_necrosis_ce, brain_mask, t1km_downsampled) = \
                                                                                                    pool.map(resize_star,
                                                                                                    [(t1km_img, new_shape, 3, 'edge'),
                                                                                                     (flair_img, new_shape, 3, 'edge'),
                                                                                                     (adc_img, new_shape, 3, 'edge'),
                                                                                                     (cbv_img, new_shape, 3, 'edge'),
                                                                                                     (seg_ce, new_shape, 3, 'edge'),
                                                                                                     (seg_edema, new_shape, 3, 'edge'),
                                                                                                     (seg_necrosis_ce, new_shape, 3, 'edge'),
                                                                                                     (brain_mask, new_shape, 3, 'edge'),
                                                                                                     (t1km_img, new_shape_downsampled, 3, 'edge')])
    pool.close()
    pool.join()
    t1km_img = t1km_img.astype(np.float32)
    t1km_downsampled = t1km_downsampled.astype(np.float32)
    flair_img = flair_img.astype(np.float32)
    adc_img = adc_img.astype(np.float32)
    cbv_img = cbv_img.astype(np.float32)
    seg_ce = np.round(seg_ce).astype(np.int32)
    seg_edema = np.round(seg_edema).astype(np.int32)
    seg_necrosis_ce = np.round(seg_necrosis_ce).astype(np.int32)
    brain_mask = np.round(brain_mask).astype(np.int32)
    seg_ce[seg_ce > 1] = 1
    seg_edema[seg_edema > 1] = 1
    seg_necrosis_ce[seg_necrosis_ce > 1] = 1
    brain_mask[brain_mask > 1] = 1

    t1km_img[brain_mask == 0] = outside_value_t1km
    flair_img[brain_mask == 0] = outside_value_flair
    adc_img[brain_mask == 0] = outside_value_adc
    cbv_img[brain_mask == 0] = outside_value_cbv
    # outside_value = 0
    # t1_image -= outside_value

    # create joint segmentation map (useful for later)
    seg_combined = np.zeros(t1km_img.shape, dtype=np.int32)

    # necrosis is 3, ce tumor is 2, edema is 1
    if seg_format == 'old':
        seg_combined[seg_necrosis_ce == 1] = 3
        seg_combined[seg_ce == 1] = 2
        seg_combined[seg_edema == 1] = 1
    elif seg_format == 'new':
        seg_combined[brain_mask == 1] = 1
        seg_combined[seg_necrosis_ce == 1] = 4
        seg_combined[seg_ce == 1] = 3
        seg_combined[seg_edema == 1] = 2
    else:
        return [None]*6

    return t1km_img, flair_img, adc_img, cbv_img, t1km_downsampled, seg_combined

def load_patient_with_t1(id, spacing_target=[1, 1, 1], seg_format='old'):
    assert seg_format in ['old', 'new']

    curr_dir = "/home/fabian/datasets/Hirntumor_von_David/%03.0f"%id
    if not os.path.isfile(os.path.join(curr_dir, "T1KM_m2_bc_ws.nii.gz")):
        return [None]*7
    if not os.path.isfile(os.path.join(curr_dir, "T1_m2_bc_ws.nii.gz")):
        return [None]*7
    if not os.path.isfile(os.path.join(curr_dir, "FLAIR_m2_bc_ws.nii.gz")):
        return [None]*7
    if not os.path.isfile(os.path.join(curr_dir, "seg_ce.nii.gz")):
        return [None]*7
    if not os.path.isfile(os.path.join(curr_dir, "seg_edema.nii.gz")):
        return [None]*7
    if not os.path.isfile(os.path.join(curr_dir, "seg_necrosis_ce.nii.gz")):
        return [None]*7
    if not os.path.isfile(os.path.join(curr_dir, "CBV_mutualinfo2_reg.nii.gz")):
        return [None]*7
    if not os.path.isfile(os.path.join(curr_dir, "ADC_mutualinfo2_reg.nii.gz")):
        return [None]*7

    # load image
    itk_img = sitk.ReadImage(os.path.join(curr_dir, "T1KM_m2_bc_ws.nii.gz"))
    flair_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "FLAIR_m2_bc_ws.nii.gz"))).astype(np.float)
    adc_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "ADC_mutualinfo2_reg.nii.gz"))).astype(np.float)
    cbv_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "CBV_mutualinfo2_reg.nii.gz"))).astype(np.float)
    seg_ce = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "seg_ce.nii.gz"))).astype(np.float)
    seg_edema = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "seg_edema.nii.gz"))).astype(np.float)
    seg_necrosis_ce = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "seg_necrosis_ce.nii.gz"))).astype(np.float)
    brain_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "T1_mutualinfo2_bet_mask.nii.gz"))).astype(np.float)
    t1km_img = sitk.GetArrayFromImage(itk_img).astype(np.float)
    t1_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(curr_dir, "T1_m2_bc_ws.nii.gz"))).astype(np.float)

    assert seg_ce.shape == t1km_img.shape
    assert seg_edema.shape == t1km_img.shape
    assert seg_necrosis_ce.shape == t1km_img.shape
    assert brain_mask.shape == t1km_img.shape
    assert flair_img.shape == t1km_img.shape
    assert adc_img.shape == t1km_img.shape
    assert cbv_img.shape == t1km_img.shape
    assert t1_img.shape == t1km_img.shape

    t1km_img = extract_brain_region(t1km_img, brain_mask)
    flair_img = extract_brain_region(flair_img, brain_mask)
    adc_img = extract_brain_region(adc_img, brain_mask)
    cbv_img = extract_brain_region(cbv_img, brain_mask)
    seg_ce = extract_brain_region(seg_ce, brain_mask)
    seg_edema = extract_brain_region(seg_edema, brain_mask)
    seg_necrosis_ce = extract_brain_region(seg_necrosis_ce, brain_mask)
    t1_img = extract_brain_region(t1_img, brain_mask)
    brain_mask = extract_brain_region(brain_mask, brain_mask)

    # correct nans
    t1km_img = correct_nans(t1km_img)
    flair_img = correct_nans(flair_img)
    cbv_img = correct_nans(cbv_img)
    adc_img = correct_nans(adc_img)
    t1_img = correct_nans(t1_img)

    spacing = np.array(itk_img.GetSpacing())[[2, 1, 0]]

    new_shape = (int(spacing[0]/spacing_target[0]*float(t1km_img.shape[0])),
                 int(spacing[1]/spacing_target[1]*float(t1km_img.shape[1])),
                 int(spacing[2]/spacing_target[2]*float(t1km_img.shape[2])))

    tmp = float(128) / np.max(new_shape)
    new_shape_downsampled = (int(np.round(new_shape[0] * tmp)),
                             int(np.round(new_shape[1] * tmp)),
                             int(np.round(new_shape[2] * tmp))
                             )

    # t1_image = resize(t1_image/maxVal, new_shape)

    outside_value_t1km = t1km_img[0,0,0]
    outside_value_t1 = t1_img[0,0,0]
    outside_value_flair = flair_img[0,0,0]
    outside_value_adc = adc_img[0,0,0] # not really necessary, outside is 0
    outside_value_cbv = cbv_img[0,0,0] # not really necessary, outside is 0
    print "reshaping images..."
    pool = ThreadPool(10)
    (t1km_img, flair_img, adc_img, cbv_img, seg_ce, seg_edema, seg_necrosis_ce, brain_mask, t1km_downsampled, t1_img) = \
                                                                                                    pool.map(resize_star,
                                                                                                    [(t1km_img, new_shape, 3, 'edge'),
                                                                                                     (flair_img, new_shape, 3, 'edge'),
                                                                                                     (adc_img, new_shape, 3, 'edge'),
                                                                                                     (cbv_img, new_shape, 3, 'edge'),
                                                                                                     (seg_ce, new_shape, 3, 'edge'),
                                                                                                     (seg_edema, new_shape, 3, 'edge'),
                                                                                                     (seg_necrosis_ce, new_shape, 3, 'edge'),
                                                                                                     (brain_mask, new_shape, 3, 'edge'),
                                                                                                     (t1km_img, new_shape_downsampled, 3, 'edge'),
                                                                                                     (t1_img, new_shape, 3, 'edge')])
    pool.close()
    pool.join()
    t1km_img = t1km_img.astype(np.float32)
    t1_img = t1_img.astype(np.float32)
    t1km_downsampled = t1km_downsampled.astype(np.float32)
    flair_img = flair_img.astype(np.float32)
    adc_img = adc_img.astype(np.float32)
    cbv_img = cbv_img.astype(np.float32)
    seg_ce = np.round(seg_ce).astype(np.int32)
    seg_edema = np.round(seg_edema).astype(np.int32)
    seg_necrosis_ce = np.round(seg_necrosis_ce).astype(np.int32)
    brain_mask = np.round(brain_mask).astype(np.int32)

    seg_ce[seg_ce > 1] = 1
    seg_edema[seg_edema > 1] = 1
    seg_necrosis_ce[seg_necrosis_ce > 1] = 1
    brain_mask[brain_mask > 1] = 1

    t1km_img[brain_mask == 0] = outside_value_t1km
    flair_img[brain_mask == 0] = outside_value_flair
    adc_img[brain_mask == 0] = outside_value_adc
    cbv_img[brain_mask == 0] = outside_value_cbv
    t1_img[brain_mask == 0] = outside_value_t1
    # outside_value = 0
    # t1_image -= outside_value

    # create joint segmentation map (useful for later)
    seg_combined = np.zeros(t1km_img.shape, dtype=np.int32)

    # necrosis is 3, ce tumor is 2, edema is 1
    if seg_format == 'old':
        seg_combined[seg_necrosis_ce == 1] = 3
        seg_combined[seg_ce == 1] = 2
        seg_combined[seg_edema == 1] = 1
    elif seg_format == 'new':
        seg_combined[brain_mask == 1] = 1
        seg_combined[seg_necrosis_ce == 1] = 4
        seg_combined[seg_ce == 1] = 3
        seg_combined[seg_edema == 1] = 2
    else:
        return [None]*7

    return t1_img, t1km_img, flair_img, adc_img, cbv_img, t1km_downsampled, seg_combined


def resize_star(args):
    return resize(*args)

def reshape_and_save_all_patients():
    path = "/home/fabian/datasets/Hirntumor_von_David/"
    for i in range(150):
        print i
        t1km_img, flair_img, adc_img, cbv_img, t1km_downsampled, seg_combined = load_patient(i, seg_format='new')
        if t1km_img is not None:
            np.save(path + "%03.0f"%i + "/T1KM_m2_bc_ws", t1km_img)
            np.save(path + "%03.0f"%i + "/T1KM_m2_bc_ws_downsampled128", t1km_downsampled)
            np.save(path + "%03.0f"%i + "/FLAIR_m2_bc_ws", flair_img)
            np.save(path + "%03.0f"%i + "/ADC_mutualinfo2_reg", adc_img)
            np.save(path + "%03.0f"%i + "/CBV_mutualinfo2_reg", cbv_img)
            np.save(path + "%03.0f"%i + "/seg_all", seg_combined)

def reshape_and_save_all_patients_with_t1():
    path = "/media/fabian/DeepLearningData/datasets/Hirntumor_raw_data/"
    for i in range(150):
        print i
        t1_img, t1km_img, flair_img, adc_img, cbv_img, t1km_downsampled, seg_combined = load_patient_with_t1(i, seg_format='new')
        if t1km_img is not None:
            np.save(path + "patient_%03.0d_t1km_data" % i, t1km_img)
            np.save(path + "patient_%03.0d_t1_data" % i, t1_img)
            np.save(path + "patient_%03.0d_adc_data" % i, adc_img)
            np.save(path + "patient_%03.0d_flair_data" % i, flair_img)
            np.save(path + "patient_%03.0d_cbv_data" % i, cbv_img)
            np.save(path + "patient_%03.0d_t1km_downsampled_128_data" % i, t1km_downsampled)
            np.save(path + "patient_%03.0d_segmentation" % i, seg_combined)


def load_patient_resampled(id):
    curr_dir = "/home/fabian/datasets/Hirntumor_von_David/%03.0f"%id
    if not os.path.isfile(curr_dir + "/T1KM_m2_bc_ws.npy"):
        return [None]*6
    t1km_img = np.load(curr_dir + "/T1KM_m2_bc_ws.npy")
    flair_img = np.load(curr_dir + "/FLAIR_m2_bc_ws.npy")
    adc_img = np.load(curr_dir + "/ADC_mutualinfo2_reg.npy")
    cbv_img = np.load(curr_dir + "/CBV_mutualinfo2_reg.npy")
    seg_combined = np.load(curr_dir + "/seg_all.npy")
    t1km_downsampled = np.load(curr_dir + "/T1KM_m2_bc_ws_downsampled128.npy")
    return t1km_img, flair_img, adc_img, cbv_img, t1km_downsampled, seg_combined

def extract_brain_region(image, segmentation, outside_value=0):
    brain_voxels = np.where(segmentation != outside_value)
    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))

    # resize images
    resizer = (slice(minZidx, maxZidx), slice(minXidx, maxXidx), slice(minYidx, maxYidx))
    return image[resizer]


def center_crop_image(image, output_size):
    center = np.array(image.shape[1:])/2
    return image[:, int(center[0]-output_size[0]/2.):int(center[0]+output_size[0]/2.), int(center[1]-output_size[1]/2.):int(center[1]+output_size[1]/2.)]
