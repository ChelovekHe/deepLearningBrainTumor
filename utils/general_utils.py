__author__ = 'fabian'
import numpy as np
from multiprocessing import Process
from multiprocessing import Queue as MPQueue
from time import sleep
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import cPickle
import lasagne
import theano
import theano.tensor
import os
import SimpleITK as sitk
from multiprocessing import Pool as ThreadPool
from skimage.transform import resize
from matplotlib.colors import ListedColormap
import sys
sys.path.append("../experiments/patchSegmentation/")
sys.path.append("../neural_networks/")
sys.path.append("../dataset_utils/")
import UNet
from dataset_utility import load_patient_resampled
from sklearn.metrics import roc_auc_score

def calculate_validation_metrics(probas_pred, image_gt, class_labels=None, num_classes=5):
    classes = np.arange(probas_pred.shape[-1])
    # determine valid classes (those that actually appear in image_gt). Some images may miss some classes
    classes = [c for c in classes if np.sum(image_gt==c) != 0]
    image_pred = probas_pred.argmax(-1)
    assert image_gt.shape == image_pred.shape
    accuracy = np.sum(image_gt == image_pred) / float(image_pred.size)
    class_metrics = {}
    y_true = convert_seg_flat_to_binary_label_indicator_array(image_gt.ravel(), num_classes).astype(int)[:, classes]
    y_pred = probas_pred.transpose(3, 0, 1, 2).reshape(num_classes, -1).transpose(1, 0)[:, classes]
    scores = roc_auc_score(y_true, y_pred, None)
    for i, c in enumerate(classes):
        true_positives = np.sum((image_gt == c) & (image_pred == c))
        true_negatives = np.sum((image_gt != c) & (image_pred != c))
        false_positives = np.sum((image_gt != c) & (image_pred == c))
        false_negatives = np.sum((image_gt == c) & (image_pred != c))
        specificity = true_negatives / float(true_negatives + false_positives)
        sensitivity = true_positives / float(true_positives + false_negatives)
        f1_score = 2. * true_positives / float(2 * true_positives + false_positives + false_negatives)
        jaccard_index = np.sum((image_pred == c) & (image_gt == c)) / float(np.sum((image_gt == c) | (image_pred == c)))
        dice_score = 2. * np.sum((image_pred == c) & (image_gt == c)) / float(np.sum(image_gt == c) + np.sum(image_pred == c))
        label = c
        if class_labels is not None and c in class_labels.keys():
            label = class_labels[c]
        class_metrics[label] = {'specificity': specificity,
                                'sensitivity': sensitivity,
                                'f1_score': f1_score,
                                'jaccard_index': jaccard_index,
                                'dice_score': dice_score,
                                'roc_auc': scores[i]}
    return accuracy, class_metrics

def convert_seg_flat_to_binary_label_indicator_array(seg_flat, num_classes=5):
    seg2 = np.zeros((len(seg_flat), num_classes))
    for i in xrange(seg2.shape[0]):
        seg2[i, int(seg_flat[i])] = 1
    return seg2


def predict_whole_dataset(patient_id):
    t1km_img, flair_img, adc_img, cbv_img, t1km_downsampled, seg_combined = load_patient_resampled(patient_id)
    import matplotlib.pyplot as plt
    assert t1km_img.shape == flair_img.shape
    assert t1km_img.shape == adc_img.shape
    assert t1km_img.shape == cbv_img.shape
    shape_0_new = t1km_img.shape[1] - t1km_img.shape[1]%16
    shape_1_new = t1km_img.shape[2] - t1km_img.shape[2]%16
    t1km_img = t1km_img[:, :shape_0_new, :shape_1_new]
    flair_img = flair_img[:, :shape_0_new, :shape_1_new]
    adc_img = adc_img[:, :shape_0_new, :shape_1_new]
    cbv_img = cbv_img[:, :shape_0_new, :shape_1_new]
    seg_combined = seg_combined[:, :shape_0_new, :shape_1_new]

    # uild_UNet(n_input_channels=1, BATCH_SIZE=None, num_output_classes=2, pad='same', nonlinearity=lasagne.nonlinearities.elu, input_dim=(128, 128), base_n_filters=64, do_dropout=False):
    net = UNet.build_UNet(20, 1, 5, input_dim=(t1km_img.shape[1], t1km_img.shape[2]), base_n_filters=16)
    output_layer = net["output_flattened"]
    with open("/home/fabian/datasets/Hirntumor_von_David/experiments/results/segment_tumor_v0.2_UNet_lossSampling/segment_tumor_v0.2_Unet_lossSampling_Params_ep30.pkl", 'r') as f:
        params = cPickle.load(f)
        lasagne.layers.set_all_param_values(output_layer, params)

    import theano.tensor as T
    data_sym = T.tensor4()

    output = lasagne.layers.get_output(output_layer, data_sym, deterministic=False)
    pred_fn = theano.function([data_sym], output)

    cmap = ListedColormap([(0,0,0), (0,0,1), (0,1,0), (1,0,0), (1,1,0)])

    z = 2
    data = np.zeros((1, 20, t1km_img.shape[1], t1km_img.shape[2])).astype(np.float32)
    res = np.zeros((t1km_img.shape[0], t1km_img.shape[1] - t1km_img.shape[1]%16, t1km_img.shape[2] - t1km_img.shape[2]%16))
    while z < t1km_img.shape[0]-2:
        data[0,0:5] = t1km_img[z-2:z+3]
        data[0,5:10] = flair_img[z-2:z+3]
        data[0,10:15] = adc_img[z-2:z+3]
        data[0,15:20] = cbv_img[z-2:z+3]
        pred = pred_fn(data).argmax(-1).reshape((t1km_img.shape[1] - t1km_img.shape[1]%16, t1km_img.shape[2] - t1km_img.shape[2]%16))
        res[z] = pred
        z += 1

    for i in xrange(t1km_img.shape[0]):
        res[i][0,0:5] = [0,1,2,3,4]
        seg_combined[i][0,0:5] = [0,1,2,3,4]
        errors = seg_combined[i] == res[i]
        errors[0, 0:2] = [True, False]
        plt.figure(figsize=(24,12))
        plt.subplot(2,4,1)
        plt.imshow(t1km_img[i], cmap="gray")
        plt.subplot(2,4,2)
        plt.imshow(flair_img[i], cmap="gray")
        plt.subplot(2,4,3)
        plt.imshow(adc_img[i], cmap="gray")
        plt.subplot(2,4,4)
        plt.imshow(cbv_img[i], cmap="gray")
        plt.subplot(2,4,5)
        plt.imshow(res[i], cmap=cmap)
        plt.subplot(2,4,6)
        plt.imshow(seg_combined[i], cmap=cmap)
        plt.subplot(2,4,7)
        plt.imshow(seg_combined[i] == res[i], cmap="gray")
        plt.savefig("/home/fabian/datasets/Hirntumor_von_David/experiments/some_images/patient%d_segWholeDataset_z%03.0f"%(patient_id, i))
        plt.close()

def predict_whole_dataset_patchwise(t1km_img, flair_img, adc_img, cbv_img, seg_combined):
    import matplotlib.pyplot as plt
    assert t1km_img.shape == flair_img.shape
    assert t1km_img.shape == adc_img.shape
    assert t1km_img.shape == cbv_img.shape

    patch_size = (128, 128)

    net = None
    # build_residual_UNet_noBN(4, 1, 4, patch_size, 16)
    output_layer = net["output"]

    with open("../results/segment_tumor_v0.1_residualUnet_noBN_Params_ep0.pkl", 'r') as f:
        params = cPickle.load(f)
        lasagne.layers.set_all_param_values(output_layer, params)

    import theano.tensor as T
    data_sym = T.tensor4()

    output = lasagne.layers.get_output(output_layer, data_sym, deterministic=False)
    pred_fn = theano.function([data_sym], output)
    data = np.zeros((1, 4, patch_size[0], patch_size[1])).astype(np.float32)
    res = np.zeros((t1km_img.shape[0], 528, 528))
    z = 0
    while z < t1km_img.shape[0]:
        y = 0
        while y < t1km_img.shape[2] - patch_size[1]:
            x = 0
            while x < t1km_img.shape[1] - patch_size[0]:
                patch_t1km = t1km_img[z, x:x+patch_size[0], y:y+patch_size[1]]
                patch_flair = flair_img[z, x:x+patch_size[0], y:y+patch_size[1]]
                patch_adc = adc_img[z, x:x+patch_size[0], y:y+patch_size[1]]
                patch_cbv = cbv_img[z, x:x+patch_size[0], y:y+patch_size[1]]
                data[0,0] = patch_t1km
                data[0,1] = patch_flair
                data[0,2] = patch_adc
                data[0,3] = patch_cbv
                res[z, x:x+patch_size[0], y:y+patch_size[1]] = pred_fn(data).argmax(-1).reshape(patch_size)
                x += patch_size[0]
            y += patch_size[1]
        z += 1

    for i in xrange(t1km_img.shape[0]):
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(res[i])
        plt.subplot(1,2,2)
        plt.imshow(seg_combined[i])
        plt.savefig("../some_images/segWholeDataset_z%03.0f"%i)
        plt.close()
    return res



def find_entries_in_array(entries, myarray):
    entries = np.array(entries)
    values = np.arange(np.max(myarray) + 1)
    lut = np.zeros(len(values),'bool')
    lut[entries.astype("int")] = True
    return np.take(lut, myarray.astype(int))

def pad_3d_image(image, pad_size, pad_value=None):
    '''
    :param pad_size: must be a np array with 3 entries, one for each dimension of the image
    '''
    image_shape = image.shape
    new_shape = np.array(list(image_shape)) + pad_size
    if pad_value is None:
        pad_value = image[0,0,0]
    new_image = np.ones(new_shape) * pad_value
    new_image[pad_size[0]/2.:pad_size[0]/2.+image_shape[0], pad_size[1]/2.:pad_size[1]/2.+image_shape[1], pad_size[2]/2.:pad_size[2]/2.+image_shape[2]] = image
    return new_image
