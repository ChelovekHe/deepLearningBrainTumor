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
import os
import SimpleITK as sitk
from multiprocessing import Pool as ThreadPool
from skimage.transform import resize
from matplotlib.colors import ListedColormap
import sys
sys.path.append("../../experiments/patchSegmentation/")
sys.path.append("../../neural_networks/")
sys.path.append("../../dataset_utils/")
sys.path.append("../../utils/")
import UNet
from dataset_utility import load_patient_resampled
from general_utils import calculate_validation_metrics
import theano.tensor

def softmax_helper(x):
    e_x = theano.tensor.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

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
    output_layer = net["output_segmentation"]
    with open("/home/fabian/datasets/Hirntumor_von_David/experiments/results/segmentPatches_UNet_v0.2_lossSampling/segment_tumor_v0.2_Unet_lossSampling_allLossesNAccur_ep4.pkl", 'r') as f:
        tmp = cPickle.load(f)

    with open("/home/fabian/datasets/Hirntumor_von_David/experiments/results/segmentPatches_UNet_v0.2_lossSampling/segment_tumor_v0.2_Unet_lossSampling_Params_ep4.pkl", 'r') as f:
        params = cPickle.load(f)
        lasagne.layers.set_all_param_values(output_layer, params)

    import theano.tensor as T
    data_sym = T.tensor4()

    output = softmax_helper(lasagne.layers.get_output(output_layer, data_sym, deterministic=False))
    pred_fn = theano.function([data_sym], output)

    cmap = ListedColormap([(0,0,0), (0,0,1), (0,1,0), (1,0,0), (1,1,0)])

    z = 2
    data = np.zeros((1, 20, t1km_img.shape[1], t1km_img.shape[2])).astype(np.float32)
    res = np.zeros((t1km_img.shape[0], t1km_img.shape[1] - t1km_img.shape[1]%16, t1km_img.shape[2] - t1km_img.shape[2]%16, 5))
    while z < t1km_img.shape[0]-2:
        data[0,0:5] = t1km_img[z-2:z+3]
        data[0,5:10] = flair_img[z-2:z+3]
        data[0,10:15] = adc_img[z-2:z+3]
        data[0,15:20] = cbv_img[z-2:z+3]
        pred = pred_fn(data).transpose((0,2,3,1)).reshape((t1km_img.shape[1] - t1km_img.shape[1]%16, t1km_img.shape[2] - t1km_img.shape[2]%16, 5))
        res[z] = pred
        z += 1

    res = res[2:t1km_img.shape[0]-2]
    seg_combined = seg_combined[2:t1km_img.shape[0]-2]
    res_img = res.argmax(-1)

    acc, metrics_by_class = calculate_validation_metrics(res, seg_combined, {0: 'background', 1: 'brain', 2: 'edema', 3: 'ce_tumor', 4: 'necrosis'})

    for i in xrange(t1km_img.shape[0]):
        res_img[i][0,0:5] = [0,1,2,3,4]
        seg_combined[i][0,0:5] = [0,1,2,3,4]
        errors = seg_combined[i] == res_img[i]
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
        plt.imshow(res_img[i], cmap=cmap)
        plt.subplot(2,4,6)
        plt.imshow(seg_combined[i], cmap=cmap)
        plt.subplot(2,4,7)
        plt.imshow(seg_combined[i] == res_img[i], cmap="gray")
        plt.savefig("/home/fabian/datasets/Hirntumor_von_David/experiments/some_images/patient%d_segWholeDataset_z%03.0f"%(patient_id, i))
        plt.close()
