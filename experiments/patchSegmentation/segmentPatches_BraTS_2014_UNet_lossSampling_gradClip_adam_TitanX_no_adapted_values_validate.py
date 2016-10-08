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
sys.path.append("../../generators/")
sys.path.append("../../utils/")
import UNet
from general_utils import calculate_validation_metrics
from dataset_utility import center_crop_image
from data_generators import SegmentationBatchGeneratorFromRawData
import theano.tensor

def softmax_helper(x):
    e_x = theano.tensor.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def load_pat(id):
    a = np.load("/media/fabian/DeepLearningData/datasets/BraTS_2014/HGG/adapted_value_range/%03.0d.npy"%id)
    return a[0], a[1], a[2], a[3], a[4]

def load_pat_orig(id):
    a = np.load("/media/fabian/DeepLearningData/datasets/BraTS_2014/HGG/original_value_range/%03.0d.npy"%id)
    return a[0], a[1], a[2], a[3], a[4]

validation_patients = [7,  97, 235,  12, 231, 200, 177, 104, 247,  41, 237, 24, 118, 198, 103,   6, 243,  35,   0,  18, 112, 180,  25, 157,  69]
experiment_name = "segmentPatches_BraTS_2014_UNet_lossSampling_gradClip_adam_TitanX_no_adapted_values"
results_folder = "/home/fabian/datasets/Hirntumor_von_David/experiments/results/%s/" % experiment_name
epoch = 37

input_shape = (368, 352)

print "compiling theano functions"
# uild_UNet(n_input_channels=1, BATCH_SIZE=None, num_output_classes=2, pad='same', nonlinearity=lasagne.nonlinearities.elu, input_dim=(128, 128), base_n_filters=64, do_dropout=False):
net = UNet.build_UNet(20, 1, 6, input_dim=input_shape, base_n_filters=16, pad="valid")
output_shape = net["output_segmentation"].output_shape[-2:]
output_layer = net["output_segmentation"]
with open(os.path.join(results_folder, "%s_allLossesNAccur_ep%d.pkl" % (experiment_name, epoch)), 'r') as f:
    tmp = cPickle.load(f)

with open(os.path.join(results_folder, "%s_Params_ep%d.pkl" % (experiment_name, epoch)), 'r') as f:
    params = cPickle.load(f)
    lasagne.layers.set_all_param_values(output_layer, params)

import theano.tensor as T
data_sym = T.tensor4()

output = softmax_helper(lasagne.layers.get_output(output_layer, data_sym, deterministic=False))
pred_fn = theano.function([data_sym], output)

for patient_id in validation_patients:
    print patient_id
    output_folder = os.path.join(results_folder, "%03.0d" % patient_id)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    t1_img, t1km_img, t2_img, flair_img, seg_combined = load_pat_orig(patient_id)
    import matplotlib.pyplot as plt

    t1_img = SegmentationBatchGeneratorFromRawData.resize_image_by_padding(t1_img, input_shape, pad_value=None)
    t1km_img = SegmentationBatchGeneratorFromRawData.resize_image_by_padding(t1km_img, input_shape, pad_value=None)
    t2_img = SegmentationBatchGeneratorFromRawData.resize_image_by_padding(t2_img, input_shape, pad_value=None)
    flair_img = SegmentationBatchGeneratorFromRawData.resize_image_by_padding(flair_img, input_shape, pad_value=None)
    seg_combined = SegmentationBatchGeneratorFromRawData.resize_image_by_padding(seg_combined, input_shape, pad_value=0)

    seg_combined = center_crop_image(seg_combined, output_shape)

    print "predicting image"
    cmap = ListedColormap([(0,0,0), (0,0,1), (0,1,0), (1,0,0), (1,1,0), (0.3, 0.5, 1)])

    z = 2
    data = np.zeros((1, 20, input_shape[0], input_shape[1])).astype(np.float32)
    res = np.zeros((t1km_img.shape[0], output_shape[0], output_shape[1], 6))
    while z < t1km_img.shape[0]-2:
        data[0,0:5] = t1_img[z-2:z+3]
        data[0,5:10] = t1km_img[z-2:z+3]
        data[0,10:15] = t2_img[z-2:z+3]
        data[0,15:20] = flair_img[z-2:z+3]
        pred = pred_fn(data).transpose((0,2,3,1)).reshape((output_shape[0], output_shape[1], 6))
        res[z] = pred
        z += 1

    res = res[2:t1km_img.shape[0]-2]
    seg_combined = seg_combined[2:t1km_img.shape[0]-2]
    res_img = res.argmax(-1)

    print "calculating metrics"
    acc, metrics_by_class = calculate_validation_metrics(res, seg_combined, {0: 'background', 1: 'brain', 2: 'necrosis', 3: 'edema', 4: 'non-enhancing tumor', 5:'enhancing tumor'}, 6)
    with open(os.path.join(output_folder, "validation_metrics.txt"), 'w') as f:
        f.write("The overall accuracy on this dataset was: \t%f\n\n" % acc)
        for c in metrics_by_class.keys():
            f.write("Results for label: %s\n" % c)
            for metric in metrics_by_class[c].keys():
                f.write("%s: \t%f\n" % (metric, metrics_by_class[c][metric]))
            f.write("\n")

    output_folder_images = os.path.join(output_folder, "segmentation_slices")
    if not os.path.isdir(output_folder_images):
        os.mkdir(output_folder_images)

    np.save(os.path.join(output_folder, "seg_gt.npy"), seg_combined)
    np.save(os.path.join(output_folder, "seg_pred.npy"), res_img)

    t1_img = center_crop_image(t1_img, output_shape)
    t1km_img = center_crop_image(t1km_img, output_shape)
    t2_img = center_crop_image(t2_img, output_shape)
    flair_img = center_crop_image(flair_img, output_shape)

    print "writing segmentation images"
    for i in range(2, t1km_img.shape[0]-2):
        res_img[i-2][0,0:6] = [0,1,2,3,4,5]
        seg_combined[i-2][0,0:6] = [0,1,2,3,4,5]
        errors = seg_combined[i-2] == res_img[i-2]
        errors[0, 0:2] = [True, False]
        plt.figure(figsize=(24,12))
        plt.subplot(2,4,1)
        plt.imshow(t1_img[i], cmap="gray")
        plt.subplot(2,4,2)
        plt.imshow(t1km_img[i], cmap="gray")
        plt.subplot(2,4,3)
        plt.imshow(t2_img[i], cmap="gray")
        plt.subplot(2,4,4)
        plt.imshow(flair_img[i], cmap="gray")
        plt.subplot(2,4,5)
        plt.imshow(res_img[i-2], cmap=cmap)
        plt.subplot(2,4,6)
        plt.imshow(seg_combined[i-2], cmap=cmap)
        plt.subplot(2,4,7)
        plt.imshow(seg_combined[i-2] == res_img[i-2], cmap="gray")
        plt.savefig(os.path.join(output_folder_images, "patient%d_segWholeDataset_z%03.0f"%(patient_id, i)))
        plt.close()
