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
from dataset_utility import center_crop_image
from data_generators import SegmentationBatchGeneratorFromRawData
import theano.tensor
from sklearn.metrics import roc_auc_score
from general_utils import convert_seg_flat_to_binary_label_indicator_array
from scipy import ndimage
from medpy import metric
from dataset_utility import load_patient_resampled


def calculate_validation_metrics(image_pred, image_gt):
    image_gt = np.array(image_gt)
    image_pred = np.array(image_pred)
    def calculate_metrics(mask1, mask2):
        true_positives = metric.obj_tpr(mask1, mask2)
        false_positives = metric.obj_fpr(mask1, mask2)
        dc = metric.dc(mask1, mask2)
        hd = metric.hd(mask1, mask2)
        precision = metric.precision(mask1, mask2)
        recall = metric.recall(mask1, mask2)
        ravd = metric.ravd(mask1, mask2)
        assd = metric.assd(mask1, mask2)
        asd = metric.asd(mask1, mask2)
        return true_positives, false_positives, dc, hd, precision, recall, ravd, assd, asd

    # after post processing we have: 0: background, 1 necrosis, 2 edema, 3 non enhancing tumor, 4 enhancing tumor
    class_labels = {
        0: 'background',
        1: 'edema',
        2: 'tumor',
        3: 'necrosis'
    }
    num_classes = 4
    classes = np.arange(4)

    # determine valid classes (those that actually appear in image_gt). Some images may miss some classes
    classes = [c for c in classes if np.sum(image_gt==c) != 0]
    assert image_gt.shape == image_pred.shape
    accuracy = np.sum(image_gt == image_pred) / float(image_pred.size)
    class_metrics = {}

    # complete tumor
    mask1 = (image_gt==1) | (image_gt==2) | (image_gt==3)
    mask2 = (image_pred==1) | (image_pred==2) | (image_pred==3)
    if np.sum(mask1) != 0 and np.sum(mask2) != 0:
        true_positives, false_positives, dc, hd, precision, recall, ravd, assd, asd = calculate_metrics(mask1, mask2)
        label = "complete tumor"
        class_metrics[label] = {'true_positives': true_positives,
                                'false_positives': false_positives,
                                'DICE\t\t': dc,
                                'Hausdorff dist': hd,
                                'precision\t': precision,
                                'recall\t\t': recall,
                                'rel abs vol diff': ravd,
                                'avg surf dist symm': assd,
                                'avg surf dist\t': asd}

    # tumor core
    mask1 = (image_gt==3) | (image_gt==2)
    mask2 = (image_pred==3) | (image_pred==2)
    if np.sum(mask1) != 0 and np.sum(mask2) != 0:
        true_positives, false_positives, dc, hd, precision, recall, ravd, assd, asd = calculate_metrics(mask1, mask2)
        label = "tumor core"
        class_metrics[label] = {'true_positives': true_positives,
                                'false_positives': false_positives,
                                'DICE\t\t': dc,
                                'Hausdorff dist': hd,
                                'precision\t': precision,
                                'recall\t\t': recall,
                                'rel abs vol diff': ravd,
                                'avg surf dist symm': assd,
                                'avg surf dist\t': asd}

    for i, c in enumerate(classes):
        mask1 = image_gt==c
        mask2 = image_pred==c
        if mask1.sum()==0 or mask2.sum()==0:
            continue
        true_positives, false_positives, dc, hd, precision, recall, ravd, assd, asd = calculate_metrics(mask1, mask2)
        label = c
        if class_labels is not None and c in class_labels.keys():
            label = class_labels[c]
        class_metrics[label] = {'true_positives': true_positives,
                                'false_positives': false_positives,
                                'DICE\t\t': dc,
                                'Hausdorff dist': hd,
                                'precision\t': precision,
                                'recall\t\t': recall,
                                'rel abs vol diff': ravd,
                                'avg surf dist symm': assd,
                                'avg surf dist\t': asd}
    return accuracy, class_metrics

def create_ball_3d(size=11):
    ball = np.zeros((size, size, size))
    center_coords = np.array([(size-1)/2., (size-1)/2., (size-1)/2.])
    radius = (size-1)/2.
    for x in xrange(size):
        for y in xrange(size):
            for z in xrange(size):
                dist = np.linalg.norm(np.array([x, y, z])-center_coords)
                if dist <= radius:
                    ball[x, y, z] = 1
    return ball

def post_process_prediction(prediction):
    # find connected components and remove small ones
    # create structure element (ball)
    str_el = create_ball_3d(3)
    img_2 = ndimage.binary_opening(prediction >= 2, str_el)
    connected_components, n_components = ndimage.label(img_2) # 0 and 1 are background/brain
    discard_components = []
    component_sizes = []
    all_components = np.arange(n_components)+1
    for component in all_components:
        size_of_component = np.sum(connected_components == component)
        if size_of_component < 3000:
            discard_components.append(component)
        component_sizes.append(size_of_component)
    if len(discard_components) == n_components:
        discard_components = discard_components[discard_components!=np.argmax(component_sizes)]
    keep_components = [i for i in all_components if i not in discard_components]
    new_mask = np.zeros(prediction.shape, dtype=bool)
    for keep_me in keep_components:
        mask = ndimage.binary_dilation(connected_components == keep_me, create_ball_3d(5))
        new_mask = (new_mask | mask)
    prediction_cleaned = np.zeros(prediction.shape, dtype=np.int32)
    prediction_cleaned[new_mask] = prediction[new_mask]
    prediction_cleaned[prediction_cleaned == 1] = 0
    prediction_cleaned[prediction_cleaned > 0] -= 1
    return prediction_cleaned


def softmax_helper(x):
    e_x = theano.tensor.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def estimate_tumor_sizes(train_patient_ids, patient_loading_fun):
    sizes = []
    number_of_tumors = []
    for id in train_patient_ids:
        _, _, _, _, seg = patient_loading_fun(id)
        components, n_components = ndimage.label(seg > 1, ndimage.morphology.generate_binary_structure(3, 3))
        number_of_tumors.append(n_components)
        sizes += [np.sum(components==s+1) for s in range(n_components)]
    return number_of_tumors, sizes


validation_patients = np.unique([ 75,   1,  67,   1, 127, 120,  94, 131,  78,  74,  62,  10,  65, 47, 124])
experiment_name = "segment_tumor_Unet_lossSampling_gradClip_adam_TitanX"
results_folder = "/home/fabian/datasets/Hirntumor_von_David/experiments/results/%s/" % experiment_name
epoch = 26

all_official_metrics = np.zeros((len(validation_patients), 13))
ctr=0

for patient_id in validation_patients:
    print patient_id
    output_folder = os.path.join(results_folder, "%03.0d" % patient_id)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
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

    print "compiling theano functions"
    # uild_UNet(n_input_channels=1, BATCH_SIZE=None, num_output_classes=2, pad='same', nonlinearity=lasagne.nonlinearities.elu, input_dim=(128, 128), base_n_filters=64, do_dropout=False):
    net = UNet.build_UNet(20, 1, 5, input_dim=(t1km_img.shape[1], t1km_img.shape[2]), base_n_filters=16)
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

    print "predicting image"
    cmap = ListedColormap([(0,0,0), (0,0,1), (0,1,0), (1,0,0), (1,1,0), (0.3, 0.5, 1)])

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

    print "post processing"
    image_pred_postprocessed = post_process_prediction(res.argmax(-1))
    seg_combined[seg_combined==0] = 1
    seg_combined -= 1
    seg_combined = seg_combined.astype(np.int32)

    np.save(os.path.join(output_folder, "seg_gt.npy"), seg_combined)
    np.save(os.path.join(output_folder, "seg_pred.npy"), image_pred_postprocessed)

    print "calculating metrics"
    acc, metrics_by_class = calculate_validation_metrics(image_pred_postprocessed, seg_combined)
    with open(os.path.join(output_folder, "validation_metrics.txt"), 'w') as f:
        f.write("The overall accuracy on this dataset was: \t%f\n\n" % acc)
        for c in metrics_by_class.keys():
            f.write("Results for label: %s\n" % c)
            for metrc in metrics_by_class[c].keys():
                f.write("%s: \t%f\n" % (metrc, metrics_by_class[c][metrc]))
            f.write("\n")

    if "complete tumor" in metrics_by_class.keys():
        all_official_metrics[ctr][1] = metrics_by_class["complete tumor"]["DICE\t\t"]
        all_official_metrics[ctr][4] = metrics_by_class["complete tumor"]["precision\t"]
        all_official_metrics[ctr][7] = metrics_by_class["complete tumor"]["recall\t\t"]
        all_official_metrics[ctr][10] = metrics_by_class["complete tumor"]["Hausdorff dist"]
    else:
        all_official_metrics[ctr][1] = 999
        all_official_metrics[ctr][4] = 999
        all_official_metrics[ctr][7] = 999
        all_official_metrics[ctr][10] = 999
    if "tumor core" in metrics_by_class.keys():
        all_official_metrics[ctr][2] = metrics_by_class["tumor core"]["DICE\t\t"]
        all_official_metrics[ctr][5] = metrics_by_class["tumor core"]["precision\t"]
        all_official_metrics[ctr][8] = metrics_by_class["tumor core"]["recall\t\t"]
        all_official_metrics[ctr][11] = metrics_by_class["tumor core"]["Hausdorff dist"]
    else:
        all_official_metrics[ctr][2] = 999
        all_official_metrics[ctr][5] = 999
        all_official_metrics[ctr][8] = 999
        all_official_metrics[ctr][11] = 999
    if "tumor" in metrics_by_class.keys():
        all_official_metrics[ctr][3] = metrics_by_class["tumor"]["DICE\t\t"]
        all_official_metrics[ctr][6] = metrics_by_class["tumor"]["precision\t"]
        all_official_metrics[ctr][9] = metrics_by_class["tumor"]["recall\t\t"]
        all_official_metrics[ctr][12] = metrics_by_class["tumor"]["Hausdorff dist"]
    else:
        all_official_metrics[ctr][3] = 999
        all_official_metrics[ctr][6] = 999
        all_official_metrics[ctr][9] = 999
        all_official_metrics[ctr][12] = 999

    output_folder_images = os.path.join(output_folder, "segmentation_slices")
    if not os.path.isdir(output_folder_images):
        os.mkdir(output_folder_images)


    print "writing segmentation images"
    for i in range(2, t1km_img.shape[0]-2):
        image_pred_postprocessed[i-2][0,0:6] = [0,1,2,3,4,5]
        seg_combined[i-2][0,0:6] = [0,1,2,3,4,5]
        errors = seg_combined[i-2] == image_pred_postprocessed[i-2]
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
        plt.imshow(image_pred_postprocessed[i-2], cmap=cmap)
        plt.subplot(2,4,6)
        plt.imshow(seg_combined[i-2], cmap=cmap)
        plt.subplot(2,4,7)
        plt.imshow(seg_combined[i-2] == image_pred_postprocessed[i-2], cmap="gray")
        plt.savefig(os.path.join(output_folder_images, "patient%d_segWholeDataset_z%03.0f"%(patient_id, i)))
        plt.close()
    ctr += 1

np.save(os.path.join(output_folder, "evaluation_metrics.npy"), all_official_metrics)