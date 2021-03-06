__author__ = 'fabian'

import sys
sys.path.append("../../generators")
sys.path.append("../../dataset_utils")
from data_generators import memmapGenerator, memmapGenerator_t1km_flair, memmapGenerator_t1km_flair_adc_cbv, memmapGenerator_t1km_flair_adc_cbv_markers, memmapGenerator_tumorClassRot
from dataset_utility import correct_nans
import numpy as np
import matplotlib.pyplot as plt
import lasagne
import cPickle
from numpy import memmap
from lasagne.layers.dnn import Conv2DDNNLayer
from lasagne.layers import InputLayer, Upscale2DLayer, ConcatLayer, Pool2DLayer

def plot_layer_activations(layer, data, output_fname="../results/layerActivation.png"):
    pred = lasagne.layers.get_output(layer, data).eval()
    n_channels = pred.shape[1]
    plt.figure(figsize=(12, 12))
    plots_per_axis = int(np.ceil(np.sqrt(n_channels)))
    for i in xrange(n_channels):
        plt.subplot(plots_per_axis, plots_per_axis, i+1)
        plt.axis('off')
        plt.imshow(pred[0, i, :, :], cmap="gray", interpolation="nearest")
    plt.savefig(output_fname)
    plt.close()


def plot_all_layer_activations(net, data):
    layers = net.keys()
    for id, layer in enumerate(layers):
        if isinstance(net[layer], Conv2DDNNLayer) or isinstance(layer, Pool2DLayer) or isinstance(layer, Upscale2DLayer) or isinstance(layer, ConcatLayer) or isinstance(layer, InputLayer):
            plot_layer_activations(net[layer], data, "%03.0d-%s.png" % (id, layer))


def printLosses(all_training_losses, all_training_accs, all_validation_losses, all_valid_accur, fname, samplesPerEpoch=10, auc_scores=None, auc_labels=None, ylim_score=None):
    fig, ax1 = plt.subplots(figsize=(16, 12))
    trainLoss_x_values = np.arange(1/float(samplesPerEpoch), len(all_training_losses)/float(samplesPerEpoch)+0.000001, 1/float(samplesPerEpoch))
    val_x_values = np.arange(1, len(all_validation_losses)+0.001, 1)
    ax1.plot(trainLoss_x_values, all_training_losses, 'b--', linewidth=2)
    ax1.plot(val_x_values, all_validation_losses, color='b', linewidth=2)
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    if ylim_score is not None:
        ax1.set_ylim(ylim_score)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax2 = ax1.twinx()
    ax2.plot(trainLoss_x_values, all_training_accs, 'r--', linewidth=2)
    ax2.plot(val_x_values, all_valid_accur, color='r', linewidth=2)
    ax2.set_ylabel('accuracy')
    for t2 in ax2.get_yticklabels():
        t2.set_color('r')
    ax2_legend_text = ['trainAcc', 'validAcc']

    if auc_scores is not None:
        assert len(auc_scores) == len(all_validation_losses)
        num_auc_scores_per_timestep = auc_scores.shape[1]
        for auc_id in xrange(num_auc_scores_per_timestep):
            ax2.plot(val_x_values, auc_scores[:, auc_id], linestyle=":", linewidth=4, markersize=10)
            ax2_legend_text.append(auc_labels[auc_id])

    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax1.legend(['trainLoss', 'validLoss'], loc="center right", bbox_to_anchor=(1.3, 0.4))
    ax2.legend(ax2_legend_text, loc="center right", bbox_to_anchor=(1.3, 0.6))
    plt.savefig(fname)
    plt.close()


def validate_result(img, convLayer):
    img_for_cnn = img[np.newaxis, np.newaxis, :, :]
    filtered_by_cnn = lasagne.layers.get_output(convLayer, img_for_cnn).eval()
    plt.figure(figsize=(12, 12))
    for i in xrange(filtered_by_cnn.shape[1]):
        plt.subplot(int(np.ceil(np.sqrt(filtered_by_cnn.shape[1]))), int(np.ceil(np.sqrt(filtered_by_cnn.shape[1]))), i+1)
        plt.imshow(filtered_by_cnn[0, i, :, :], cmap="gray", interpolation="nearest")
        plt.axis('off')
    plt.savefig("../results/filtered_by_cnn.png")
    plt.close()

    weights_w = convLayer.get_params()[0].get_value()
    weights_b = convLayer.get_params()[1].get_value()

    from scipy.signal import convolve2d
    res_scipy = []
    numFilters = weights_w.shape[0]
    for i in xrange(numFilters):
        weights = weights_w[i, 0, :, :]
        res_scipy.append(lasagne.nonlinearities.rectify(convolve2d(img, weights, mode='same') + weights_b[i]))

    plt.figure(figsize=(12, 12))
    for i in xrange(numFilters):
        plt.subplot(int(np.ceil(np.sqrt(numFilters))), int(np.ceil(np.sqrt(numFilters))), i+1)
        plt.imshow(res_scipy[i], cmap="gray", interpolation="nearest")
        plt.axis('off')
    plt.savefig("../results/filtered_by_scipy.png")
    plt.close()


def plot_some_data():
    memmap_name = "patchClassification_ws_resampled"
    with open("../data/%s_properties.pkl" % memmap_name, 'r') as f:
        memmap_properties = cPickle.load(f)
    n_pos_train = memmap_properties["train_pos"]
    n_neg_train = memmap_properties["train_neg"]
    n_pos_val = memmap_properties["val_pos"]
    n_neg_val = memmap_properties["val_neg"]
    train_pos_memmap = memmap("../data/%s_train_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["train_pos_shape"])
    train_neg_memmap = memmap("../data/%s_train_neg.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["train_neg_shape"])
    val_pos_memmap = memmap("../data/%s_val_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["val_pos_shape"])
    val_neg_memmap = memmap("../data/%s_val_neg.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["val_neg_shape"])
    i = 0
    ctr = 0
    for data, seg, labels in memmapGenerator(val_neg_memmap, val_pos_memmap, 128, n_pos_val, n_neg_val):
        if i == 2:
            break
        data2 = np.array(data)
        for img, segm, lab in zip(data2, seg, labels):
            img -= img.min()
            img /= img.max()
            plt.figure(figsize=(12,12))
            img = np.array(img[0]) # dont write into memmap
            img = np.repeat(img[np.newaxis, :, :], 3, 0)
            img = img.transpose((1, 2, 0))
            img[:, :, 0][segm[0] > 1] *= 1.0
            plt.imshow(img, interpolation='nearest')
            if lab == 0:
                color = 'green'
            else:
                color = 'red'
            plt.text(0, 0, lab, color=color, bbox=dict(facecolor='white', alpha=1))
            plt.savefig("../some_images/img_%04.0f.png"%ctr)
            plt.close()
            ctr += 1
        i += 1


def plot_some_data_t1km_flair():
    memmap_name = "patchClassification_ws_resampled_t1km_flair_new"
    with open("../data/%s_properties.pkl" % memmap_name, 'r') as f:
        memmap_properties = cPickle.load(f)
    n_pos_train = memmap_properties["train_pos"]
    n_neg_train = memmap_properties["train_neg"]
    n_pos_val = memmap_properties["val_pos"]
    n_neg_val = memmap_properties["val_neg"]
    train_pos_memmap = memmap("../data/%s_train_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["train_pos_shape"])
    train_neg_memmap = memmap("../data/%s_train_neg.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["train_neg_shape"])
    val_pos_memmap = memmap("../data/%s_val_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["val_pos_shape"])
    val_neg_memmap = memmap("../data/%s_val_neg.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["val_neg_shape"])
    i = 0
    ctr = 0
    for data, seg, labels in memmapGenerator_t1km_flair(val_neg_memmap, val_pos_memmap, 128, n_pos_val, n_neg_val):
        if i == 2:
            break
        data2 = np.array(data)
        for img, segm, lab in zip(data2, seg, labels):
            t1km_img = img[0]
            flair_img = img[1]
            plt.figure(figsize=(24,8))
            plt.subplot(1, 3, 1)
            plt.imshow(t1km_img, interpolation='nearest', cmap="gray")
            if lab == 0:
                color = 'green'
            else:
                color = 'red'
            plt.text(0, 0, lab, color=color, bbox=dict(facecolor='white', alpha=1))

            plt.subplot(1, 3, 2)
            plt.imshow(flair_img, interpolation='nearest', cmap="gray")
            plt.subplot(1, 3, 3)
            plt.imshow(segm[0], cmap="jet")
            plt.savefig("../some_images/img_%04.0f.png"%ctr)
            plt.close()
            ctr += 1
        i += 1


def plot_some_data_t1km_flair_adc_cbv():
    memmap_name = "patchClassification_ws_resampled_t1km_flair_adc_cbv_new"
    with open("../data/%s_properties.pkl" % memmap_name, 'r') as f:
        memmap_properties = cPickle.load(f)
    n_pos_train = memmap_properties["train_pos"]
    n_neg_train = memmap_properties["train_neg"]
    n_pos_val = memmap_properties["val_pos"]
    n_neg_val = memmap_properties["val_neg"]
    train_pos_memmap = memmap("../data/%s_train_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["train_pos_shape"])
    train_neg_memmap = memmap("../data/%s_train_neg.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["train_neg_shape"])
    val_pos_memmap = memmap("../data/%s_val_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["val_pos_shape"])
    val_neg_memmap = memmap("../data/%s_val_neg.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["val_neg_shape"])
    i = 0
    ctr = 0
    for data, seg, labels in memmapGenerator_t1km_flair_adc_cbv(val_neg_memmap, val_pos_memmap, 128, n_pos_val, n_neg_val):
        if i == 2:
            break
        data2 = np.array(data)
        for img, segm, lab in zip(data2, seg, labels):
            t1km_img = img[0]
            flair_img = img[1]
            adc_img = img[2]
            cbv_img = img[3]
            plt.figure(figsize=(24,8))
            plt.subplot(1, 5, 1)
            plt.imshow(t1km_img, interpolation='nearest', cmap="gray")
            if lab == 0:
                color = 'green'
            else:
                color = 'red'
            plt.text(0, 0, lab, color=color, bbox=dict(facecolor='white', alpha=1))

            plt.subplot(1, 5, 2)
            plt.imshow(flair_img, interpolation='nearest', cmap="gray")
            plt.subplot(1, 5, 3)
            plt.imshow(adc_img, interpolation='nearest', cmap="gray")
            plt.subplot(1, 5, 4)
            plt.imshow(cbv_img, interpolation='nearest', cmap="gray")
            plt.subplot(1, 5, 5)
            plt.imshow(segm[0], cmap="jet")
            plt.savefig("../some_images/img_%04.0f.png"%ctr)
            plt.close()
            ctr += 1
        i += 1


def plot_some_data_varNumChannels(memmap_gen = memmapGenerator_tumorClassRot):
    memmap_name = "patchClassification_ws_resampled_t1km_flair_adc_cbv_new"
    with open("../data/%s_properties.pkl" % memmap_name, 'r') as f:
        memmap_properties = cPickle.load(f)
    n_pos_train = memmap_properties["train_pos"]
    n_neg_train = memmap_properties["train_neg"]
    n_pos_val = memmap_properties["val_pos"]
    n_neg_val = memmap_properties["val_neg"]
    train_pos_memmap = memmap("../data/%s_train_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["train_pos_shape"])
    train_neg_memmap = memmap("../data/%s_train_neg.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["train_neg_shape"])
    val_pos_memmap = memmap("../data/%s_val_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["val_pos_shape"])
    val_neg_memmap = memmap("../data/%s_val_neg.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["val_neg_shape"])
    i = 0
    ctr = 0
    for data, seg, labels in memmap_gen(val_neg_memmap, val_pos_memmap, 128, n_pos_val, n_neg_val):
        if i == 2:
            break
        data2 = np.array(data)
        for img, segm, lab in zip(data2, seg, labels):
            num_subplots = img.shape[0] + segm.shape[0]
            subplot_ctr = 1
            plt.figure(figsize=(12, 12))
            for x in xrange(img.shape[0]):
                plt.subplot(int(np.ceil(num_subplots**0.5)), int(np.ceil(num_subplots**0.5)), subplot_ctr)
                plt.imshow(img[x], interpolation='nearest', cmap="gray")
                if x == 0:
                    if lab == 0:
                        color = 'green'
                    else:
                        color = 'red'
                    plt.text(0, 0, lab, color=color, bbox=dict(facecolor='white', alpha=1))
                subplot_ctr += 1
            for x in xrange(segm.shape[0]):
                plt.subplot(int(np.ceil(num_subplots**0.5)), int(np.ceil(num_subplots**0.5)), subplot_ctr)
                plt.imshow(segm[x], interpolation='nearest', cmap="gray")
                subplot_ctr += 1
            plt.savefig("../some_images/img_%04.0f.png"%ctr)
            plt.close()
            ctr += 1
        i += 1

def plot_some_data_varNumChannels_generator(generator, num_images=100):
    ctr = 0
    for data, seg, labels in generator:
        data2 = np.array(data)
        for img, segm, lab in zip(data2, seg, labels):
            num_subplots = img.shape[0] + segm.shape[0]
            subplot_ctr = 1
            plt.figure(figsize=(12, 12))
            for x in xrange(img.shape[0]):
                plt.subplot(int(np.ceil(num_subplots**0.5)), int(np.ceil(num_subplots**0.5)), subplot_ctr)
                plt.imshow(img[x], interpolation='nearest', cmap="gray")
                if x == 0:
                    if lab == 0:
                        color = 'green'
                    else:
                        color = 'red'
                    plt.text(0, 0, lab, color=color, bbox=dict(facecolor='white', alpha=1))
                subplot_ctr += 1
            for x in xrange(segm.shape[0]):
                plt.subplot(int(np.ceil(num_subplots**0.5)), int(np.ceil(num_subplots**0.5)), subplot_ctr)
                plt.imshow(segm[x], interpolation='nearest', cmap="gray")
                subplot_ctr += 1
            plt.savefig("/home/fabian/datasets/Hirntumor_von_David/experiments/some_images/img_%04.0f.png"%ctr)
            plt.close()
            ctr += 1
            if ctr >= num_images:
                break
        if ctr >= num_images:
            break


def plot_layer_weights(layer):
    conv_1_1_weights = layer.get_params()[0].get_value()

    plt.figure(figsize=(12, 12))
    for i in range(conv_1_1_weights.shape[0]):
        plt.subplot(int(np.ceil(np.sqrt(conv_1_1_weights.shape[0]))), int(np.ceil(np.sqrt(conv_1_1_weights.shape[0]))), i+1)
        plt.imshow(conv_1_1_weights[i, 0, :, :], cmap="gray", interpolation="nearest")
        plt.axis('off')
    plt.show()


def imgSaveFalsePositiveFalseNegativeCorrectPositiveCorrectNegative(pred_fn, n_images=16, BATCH_SIZE = 50):
    with open("../data/patchClassification_memmap_properties.pkl", 'r') as f:
        memmap_properties = cPickle.load(f)
    n_pos_val = memmap_properties["val_pos"]
    n_neg_val = memmap_properties["val_neg"]
    val_pos_memmap = memmap("../data/patchClassification_val_pos.memmap", dtype=np.float32, mode="r+", shape=memmap_properties["val_pos_shape"])
    val_neg_memmap = memmap("../data/patchClassification_val_neg.memmap", dtype=np.float32, mode="r+", shape=memmap_properties["val_neg_shape"])
    n_fpos = 0
    n_fneg = 0
    n_tpos = 0
    n_tneg = 0
    # it is simpler to just extract the fpos, fneg, tpos and tneg images one cathegory after the other. speed doesnt matter here
    plt.figure(figsize=(16,16))
    for data, seg, labels in memmapGenerator(val_neg_memmap, val_pos_memmap, BATCH_SIZE, n_pos_val, n_neg_val):
        if n_fpos < n_images:
            pred = pred_fn(data).argmax(-1)
            idx = np.where((pred == 1) & (labels == 0))[0]
            for id in idx:
                plt.subplot(int(np.ceil(np.sqrt(n_images))), int(np.ceil(np.sqrt(n_images))), n_fpos)
                plt.imshow(data[id, 0, :, :], cmap="gray", interpolation="nearest")
                plt.text(0, 0, labels[id], color='red', bbox=dict(facecolor='white', alpha=1))
                plt.text(0, 12, pred[id], color='green', bbox=dict(facecolor='white', alpha=1))
                plt.text(8, 0, 'true', color='blue')
                plt.text(8, 12, 'predicted', color='blue')
                n_fpos += 1
                if n_fpos >= n_images:
                    break
        else:
            break
    plt.savefig("../results/falsePositives.png")
    plt.close()

    plt.figure(figsize=(16,16))
    for data, seg, labels in memmapGenerator(val_neg_memmap, val_pos_memmap, BATCH_SIZE, n_pos_val, n_neg_val):
        if n_fneg < n_images:
            pred = pred_fn(data).argmax(-1)
            idx = np.where((pred == 0) & (labels == 1))[0]
            for id in idx:
                plt.subplot(int(np.ceil(np.sqrt(n_images))), int(np.ceil(np.sqrt(n_images))), n_fneg)
                plt.imshow(data[id, 0, :, :], cmap="gray", interpolation="nearest")
                plt.text(0, 0, labels[id], color='green', bbox=dict(facecolor='white', alpha=1))
                plt.text(0, 12, pred[id], color='red', bbox=dict(facecolor='white', alpha=1))
                plt.text(8, 0, 'true', color='blue')
                plt.text(8, 12, 'predicted', color='blue')
                n_fneg += 1
                if n_fneg >= n_images:
                    break
        else:
            break
    plt.savefig("../results/falseNegatives.png")
    plt.close()

    plt.figure(figsize=(16,16))
    for data, seg, labels in memmapGenerator(val_neg_memmap, val_pos_memmap, BATCH_SIZE, n_pos_val, n_neg_val):
        if n_tpos < n_images:
            pred = pred_fn(data).argmax(-1)
            idx = np.where((pred == 1) & (labels == 1))[0]
            for id in idx:
                plt.subplot(int(np.ceil(np.sqrt(n_images))), int(np.ceil(np.sqrt(n_images))), n_tpos)
                plt.imshow(data[id, 0, :, :], cmap="gray", interpolation="nearest")
                plt.text(0, 0, labels[id], color='green', bbox=dict(facecolor='white', alpha=1))
                plt.text(0, 12, pred[id], color='green', bbox=dict(facecolor='white', alpha=1))
                plt.text(8, 0, 'true', color='blue')
                plt.text(8, 12, 'predicted', color='blue')
                n_tpos += 1
                if n_tpos >= n_images:
                    break
        else:
            break
    plt.savefig("../results/truePositives.png")
    plt.close()

    plt.figure(figsize=(16,16))
    for data, seg, labels in memmapGenerator(val_neg_memmap, val_pos_memmap, BATCH_SIZE, n_pos_val, n_neg_val):
        if n_tneg < n_images:
            pred = pred_fn(data).argmax(-1)
            idx = np.where((pred == 0) & (labels == 0))[0]
            for id in idx:
                plt.subplot(int(np.ceil(np.sqrt(n_images))), int(np.ceil(np.sqrt(n_images))), n_tneg)
                plt.imshow(data[id, 0, :, :], cmap="gray", interpolation="nearest")
                plt.text(0, 0, labels[id], color='red', bbox=dict(facecolor='white', alpha=1))
                plt.text(0, 12, pred[id], color='red', bbox=dict(facecolor='white', alpha=1))
                plt.text(8, 0, 'true', color='blue')
                plt.text(8, 12, 'predicted', color='blue')
                n_tneg += 1
                if n_tneg >= n_images:
                    break
        else:
            break
    plt.savefig("../results/trueNegatives.png")
    plt.close()


def show_segmentation_results(data, seg_true, seg_pred, img_ctr=0):
    n_channels = data.shape[1]
    n_images_in_figure = float(n_channels + 3)
    n_cols_and_rows = int(np.ceil(n_images_in_figure**0.5))
    seg_diff = np.zeros(seg_pred.shape)
    seg_diff[seg_true[:, 0,:,:]!=seg_pred] = 1
    for x in range(0, data.shape[0]):
        plt.figure(figsize=(10,10))
        for i in range(1, n_channels+1):
            plt.subplot(n_cols_and_rows, n_cols_and_rows, i)
            plt.imshow(data[x, i-1, :, :], cmap="gray", interpolation="none")
        plt.subplot(n_cols_and_rows, n_cols_and_rows, n_channels+1)
        plt.imshow(seg_true[x, 0, :, :], cmap="jet", interpolation="none")
        plt.subplot(n_cols_and_rows, n_cols_and_rows, n_channels+2)
        plt.imshow(seg_pred[x, :, :], cmap="jet", interpolation="none")
        plt.subplot(n_cols_and_rows, n_cols_and_rows, n_channels+3)
        plt.imshow(seg_diff[x, :, :], cmap="gray", interpolation="none")
        plt.savefig("/home/fabian/datasets/Hirntumor_von_David/experiments/some_images/seg_res_%04.0d.png"%(img_ctr+x))
        plt.close()
    return img_ctr+data.shape[0]


def plot_histograms_for_all_images():
    import os
    import SimpleITK as sitk
    import matplotlib.pyplot as plt
    folder = "/home/fabian/datasets/Hirntumor_von_David/"
    for i in range(150):
        if os.path.isdir(os.path.join(folder, "%03.0d"%i)):
            if os.path.isfile(os.path.join(folder, "%03.0f"%i, "T1_m2_bc_ws.nii.gz")):
                img = correct_nans(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(folder, "%03.0d"%i, "T1_m2_bc_ws.nii.gz"))))
                plt.figure()
                plt.hist(img[img!=img[0,0,0]].ravel(), 100)
                plt.savefig(folder+"%03.0f_t1_ws_histogram.png"%i)
                plt.close()

    import os
    import SimpleITK as sitk
    import matplotlib.pyplot as plt
    folder = "/home/fabian/datasets/Hirntumor_von_David/"
    for i in range(150):
        if os.path.isdir(os.path.join(folder, "%03.0d"%i)):
            if os.path.isfile(os.path.join(folder, "%03.0f"%i, "T1KM_m2_bc_ws.nii.gz")):
                img = correct_nans(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(folder, "%03.0d"%i, "T1KM_m2_bc_ws.nii.gz"))))
                plt.figure()
                plt.hist(img[img!=img[0,0,0]].ravel(), 100)
                plt.savefig(folder+"%03.0f_t1km_ws_histogram.png"%i)
                plt.close()


    import os
    import SimpleITK as sitk
    import matplotlib.pyplot as plt
    folder = "/home/fabian/datasets/Hirntumor_von_David/"
    for i in range(150):
        if os.path.isdir(os.path.join(folder, "%03.0d"%i)):
            if os.path.isfile(os.path.join(folder, "%03.0f"%i, "ADC_mutualinfo2_reg.nii.gz")):
                img = correct_nans(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(folder, "%03.0d"%i, "ADC_mutualinfo2_reg.nii.gz"))))
                plt.figure()
                plt.hist(img[img!=img[0,0,0]].ravel(), 100)
                plt.savefig(folder+"%03.0f_ADC_ws_histogram.png"%i)
                plt.close()

    import os
    import SimpleITK as sitk
    import matplotlib.pyplot as plt
    folder = "/home/fabian/datasets/Hirntumor_von_David/"
    for i in range(150):
        if os.path.isdir(os.path.join(folder, "%03.0d"%i)):
            if os.path.isfile(os.path.join(folder, "%03.0f"%i, "CBV_mutualinfo2_reg.nii.gz")):
                img = correct_nans(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(folder, "%03.0d"%i, "CBV_mutualinfo2_reg.nii.gz"))))
                plt.figure()
                plt.hist(img[img!=img[0,0,0]].ravel(), 100)
                plt.savefig(folder+"%03.0f_CBV_ws_histogram.png"%i)
                plt.close()

    import os
    import SimpleITK as sitk
    import matplotlib.pyplot as plt
    folder = "/home/fabian/datasets/Hirntumor_von_David/"
    for i in range(150):
        if os.path.isdir(os.path.join(folder, "%03.0d"%i)):
            if os.path.isfile(os.path.join(folder, "%03.0f"%i, "FLAIR_m2_bc_ws.nii.gz")):
                img = correct_nans(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(folder, "%03.0d"%i, "FLAIR_m2_bc_ws.nii.gz"))))
                plt.figure()
                plt.hist(img[img!=img[0,0,0]].ravel(), 100)
                plt.savefig(folder+"%03.0f_flair_ws_histogram.png"%i)
                plt.close()

