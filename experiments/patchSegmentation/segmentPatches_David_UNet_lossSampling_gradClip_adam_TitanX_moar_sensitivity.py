__author__ = 'fabian'
import theano
import lasagne
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import sys
import os
sys.path.append("../../neural_networks/")
sys.path.append("../../generators/")
sys.path.append("../../utils/")
sys.path.append("../../dataset_utils/")
import cPickle
from copy import deepcopy
from numpy import memmap

from UNet import build_UNet
from data_generators import SegmentationBatchGeneratorDavid, load_all_patients_David
from data_augmentation_generators import seg_channel_selection_generator, center_crop_generator, rotation_generator, elastric_transform_generator, center_crop_seg_generator, pad_generator, data_channel_selection_generator
from multithreaded_generators import multi_threaded_generator
from utils_plotting import show_segmentation_results, printLosses, plot_all_layer_activations
from generator_utils import elastic_transform_2d
from multithreaded_generators import Multithreaded_Generator
from sklearn.metrics import roc_auc_score
from general_utils import convert_seg_flat_to_binary_label_indicator_array

sys.setrecursionlimit(2000)

BATCH_SIZE = 20
INPUT_PATCH_SIZE =(370 + 16-370%16 + 180 + 16, 309 + 16-309%16 + 180 + 16)
OUTPUT_PATCH_SIZE = (388, 324)
#validation_patients = np.random.choice(150, 15, False)
validation_patients = [125,  68,  85,  88,   7, 112, 130,   8,  32, 122,  70, 100, 128, 91,  41]
num_classes=5

all_patients = load_all_patients_David()
'''tmp = SegmentationBatchGeneratorDavid(all_patients, 50, validation_patients, PATCH_SIZE=OUTPUT_PATCH_SIZE, mode="train", ignore=[40], losses=None, num_batches=None, seed=None)
tmp = Multithreaded_Generator(tmp, 2, 20)

ctr = 0
class_frequencies = np.zeros(5)
for data, seg, id in tmp:
    print ctr
    class_frequencies[0] += np.sum(seg[:, 2] == 0)
    class_frequencies[1] += np.sum(seg[:, 2] == 1)
    class_frequencies[2] += np.sum(seg[:, 2] == 2)
    class_frequencies[3] += np.sum(seg[:, 2] == 3)
    class_frequencies[4] += np.sum(seg[:, 2] == 4)
    ctr += 1
    if ctr >= 10000:
        break'''

class_frequencies = [  5.23746953e+08,   2.70040499e+08,   1.20614000e+07,
         4.46691700e+06,   1.88281200e+06]


dataset_folder = "/media/fabian/DeepLearningData/datasets/"
EXPERIMENT_NAME = "segmentPatches_David_UNet_lossSampling_gradClip_adam_TitanX_moar_sensitivity"
results_dir = os.path.join("/home/fabian/datasets/Hirntumor_von_David/experiments/results/", EXPERIMENT_NAME)
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

class_frequencies = np.array(class_frequencies).astype(np.float32)**0.8
class_frequencies2 = deepcopy(class_frequencies)
for i in range(len(class_frequencies)):
    class_frequencies2[i] = class_frequencies[range(len(class_frequencies)) != i] / class_frequencies[i]
class_frequencies2 /= np.sum(class_frequencies2)
class_frequencies2 *= len(class_frequencies)


'''d, s, l = data_gen_train.next()
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(d[0,0], cmap="gray")
plt.subplot(1, 3, 2)
d1=elastic_transform_2d(d[0,0], 550., 20.)
plt.imshow(d1, cmap="gray")
plt.subplot(1, 3, 3)
plt.imshow(d[0,0]-d1)
plt.show()
plt.close()'''

data_gen_validation = SegmentationBatchGeneratorDavid(all_patients, BATCH_SIZE, validation_patients, PATCH_SIZE=INPUT_PATCH_SIZE, mode="test", ignore=[81], losses=None, num_batches=None, seed=None)
data_gen_validation = seg_channel_selection_generator(data_gen_validation, [2])
data_gen_validation = center_crop_seg_generator(data_gen_validation, OUTPUT_PATCH_SIZE)
data_gen_validation = Multithreaded_Generator(data_gen_validation, 2, 10)
data_gen_validation._start()

net = build_UNet(25, BATCH_SIZE, num_output_classes=num_classes, base_n_filters=16, input_dim=INPUT_PATCH_SIZE, pad="valid")
output_layer_for_loss = net["output_flattened"]

n_batches_per_epoch = 300
# n_batches_per_epoch = np.floor(n_training_samples/float(BATCH_SIZE))
n_test_batches = 30
# n_test_batches = np.floor(n_val_samples/float(BATCH_SIZE))

x_sym = T.tensor4()
seg_sym = T.ivector()
w_sym = T.vector()

# add some weight decay
l2_loss = lasagne.regularization.regularize_network_params(output_layer_for_loss, lasagne.regularization.l2) * 1e-4

# the distinction between prediction_train and test is important only if we enable dropout
prediction_train = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=False)
# we could use a binary loss but I stuck with categorical crossentropy so that less code has to be changed if your
# application has more than two classes
loss_vec = lasagne.objectives.categorical_crossentropy(prediction_train, seg_sym)
loss_vec *= w_sym
loss = loss_vec.mean()
loss += l2_loss
acc_train = T.mean(T.eq(T.argmax(prediction_train, axis=1), seg_sym), dtype=theano.config.floatX)

prediction_test = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=True)
loss_val = lasagne.objectives.categorical_crossentropy(prediction_test, seg_sym)

# we multiply our loss by a weight map. In this example the weight map only increases the loss for road pixels and
# decreases the loss for other pixels. We do this to ensure that the network puts more focus on getting the roads
# right
loss_val *= w_sym
loss_val = loss_val.mean()
loss_val += l2_loss
acc = T.mean(T.eq(T.argmax(prediction_test, axis=1), seg_sym), dtype=theano.config.floatX)

# learning rate has to be a shared variablebecause we decrease it with every epoch
params = lasagne.layers.get_all_params(output_layer_for_loss, trainable=True)
grad = [theano.gradient.grad_clip(i, -10., 10.) for i in T.grad(loss, params)]
learning_rate = theano.shared(np.float32(0.001))
updates = lasagne.updates.adam(grad, params, learning_rate=learning_rate)
# updates = lasagne.updates.nesterov_momentum(grad, params, learning_rate, 0.9)

# create a convenience function to get the segmentation
seg_output = lasagne.layers.get_output(net["output_segmentation"], x_sym, deterministic=True)
seg_output = seg_output.argmax(1)

train_fn = theano.function([x_sym, seg_sym, w_sym], [loss_vec, acc_train], updates=updates)
val_fn = theano.function([x_sym, seg_sym, w_sym], [loss_val, acc])
get_segmentation = theano.function([x_sym], seg_output)
# we need this for calculating the AUC score
get_class_probas = theano.function([x_sym], prediction_test)

n_feedbacks_per_epoch = 10.


def update_losses(losses, idx, loss):
    losses[idx] = (losses[idx] + loss*2.) / 3.
    return losses

n_epochs = 60
auc_scores=None

start_from_epoch = 0
if not start_from_epoch == 0:
    learning_rate = learning_rate * (0.7 ** float(start_from_epoch))
    epoch = start_from_epoch + 1
    with open(os.path.join(results_dir, "%s_Params_ep%d.pkl" % (EXPERIMENT_NAME, start_from_epoch)), 'r') as f:
        params = cPickle.load(f)
        lasagne.layers.set_all_param_values(output_layer_for_loss, params)
    with open(os.path.join(results_dir, "%s_allLossesNAccur_ep%d.pkl" % (EXPERIMENT_NAME, start_from_epoch)), 'r') as f:
        [all_training_losses, all_training_accuracies, all_validation_losses, all_validation_accuracies, auc_all, losses] = cPickle.load(f)

else:
    all_training_losses = []
    all_validation_losses = []
    all_validation_accuracies = []
    all_training_accuracies = []
    auc_all = []

    tmp = SegmentationBatchGeneratorDavid(all_patients, BATCH_SIZE, validation_patients, PATCH_SIZE=OUTPUT_PATCH_SIZE, mode="train", ignore=[81], losses=None, num_batches=None, seed=None)
    losses = np.ones(tmp.get_losses().shape[0])
    del tmp
    epoch = 0

def compare_seg_with_gt(max_n_images=10, epoch=0):
    data_gen_validation = SegmentationBatchGeneratorDavid(all_patients, BATCH_SIZE, validation_patients, PATCH_SIZE=OUTPUT_PATCH_SIZE, mode="test", ignore=[81], losses=None, num_batches=None, seed=10)
    data_gen_validation = seg_channel_selection_generator(data_gen_validation, [2])
    data_gen_validation = center_crop_seg_generator(data_gen_validation, OUTPUT_PATCH_SIZE)
    data, seg, idx = data_gen_validation.next()
    seg = np.array(seg)
    seg_pred = get_segmentation(data)
    plt.figure(figsize=(6, 20))
    n_images = np.min((seg_pred.shape[0], max_n_images))
    for i in range(n_images):
        seg_pred[i][0, :6] = np.array([0,1,2,3,4,5])
        seg[i,0,0,:6] = np.array([0,1,2,3,4,5])
        plt.subplot(n_images, 2, 2*i+1)
        plt.imshow(seg[i, 0])
        plt.subplot(n_images, 2, 2*i+2)
        plt.imshow(seg_pred[i])
    plt.savefig(os.path.join(results_dir, "some_segmentations_ep_%d.png"%epoch))


while epoch < n_epochs:
    data_gen_train = SegmentationBatchGeneratorDavid(all_patients, BATCH_SIZE, validation_patients, PATCH_SIZE=OUTPUT_PATCH_SIZE, mode="train", ignore=[81], losses=losses, num_batches=None, seed=None)
    data_gen_train = seg_channel_selection_generator(data_gen_train, [2])
    data_gen_train = rotation_generator(data_gen_train)
    data_gen_train = elastric_transform_generator(data_gen_train, 200., 14.)
    data_gen_train = pad_generator(data_gen_train, INPUT_PATCH_SIZE)
    data_gen_train = center_crop_seg_generator(data_gen_train, OUTPUT_PATCH_SIZE)
    data_gen_train = Multithreaded_Generator(data_gen_train, 8, 16)
    data_gen_train._start()
    print "epoch: ", epoch
    train_loss = 0
    train_acc_tmp = 0
    train_loss_tmp = 0
    batch_ctr = 0
    for data, seg, idx in data_gen_train:
        if batch_ctr != 0 and batch_ctr % int(np.floor(n_batches_per_epoch/n_feedbacks_per_epoch)) == 0:
            print "number of batches: ", batch_ctr, "/", n_batches_per_epoch
            print "training_loss since last update: ", train_loss_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch), " train accuracy: ", train_acc_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch)
            all_training_losses.append(train_loss_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch))
            all_training_accuracies.append(train_acc_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch))
            train_loss_tmp = 0
            train_acc_tmp = 0
            if len(auc_all) > 0:
                auc_scores = np.concatenate(auc_all, axis=0).reshape(-1, len(class_frequencies2)-1)
            printLosses(all_training_losses, all_training_accuracies, all_validation_losses, all_validation_accuracies, os.path.join(results_dir, "%s.png" % EXPERIMENT_NAME), n_feedbacks_per_epoch, auc_scores=auc_scores, auc_labels=["brain", "1", "2", "3", "4"], ylim_score=(0,0.08))
        # loss, acc = train_fn(data, convert_seg_map_for_crossentropy(seg, range(4)).astype(np.float32))
        seg_flat = seg.flatten().astype(np.int32)
        w = class_frequencies2[seg_flat]
        loss_vec, acc = train_fn(data, seg_flat, w) #class_weights[seg_flat]
        loss = loss_vec.mean()
        loss_per_sample = loss_vec.reshape(BATCH_SIZE, -1).mean(axis=1)
        losses = update_losses(losses, idx, loss_per_sample)
        train_loss += loss
        train_loss_tmp += loss
        train_acc_tmp += acc
        batch_ctr += 1
        if batch_ctr > n_batches_per_epoch:
            break

    data_gen_train._finish()

    train_loss /= n_batches_per_epoch
    print "training loss average on epoch: ", train_loss
    if epoch <= 1:
        losses[:] = 100.

    y_true = []
    y_pred = []
    test_loss = 0
    accuracies = []
    valid_batch_ctr = 0
    for data, seg, labels in data_gen_validation:
        # loss, acc = val_fn(data, convert_seg_map_for_crossentropy(seg, range(4)).astype(np.float32))
        seg_flat = seg.flatten().astype(np.int32)
        w = class_frequencies2[seg_flat]
        loss, acc = val_fn(data, seg_flat, w) #, class_weights[seg_flat]
        test_loss += loss
        accuracies.append(acc)
        valid_batch_ctr += 1
        y_true.append(convert_seg_flat_to_binary_label_indicator_array(seg_flat[seg_flat!=0]-1, len(class_frequencies2)-1))
        y_pred.append(get_class_probas(data)[seg_flat!=0, :][:, 1:])
        if valid_batch_ctr > n_test_batches:
            break
    test_loss /= n_test_batches
    print "test loss: ", test_loss
    print "test acc: ", np.mean(accuracies), "\n"
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    scores = roc_auc_score(y_true, y_pred, None)
    del y_pred, y_true
    auc_all.append(scores)
    all_validation_losses.append(test_loss)
    all_validation_accuracies.append(np.mean(accuracies))
    auc_scores = np.concatenate(auc_all, axis=0).reshape(-1, len(class_frequencies2)-1)
    printLosses(all_training_losses, all_training_accuracies, all_validation_losses, all_validation_accuracies, os.path.join(results_dir, "%s.png" % EXPERIMENT_NAME), n_feedbacks_per_epoch, auc_scores=auc_scores, auc_labels=["brain", "1", "2", "3", "4"], ylim_score=(0,0.08))
    with open(os.path.join(results_dir, "%s_Params_ep%d.pkl" % (EXPERIMENT_NAME, epoch)), 'w') as f:
        cPickle.dump(lasagne.layers.get_all_param_values(output_layer_for_loss), f)
    with open(os.path.join(results_dir, "%s_allLossesNAccur_ep%d.pkl"% (EXPERIMENT_NAME, epoch)), 'w') as f:
        cPickle.dump([all_training_losses, all_training_accuracies, all_validation_losses, all_validation_accuracies, auc_all, losses], f)
    with open(os.path.join(results_dir, "%s_lossPerPatch_ep%d.pkl"% (EXPERIMENT_NAME, epoch)), 'w') as f:
        cPickle.dump(losses, f)
    if (epoch > 1) and (all_validation_losses[-1] / all_validation_losses[-2] > 1.5):
        print "ouch... lets revert to the last epoch"
        with open(os.path.join(results_dir, "%s_Params_ep%d.pkl" % (EXPERIMENT_NAME, epoch -1)), 'r') as f:
            params = cPickle.load(f)
            lasagne.layers.set_all_param_values(output_layer_for_loss, params)
        with open(os.path.join(results_dir, "%s_allLossesNAccur_ep%d.pkl" % (EXPERIMENT_NAME, epoch - 1)), 'r') as f:
            [all_training_losses, all_training_accuracies, all_validation_losses, all_validation_accuracies, auc_all, losses] = cPickle.load(f)
    else:
        compare_seg_with_gt(5, epoch)
        epoch += 1
        learning_rate *= 0.7

data_gen_validation._finish()

'''

img_ctr = 0
batch_ctr2 = 0
for data, seg, labels in data_gen_validation:
    pred = get_segmentation(data)
    img_ctr = show_segmentation_results(data, seg, pred, img_ctr=img_ctr)
    batch_ctr2 += 1
    if batch_ctr2 >= 3:
        break'''
