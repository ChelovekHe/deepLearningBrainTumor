__author__ = 'fabian'
import theano
import lasagne
import theano.tensor as T
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import cPickle as pickle
import sys
sys.path.append("../../neural_networks/")
sys.path.append("../../generators/")
sys.path.append("../../utils/")
sys.path.append("../../dataset_utils/")
import cPickle
from copy import deepcopy
from numpy import memmap

from UNet import build_UNet
from data_generators import memmapGenerator_allInOne_segmentation_lossSampling
from data_augmentation_generators import seg_channel_selection_generator, center_crop_generator, rotation_generator, elastric_transform_generator
from multithreaded_generators import multi_threaded_generator
from utils_plotting import show_segmentation_results, printLosses, plot_all_layer_activations
from generator_utils import elastic_transform_2d
from multithreaded_generators import Multithreaded_Generator

sys.setrecursionlimit(2000)

dataset_folder = "/media/fabian/DeepLearningData/datasets/"
EXPERIMENT_NAME = "segment_tumor_v0.1_Unet"
memmap_name = "patchSegmentation_allInOne_ws_t1km_flair_adc_cbv_resized"

BATCH_SIZE = 10
PATCH_SIZE = 256

with open(dataset_folder + "%s_properties.pkl" % (memmap_name), 'r') as f:
    my_dict = cPickle.load(f)

data_ctr = my_dict['n_data']
train_shape = my_dict['train_neg_shape']
info_memmap_shape = my_dict['info_shape']

class_frequencies = np.zeros(5, dtype=np.float32)
for i in range(5):
    class_frequencies[i] = my_dict['class_frequencies'][i]
class_frequencies = np.sqrt(class_frequencies)**0.5
class_frequencies2 = deepcopy(class_frequencies)
for i in range(5):
    class_frequencies2[i] = class_frequencies[range(5) != i] / class_frequencies[i]
class_frequencies2 /= np.sum(class_frequencies2)
class_frequencies2 *= 5

memmap_data = memmap(dataset_folder + "%s.memmap" % (memmap_name), dtype=np.float32, mode="r", shape=train_shape)
memmap_gt = memmap(dataset_folder + "%s_info.memmap" % (memmap_name), dtype=np.float32, mode="r", shape=info_memmap_shape)

patient_ids = np.unique(memmap_gt[:, 0]).astype(int)
# validation_patients = np.random.choice(patient_ids, 15)

validation_patients = [ 75,   1,  67,   1, 127, 120,  94, 131,  78,  74,  62,  10,  65, 47, 124]

n_training_samples = int(float(len(patient_ids) - len(validation_patients)) / float(len(patient_ids)) * memmap_data.shape[0])
n_val_samples = int(float(len(validation_patients)) / float(len(patient_ids)) * memmap_data.shape[0])

data_gen_train = memmapGenerator_allInOne_segmentation(memmap_data, memmap_gt, BATCH_SIZE, validation_patients, mode="train", shuffle=False, ignore=[40])
data_gen_train = seg_channel_selection_generator(data_gen_train, [2])
data_gen_train = rotation_generator(data_gen_train)
data_gen_train = center_crop_generator(data_gen_train, (PATCH_SIZE, PATCH_SIZE))
data_gen_train = elastric_transform_generator(data_gen_train, 550., 20.)
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
data_gen_train = multi_threaded_generator(data_gen_train, num_threads=8, num_cached=100)
_ = data_gen_train.next()

data_gen_validation = memmapGenerator_allInOne_segmentation(memmap_data, memmap_gt, BATCH_SIZE, validation_patients, mode="test", shuffle=False, ignore=[40])
data_gen_validation = center_crop_generator(data_gen_validation, (PATCH_SIZE, PATCH_SIZE))
data_gen_validation = seg_channel_selection_generator(data_gen_validation, [2])
data_gen_validation = multi_threaded_generator(data_gen_validation, num_threads=4, num_cached=10)

net = build_UNet(20, BATCH_SIZE, num_output_classes=5, base_n_filters=16, input_dim=(PATCH_SIZE, PATCH_SIZE))
output_layer_for_loss = net["output_flattened"]

with open("../../../results/segment_tumor_v0.1_Unet_Params_ep1.pkl", 'r') as f:
    params = cPickle.load(f)
    lasagne.layers.set_all_param_values(output_layer_for_loss, params)
with open("../../../results/segment_tumor_v0.1_Unet_allLossesNAccur_ep1.pkl", 'r') as f:
    [all_training_losses, all_training_accuracies, all_validation_losses, all_validation_accuracies] = cPickle.load(f)

n_batches_per_epoch = 2000
# n_batches_per_epoch = np.floor(n_training_samples/float(BATCH_SIZE))
n_test_batches = 200
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
loss = lasagne.objectives.categorical_crossentropy(prediction_train, seg_sym)
loss *= w_sym
loss = loss.mean()
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
learning_rate = theano.shared(np.float32(0.00001))
updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

# create a convenience function to get the segmentation
seg_output = lasagne.layers.get_output(net["output_segmentation"], x_sym, deterministic=True)
seg_output = seg_output.argmax(1)

train_fn = theano.function([x_sym, seg_sym, w_sym], [loss, acc_train], updates=updates)
val_fn = theano.function([x_sym, seg_sym, w_sym], [loss_val, acc])
get_segmentation = theano.function([x_sym], seg_output)
# we need this for calculating the AUC score
get_class_probas = theano.function([x_sym], prediction_test)

n_feedbacks_per_epoch = 30.

all_training_losses = []
all_validation_losses = []
all_validation_accuracies = []
all_training_accuracies = []


n_epochs = 10
for epoch in range(2, n_epochs):
    print "epoch: ", epoch
    train_loss = 0
    train_acc_tmp = 0
    train_loss_tmp = 0
    batch_ctr = 0
    for data, seg, labels in data_gen_train:
        if batch_ctr != 0 and batch_ctr % int(np.floor(n_batches_per_epoch/n_feedbacks_per_epoch)) == 0:
            print "number of batches: ", batch_ctr, "/", n_batches_per_epoch
            print "training_loss since last update: ", train_loss_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch), " train accuracy: ", train_acc_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch)
            all_training_losses.append(train_loss_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch))
            all_training_accuracies.append(train_acc_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch))
            train_loss_tmp = 0
            train_acc_tmp = 0
            printLosses(all_training_losses, all_training_accuracies, all_validation_losses, all_validation_accuracies, "../../../results/%s.png" % EXPERIMENT_NAME, n_feedbacks_per_epoch)
        # loss, acc = train_fn(data, convert_seg_map_for_crossentropy(seg, range(4)).astype(np.float32))
        seg_flat = seg.flatten().astype(np.int32)
        w = class_frequencies2[seg_flat]
        loss, acc = train_fn(data, seg_flat, w) #class_weights[seg_flat]
        train_loss += loss
        train_loss_tmp += loss
        train_acc_tmp += acc
        batch_ctr += 1
        if batch_ctr > n_batches_per_epoch:
            break

    train_loss /= n_batches_per_epoch
    print "training loss average on epoch: ", train_loss

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
        if valid_batch_ctr > n_test_batches:
            break
    test_loss /= n_test_batches
    print "test loss: ", test_loss
    print "test acc: ", np.mean(accuracies), "\n"
    all_validation_losses.append(test_loss)
    all_validation_accuracies.append(np.mean(accuracies))
    printLosses(all_training_losses, all_training_accuracies, all_validation_losses, all_validation_accuracies, "../../../results/%s.png" % EXPERIMENT_NAME, n_feedbacks_per_epoch)
    learning_rate *= 0.1
    with open("../../../results/%s_Params_ep%d.pkl" % (EXPERIMENT_NAME, epoch), 'w') as f:
        cPickle.dump(lasagne.layers.get_all_param_values(output_layer_for_loss), f)
    with open("../../../results/%s_allLossesNAccur_ep%d.pkl"% (EXPERIMENT_NAME, epoch), 'w') as f:
        cPickle.dump([all_training_losses, all_training_accuracies, all_validation_losses, all_validation_accuracies], f)

import cPickle
with open("../../../results/%s_Params.pkl"%EXPERIMENT_NAME, 'w') as f:
    cPickle.dump(lasagne.layers.get_all_param_values(output_layer_for_loss), f)
with open("../../../results/%s_allLossesNAccur.pkl"%EXPERIMENT_NAME, 'w') as f:
    cPickle.dump([all_training_losses, all_training_accuracies, all_validation_losses, all_validation_accuracies], f)

img_ctr = 0
batch_ctr2 = 0
for data, seg, labels in data_gen_validation:
    pred = get_segmentation(data)
    img_ctr = show_segmentation_results(data, seg, pred, img_ctr=img_ctr)
    batch_ctr2 += 1
    if batch_ctr2 >= 3:
        break