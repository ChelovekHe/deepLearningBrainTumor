__author__ = 'fabian'
import theano
import lasagne
import theano.tensor as T
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
from lasagne.layers import FlattenLayer
import cPickle as pickle
import sys
from utils import *
from data_generators import memmapGenerator, memmapGeneratorDataAugm, memmapGenerator_t1km_flair, memmapGeneratorDataAugm_t1km_flair, memmapGeneratorDataAugm_t1km_flair_adc_cbv, memmapGenerator_t1km_flair_adc_cbv, memmapGenerator_tumorClassRot
import cPickle
from lasagne.layers import batch_norm
from neural_networks import *
from experimentsWithMPQueue import multi_threaded_generator

sys.setrecursionlimit(2000)

EXPERIMENT_NAME = "segment_tumor_v0.1_residualUnet"
memmap_name = "patchClassification_ws_resampled_t1km_flair_adc_cbv_new"
BATCH_SIZE = 40

with open("/media/fabian/DeepLearningData/datasets/%s_properties.pkl" % memmap_name, 'r') as f:
    memmap_properties = cPickle.load(f)
n_pos_train = memmap_properties["train_pos"]
n_neg_train = memmap_properties["train_neg"]
n_pos_val = memmap_properties["val_pos"]
n_neg_val = memmap_properties["val_neg"]
n_training_samples = memmap_properties["train_total"]
n_val_samples = memmap_properties["val_total"]


net = build_residual_UNet(4, BATCH_SIZE, num_output_classes=4, base_n_filters=16)
output_layer = net["output"]

'''params_from = EXPERIMENT_NAME
with open("../results/%s_Params_ep8.pkl"%params_from, 'r') as f:
    params = cPickle.load(f)
    lasagne.layers.set_all_param_values(output_layer, params)'''


n_batches_per_epoch = np.floor(n_training_samples/float(BATCH_SIZE))
n_test_batches = np.floor(n_val_samples/float(BATCH_SIZE))

x_sym = T.tensor4()
seg_sym = T.ivector()
x_sym.tag.test_value = np.random.random((30, 4, 128, 128)).astype(np.float32)
seg_sym.tag.test_value = np.zeros((30*128*128,)).astype(np.int32)
# w_sym = T.vector()

l2_loss = lasagne.regularization.regularize_network_params(output_layer, lasagne.regularization.l2) * 1e-4

prediction_train = lasagne.layers.get_output(output_layer, x_sym, deterministic=False)
loss = lasagne.objectives.categorical_crossentropy(prediction_train, seg_sym)
# loss = categorical_crossentropy_fcn(prediction_train, seg_sym)
# loss *= w_sym
loss = loss.mean()
loss += l2_loss
acc_train = T.mean(T.eq(T.argmax(prediction_train, axis=1), seg_sym), dtype=theano.config.floatX)

prediction_test = lasagne.layers.get_output(output_layer, x_sym, deterministic=True)
loss_val = lasagne.objectives.categorical_crossentropy(prediction_test, seg_sym)
# loss_val = categorical_crossentropy_fcn(prediction_test, seg_sym.argmax(1))
# loss_val *= w_sym
loss_val = loss_val.mean()
loss_val += l2_loss
acc = T.mean(T.eq(T.argmax(prediction_test, axis=1), seg_sym), dtype=theano.config.floatX)

params = lasagne.layers.get_all_params(output_layer, trainable=True)
learning_rate = theano.shared(np.float32(0.001))
updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

train_fn = theano.function([x_sym, seg_sym], [loss, acc_train], updates=updates)
val_fn = theano.function([x_sym, seg_sym], [loss_val, acc])
pred_fn = theano.function([x_sym], prediction_test)


from numpy import memmap

train_pos_memmap = memmap("/media/fabian/DeepLearningData/datasets/%s_train_pos.memmap" % memmap_name, dtype=np.float32, mode="r", shape=memmap_properties["train_pos_shape"])
train_neg_memmap = memmap("/media/fabian/DeepLearningData/datasets/%s_train_neg.memmap" % memmap_name, dtype=np.float32, mode="r", shape=memmap_properties["train_neg_shape"])
val_pos_memmap = memmap("/media/fabian/DeepLearningData/datasets/%s_val_pos.memmap" % memmap_name, dtype=np.float32, mode="r", shape=memmap_properties["val_pos_shape"])
val_neg_memmap = memmap("/media/fabian/DeepLearningData/datasets/%s_val_neg.memmap" % memmap_name, dtype=np.float32, mode="r", shape=memmap_properties["val_neg_shape"])


def estimate_class_weights(train_neg_memmap, train_pos_memmap, n_neg_train, n_pos_train, BATCH_SIZE, classes, n_batches=100):
    d = {}
    for c in classes:
        d[c] = 0
    n = 0
    for data, seg, labels in multi_threaded_generator(memmapGeneratorDataAugm_t1km_flair_adc_cbv(train_neg_memmap, train_pos_memmap, BATCH_SIZE, n_pos_train, n_neg_train), num_threads=2):
        for c in d.keys():
            d[c] += np.sum(seg == c)
        n += 1
        if n >= n_batches:
            break
    class_weights = np.zeros(len(classes))
    n_pixels = np.sum([d[c] for c in d.keys()])
    for c in classes:
        class_weights[c] =  float(n_pixels) /d[c]
    class_weights /= class_weights.sum()
    class_weights *= float(len(classes))
    return class_weights

# class_weights = estimate_class_weights(train_neg_memmap, train_pos_memmap, n_neg_train, n_pos_train, BATCH_SIZE, range(4), n_batches=200).astype(np.float32)

n_feedbacks_per_epoch = 50.

all_training_losses = []
all_validation_losses = []
all_validation_accuracies = []
all_training_accs = []

'''with open("../results/%s_allLossesNAccur_ep%d.pkl"% (EXPERIMENT_NAME, 0), 'r') as f:
    [all_training_losses, all_training_accs, all_validation_losses, all_validation_accuracies] = cPickle.load(f)'''

n_epochs = 10
for epoch in range(0, n_epochs):
    print "epoch: ", epoch
    train_loss = 0
    train_acc_tmp = 0
    train_loss_tmp = 0
    batch_ctr = 0
    for data, seg, labels in multi_threaded_generator(memmapGeneratorDataAugm_t1km_flair_adc_cbv(train_neg_memmap, train_pos_memmap, BATCH_SIZE, n_pos_train, n_neg_train), num_threads=2):
        if batch_ctr != 0 and batch_ctr % int(np.floor(n_batches_per_epoch/n_feedbacks_per_epoch)) == 0:
            print "number of batches: ", batch_ctr, "/", n_batches_per_epoch
            print "training_loss since last update: ", train_loss_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch), " train accuracy: ", train_acc_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch)
            all_training_losses.append(train_loss_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch))
            all_training_accs.append(train_acc_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch))
            train_loss_tmp = 0
            train_acc_tmp = 0
            printLosses(all_training_losses, all_training_accs, all_validation_losses, all_validation_accuracies, "../results/%s.png" % EXPERIMENT_NAME, n_feedbacks_per_epoch)
        # loss, acc = train_fn(data, convert_seg_map_for_crossentropy(seg, range(4)).astype(np.float32))
        seg_flat = seg.flatten()
        loss, acc = train_fn(data, seg_flat) #class_weights[seg_flat]
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
    for data, seg, labels in multi_threaded_generator(memmapGenerator_t1km_flair_adc_cbv(val_neg_memmap, val_pos_memmap, BATCH_SIZE, n_pos_val, n_neg_val), num_threads=2):
        # loss, acc = val_fn(data, convert_seg_map_for_crossentropy(seg, range(4)).astype(np.float32))
        seg_flat = seg.flatten()
        loss, acc = val_fn(data, seg_flat) #, class_weights[seg_flat]
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
    printLosses(all_training_losses, all_training_accs, all_validation_losses, all_validation_accuracies, "../results/%s.png" % EXPERIMENT_NAME, n_feedbacks_per_epoch)
    learning_rate *= 0.1
    with open("../results/%s_Params_ep%d.pkl" % (EXPERIMENT_NAME, epoch), 'w') as f:
        cPickle.dump(lasagne.layers.get_all_param_values(output_layer), f)
    with open("../results/%s_allLossesNAccur_ep%d.pkl"% (EXPERIMENT_NAME, epoch), 'w') as f:
        cPickle.dump([all_training_losses, all_training_accs, all_validation_losses, all_validation_accuracies], f)

import cPickle
with open("../results/%s_Params.pkl"%EXPERIMENT_NAME, 'w') as f:
    cPickle.dump(lasagne.layers.get_all_param_values(output_layer), f)
with open("../results/%s_allLossesNAccur.pkl"%EXPERIMENT_NAME, 'w') as f:
    cPickle.dump([all_training_losses, all_training_accs, all_validation_losses, all_validation_accuracies], f)

img_ctr = 0
epoch_ctr=0
for data, seg, labels in threaded_generator(memmapGenerator_t1km_flair_adc_cbv(val_neg_memmap, val_pos_memmap, BATCH_SIZE, n_pos_val, n_neg_val)):
    pred = pred_fn(data).argmax(-1).reshape(BATCH_SIZE, 1, 128, 128)
    img_ctr = show_segmentation_results(data, seg, pred, img_ctr=img_ctr)
    epoch_ctr += 1
    if epoch_ctr >= 8:
        break