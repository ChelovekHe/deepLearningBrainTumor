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
from utils import threaded_generator, printLosses, validate_result, plot_layer_weights
from memmap_negPos_batchgen import memmapGenerator, memmapGeneratorDataAugm, memmapGenerator_t1km_flair, memmapGeneratorDataAugm_t1km_flair, memmapGeneratorDataAugm_t1km_flair_adc_cbv, memmapGenerator_t1km_flair_adc_cbv, memmapGenerator_tumorClassRot
import cPickle
from lasagne.layers import batch_norm
from neural_networks import build_UNet, build_residual_net


EXPERIMENT_NAME = "segment_tumor_v0.1"
memmap_name = "patchClassification_ws_resampled_t1km_flair_adc_cbv_new"
BATCH_SIZE = 10

with open("../data/%s_properties.pkl" % memmap_name, 'r') as f:
    memmap_properties = cPickle.load(f)
n_pos_train = memmap_properties["train_pos"]
n_neg_train = memmap_properties["train_neg"]
n_pos_val = memmap_properties["val_pos"]
n_neg_val = memmap_properties["val_neg"]
n_training_samples = memmap_properties["train_total"]
n_val_samples = memmap_properties["val_total"]


net = build_UNet(4, BATCH_SIZE, num_output_classes=4)

'''params_from = "classifyPatches_memmap_v0.3.py"
with open("../results/%s_Params.pkl"%params_from, 'r') as f:
    params = cPickle.load(f)
    lasagne.layers.set_all_param_values(net['prob'], params)'''


output_layer = net["output"]
n_batches_per_epoch = np.floor(n_training_samples/float(BATCH_SIZE))
n_test_batches = np.floor(n_val_samples/float(BATCH_SIZE))

x_sym = T.tensor4()
seg_sym = T.ivector()

l2_loss = lasagne.regularization.regularize_network_params(output_layer, lasagne.regularization.l2) * 5e-4

prediction_train = lasagne.layers.get_output(output_layer, x_sym, deterministic=False)
loss = lasagne.objectives.categorical_crossentropy(prediction_train, seg_sym)
loss = loss.mean()
loss += l2_loss
acc_train = T.mean(T.eq(T.argmax(prediction_train, axis=1), seg_sym), dtype=theano.config.floatX)

prediction_test = lasagne.layers.get_output(output_layer, x_sym, deterministic=True)
loss_val = lasagne.objectives.categorical_crossentropy(prediction_test, seg_sym)
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

train_pos_memmap = memmap("../data/%s_train_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["train_pos_shape"])
train_neg_memmap = memmap("../data/%s_train_neg.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["train_neg_shape"])
val_pos_memmap = memmap("../data/%s_val_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["val_pos_shape"])
val_neg_memmap = memmap("../data/%s_val_neg.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["val_neg_shape"])


all_training_losses = []
all_validation_losses = []
all_validation_accuracies = []
all_training_accs = []
n_epochs = 20
for epoch in range(0,n_epochs):
    print "epoch: ", epoch
    train_loss = 0
    train_acc_tmp = 0
    train_loss_tmp = 0
    batch_ctr = 0
    for data, seg, labels in threaded_generator(memmapGeneratorDataAugm_t1km_flair_adc_cbv(train_neg_memmap, train_pos_memmap, BATCH_SIZE, n_pos_train, n_neg_train)):
        if batch_ctr != 0 and batch_ctr % int(np.floor(n_batches_per_epoch/10.)) == 0:
            print "number of batches: ", batch_ctr, "/", n_batches_per_epoch
            print "training_loss since last update: ", train_loss_tmp/np.floor(n_batches_per_epoch/10.), " train accuracy: ", train_acc_tmp/np.floor(n_batches_per_epoch/10.)
            all_training_losses.append(train_loss_tmp/np.floor(n_batches_per_epoch/10.))
            all_training_accs.append(train_acc_tmp/np.floor(n_batches_per_epoch/10.))
            train_loss_tmp = 0
            train_acc_tmp = 0
            printLosses(all_training_losses, all_training_accs, all_validation_losses, all_validation_accuracies, "../results/%s.png" % EXPERIMENT_NAME, 10)
        loss, acc = train_fn(data, seg.flatten())
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
    for data, seg, labels in threaded_generator(memmapGenerator_t1km_flair_adc_cbv(val_neg_memmap, val_pos_memmap, BATCH_SIZE, n_pos_val, n_neg_val)):
        loss, acc = val_fn(data, seg.flatten())
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
    printLosses(all_training_losses, all_training_accs, all_validation_losses, all_validation_accuracies, "../results/%s.png" % EXPERIMENT_NAME, 10)
    learning_rate *= 0.1
    with open("../results/%s_Params_ep%d.pkl" % (EXPERIMENT_NAME, epoch), 'w') as f:
        cPickle.dump(lasagne.layers.get_all_param_values(output_layer), f)
    with open("../results/%s_allLossesNAccur_ep%d.pkl"% (EXPERIMENT_NAME, epoch), 'w') as f:
        cPickle.dump([all_training_losses, all_validation_losses, all_validation_accuracies], f)

import cPickle
with open("../results/%s_Params.pkl"%EXPERIMENT_NAME, 'w') as f:
    cPickle.dump(lasagne.layers.get_all_param_values(net), f)
with open("../results/%s_allLossesNAccur.pkl"%EXPERIMENT_NAME, 'w') as f:
    cPickle.dump([all_training_losses, all_validation_losses, all_validation_accuracies], f)

