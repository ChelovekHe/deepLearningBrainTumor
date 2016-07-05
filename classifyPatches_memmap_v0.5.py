__author__ = 'fabian'
import theano
import lasagne
import theano.tensor as T
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
from lasagne.layers import InputLayer, Pool2DLayer, DenseLayer, NonlinearityLayer, DropoutLayer, BatchNormLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Deconv2DLayer, ConcatLayer
import cPickle as pickle
from collections import OrderedDict
import sys
import lmdb
from utils import threaded_generator, printLosses, validate_result, plot_layer_weights
from memmap_negPos_batchgen import memmapGenerator, memmapGeneratorDataAugm
import cPickle
from lasagne.layers import batch_norm

with open("../data/patchClassification_memmap_properties_2.pkl", 'r') as f:
    memmap_properties = cPickle.load(f)
n_pos_train = memmap_properties["train_pos"]
n_neg_train = memmap_properties["train_neg"]
n_pos_val = memmap_properties["val_pos"]
n_neg_val = memmap_properties["val_neg"]
n_training_samples = memmap_properties["train_total"]
n_val_samples = memmap_properties["val_total"]

EXPERIMENT_NAME = "classifyPatches_memmap_v0.5.py"
BATCH_SIZE = 100

def build_net():
    net = OrderedDict()

    net['input'] = InputLayer((BATCH_SIZE, 1, 128, 128))

    net['conv_1_1'] = batch_norm(ConvLayer(net['input'], 12, 7, pad='same', stride=1, nonlinearity=lasagne.nonlinearities.elu))
    net['conv_1_1_do'] = DropoutLayer(net['conv_1_1'], p=0.1)
    net['conv_1_2'] = batch_norm(ConvLayer(net['conv_1_1_do'], 12, 5, pad='same', stride=1, nonlinearity=lasagne.nonlinearities.elu))
    net['maxPool_1_1'] = Pool2DLayer(net['conv_1_2'], 2, mode='max')
    net['conv_1_2_do'] = DropoutLayer(net['maxPool_1_1'], p=0.1)

    net['conv_2_1'] = batch_norm(ConvLayer(net['conv_1_2_do'], 24, 3, pad='same', stride=1, nonlinearity=lasagne.nonlinearities.elu))
    net['conv_2_1_do'] = DropoutLayer(net['conv_2_1'], p=0.2)
    net['conv_2_2'] = batch_norm(ConvLayer(net['conv_2_1_do'], 24, 3, pad='same', stride=1, nonlinearity=lasagne.nonlinearities.elu))
    net['maxPool_2_1'] = Pool2DLayer(net['conv_2_2'], 2, mode='max')
    net['conv_2_2_do'] = DropoutLayer(net['maxPool_2_1'], p=0.2)

    net['conv_3_1'] = batch_norm(ConvLayer(net['conv_2_2_do'], 48, 3, pad=1, stride=1, nonlinearity=lasagne.nonlinearities.elu))
    net['maxPool_3_1'] = Pool2DLayer(net['conv_3_1'], 2, mode='max')
    net['conv_3_1_do'] = DropoutLayer(net['maxPool_3_1'], p=0.3)

    net['conv_3_2'] = batch_norm(ConvLayer(net['conv_3_1_do'], 48, 3, pad=1, stride=1, nonlinearity=lasagne.nonlinearities.elu))
    net['maxPool_3_2'] = Pool2DLayer(net['conv_3_2'], 2, mode='max')
    net['conv_3_2_do'] = DropoutLayer(net['maxPool_3_2'], p=0.3)

    net['conv_3_3'] = batch_norm(ConvLayer(net['conv_3_2_do'], 48, 3, pad=1, stride=1, nonlinearity=lasagne.nonlinearities.elu))
    net['maxPool_3_3'] = Pool2DLayer(net['conv_3_3'], 2, mode='max')
    net['conv_3_3_do'] = DropoutLayer(net['maxPool_3_3'], p=0.3)

    net['fc_4'] = batch_norm(DenseLayer(net['conv_3_3_do'], 200, nonlinearity=lasagne.nonlinearities.elu))
    net['fc_4_dropOut'] = DropoutLayer(net['fc_4'], p=0.5)

    net['prob'] = batch_norm(DenseLayer(net['fc_4_dropOut'], 2, nonlinearity=lasagne.nonlinearities.softmax))

    return net

net = build_net()

params_from = "classifyPatches_memmap_v0.3.py"
with open("../results/%s_Params.pkl"%params_from, 'r') as f:
    params = cPickle.load(f)
    lasagne.layers.set_all_param_values(net['prob'], params)


n_batches_per_epoch = np.floor(n_training_samples/float(BATCH_SIZE))
n_test_batches = np.floor(n_val_samples/float(BATCH_SIZE))

x_sym = T.tensor4()
y_sym = T.ivector()

l2_loss = lasagne.regularization.regularize_network_params(net['prob'], lasagne.regularization.l2) * 5e-4

prediction_train = lasagne.layers.get_output(net['prob'], x_sym, deterministic=False)
loss = lasagne.objectives.categorical_crossentropy(prediction_train, y_sym)
loss = loss.mean()
loss += l2_loss
acc_train = T.mean(T.eq(T.argmax(prediction_train, axis=1), y_sym), dtype=theano.config.floatX)

prediction_test = lasagne.layers.get_output(net['prob'], x_sym, deterministic=True)
loss_val = lasagne.objectives.categorical_crossentropy(prediction_test, y_sym)
loss_val = loss_val.mean()
loss_val += l2_loss
acc = T.mean(T.eq(T.argmax(prediction_test, axis=1), y_sym), dtype=theano.config.floatX)

params = lasagne.layers.get_all_params(net['prob'], trainable=True)
learning_rate = theano.shared(np.float32(0.001))
updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

train_fn = theano.function([x_sym, y_sym], [loss, acc_train], updates=updates)
val_fn = theano.function([x_sym, y_sym], [loss_val, acc])
pred_fn = theano.function([x_sym], prediction_test)


from numpy import memmap

train_pos_memmap = memmap("../data/patchClassification_train_pos_2.memmap", dtype=np.float32, mode="r+", shape=memmap_properties["train_pos_shape"])
train_neg_memmap = memmap("../data/patchClassification_train_neg_2.memmap", dtype=np.float32, mode="r+", shape=memmap_properties["train_neg_shape"])
val_pos_memmap = memmap("../data/patchClassification_val_pos_2.memmap", dtype=np.float32, mode="r+", shape=memmap_properties["val_pos_shape"])
val_neg_memmap = memmap("../data/patchClassification_val_neg_2.memmap", dtype=np.float32, mode="r+", shape=memmap_properties["val_neg_shape"])


all_training_losses = []
all_validation_losses = []
all_validation_accuracies = []
all_training_accs = []
n_epochs = 7
for epoch in range(n_epochs):
    print "epoch: ", epoch
    train_loss = 0
    train_acc_tmp = 0
    train_loss_tmp = 0
    batch_ctr = 0
    for data, seg, labels in threaded_generator(memmapGeneratorDataAugm(train_neg_memmap, train_pos_memmap, BATCH_SIZE, n_pos_train, n_neg_train)):
        if batch_ctr != 0 and batch_ctr % int(np.floor(n_batches_per_epoch/10.)) == 0:
            print "number of batches: ", batch_ctr, "/", n_batches_per_epoch
            print "training_loss since last update: ", train_loss_tmp/np.floor(n_batches_per_epoch/10.), " train accuracy: ", train_acc_tmp/np.floor(n_batches_per_epoch/10.)
            all_training_losses.append(train_loss_tmp/np.floor(n_batches_per_epoch/10.))
            all_training_accs.append(train_acc_tmp/np.floor(n_batches_per_epoch/10.))
            train_loss_tmp = 0
            train_acc_tmp = 0
            printLosses(all_training_losses, all_training_accs, all_validation_losses, all_validation_accuracies, "../results/%s.png" % EXPERIMENT_NAME, 10)
        loss, acc = train_fn(data, labels)
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
    for data, seg, labels in threaded_generator(memmapGenerator(val_neg_memmap, val_pos_memmap, BATCH_SIZE, n_pos_val, n_neg_val)):
        loss, acc = val_fn(data, labels)
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
    learning_rate *= 0.3
    with open("../results/%s_Params_ep%d.pkl" % (EXPERIMENT_NAME, epoch), 'w') as f:
        cPickle.dump(lasagne.layers.get_all_param_values(net['prob']), f)
    with open("../results/%s_allLossesNAccur_ep%d.pkl"% (EXPERIMENT_NAME, epoch), 'w') as f:
        cPickle.dump([all_training_losses, all_validation_losses, all_validation_accuracies], f)

import cPickle
with open("../results/%s_Params.pkl"%EXPERIMENT_NAME, 'w') as f:
    cPickle.dump(lasagne.layers.get_all_param_values(net['prob']), f)
with open("../results/%s_allLossesNAccur.pkl"%EXPERIMENT_NAME, 'w') as f:
    cPickle.dump([all_training_losses, all_validation_losses, all_validation_accuracies], f)

