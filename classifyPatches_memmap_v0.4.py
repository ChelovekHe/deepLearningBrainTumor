# DESCR: Now using elu instead of ReLU's. Seems to work better (this network actually achieves 98% accuracy). Also uses Batch Normalization

__author__ = 'fabian'
import theano
import lasagne
import theano.tensor as T
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
from lasagne.layers import InputLayer, Pool2DLayer, DenseLayer, NonlinearityLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Deconv2DLayer, ConcatLayer
import cPickle as pickle
from collections import OrderedDict
import sys
import lmdb
import cPickle
from utils import threaded_generator, printLosses, validate_result
from memmap_negPos_batchgen import memmapGenerator

n_training_samples = 374546
n_pos_train = 25702
n_neg_train = 348844
n_pos_val = 1363
n_neg_val = 18614
n_val_samples = 19977
CLASS_IMBALANCE = n_training_samples/float(n_pos_train) # negative/positive examples
w_0 = 1
w_1 = n_training_samples/float(n_pos_train)
EXPERIMENT_NAME = "classifyPatches_memmap_v0.4.py"
BATCH_SIZE = 128


def build_net():
    net = OrderedDict()

    net['input'] = InputLayer((BATCH_SIZE, 1, 128, 128))
    net['norm'] = lasagne.layers.BatchNormLayer(net['input'])

    net['conv_1_1'] = ConvLayer(net['norm'], 16, 3, pad=1, stride=1, nonlinearity=lasagne.nonlinearities.elu)
    net['conv_1_1_do'] = DropoutLayer(net['conv_1_1'], p=0.1)
    net['conv_1_2'] = ConvLayer(net['conv_1_1'], 16, 3, pad=1, stride=1, nonlinearity=lasagne.nonlinearities.elu)
    net['conv_1_2_do'] = DropoutLayer(net['conv_1_2'], p=0.1)
    net['maxPool_1'] = Pool2DLayer(net['conv_1_2'], 2, mode='max')

    net['conv_2_1'] = ConvLayer(net['maxPool_1'], 32, 3, pad=1, stride=1, nonlinearity=lasagne.nonlinearities.elu)
    net['conv_2_1_do'] = DropoutLayer(net['conv_2_1'], p=0.2)
    net['conv_2_2'] = ConvLayer(net['conv_2_1'], 32, 3, pad=1, stride=1, nonlinearity=lasagne.nonlinearities.elu)
    net['conv_2_2_do'] = DropoutLayer(net['conv_2_2'], p=0.2)
    net['maxPool_2'] = Pool2DLayer(net['conv_2_2'], 2, mode='max')

    net['conv_3_1'] = ConvLayer(net['maxPool_2'], 64, 3, pad=1, stride=1, nonlinearity=lasagne.nonlinearities.elu)
    net['conv_3_1_do'] = DropoutLayer(net['conv_3_1'], p=0.3)
    net['conv_3_2'] = ConvLayer(net['conv_3_1'], 64, 3, pad=1, stride=1, nonlinearity=lasagne.nonlinearities.elu)
    net['conv_3_2_do'] = DropoutLayer(net['conv_3_2'], p=0.3)
    net['conv_3_3'] = ConvLayer(net['conv_3_2'], 64, 3, pad=1, stride=1, nonlinearity=lasagne.nonlinearities.elu)
    net['conv_3_3_do'] = DropoutLayer(net['conv_3_3'], p=0.3)
    net['maxPool_3'] = Pool2DLayer(net['conv_3_3'], 2, mode='max')

    net['fc_4'] = DenseLayer(net['maxPool_3'], 150, nonlinearity=lasagne.nonlinearities.elu)
    net['fc_4_dropOut'] = DropoutLayer(net['fc_4'], p=0.5)

    net['prob'] = DenseLayer(net['fc_4_dropOut'], 2, nonlinearity=lasagne.nonlinearities.softmax)

    return net

net = build_net()

'''with open("../results/classifyPatches_memmap_v0.4.py_Params.pkl", 'r') as f:
    params = cPickle.load(f)
lasagne.layers.set_all_param_values(net['prob'], params)
with open("../results/classifyPatches_memmap_v0.4.py_allLossesNAccur.pkl", 'r') as f:
    [all_training_losses, all_validation_losses, all_validation_accuracies] = cPickle.load(f)'''

n_batches_per_epoch = np.floor(n_training_samples/float(BATCH_SIZE))
n_test_batches = np.floor(n_val_samples/float(BATCH_SIZE))

x_sym = T.tensor4()
y_sym = T.ivector()
w_sym = T.vector()

prediction_train = lasagne.layers.get_output(net['prob'], x_sym, deterministic=False)
prediction_test = lasagne.layers.get_output(net['prob'], x_sym, deterministic=True)
loss = lasagne.objectives.categorical_crossentropy(prediction_train, y_sym)

loss = loss.mean()

l2_loss = lasagne.regularization.regularize_network_params(net['prob'], lasagne.regularization.l2) * 5e-4
loss += l2_loss

acc = T.mean(T.eq(T.argmax(prediction_test, axis=1), y_sym), dtype=theano.config.floatX)

params = lasagne.layers.get_all_params(net['prob'], trainable=True)
learning_rate = theano.shared(np.float32(0.001))
updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

train_fn = theano.function([x_sym, y_sym], [loss, acc], updates=updates)
val_fn = theano.function([x_sym, y_sym], [loss, acc])
pred_fn = theano.function([x_sym], prediction_test)


from numpy import memmap
train_pos_memmap = memmap("../data/patchClassification128_pos_train_2.memmap", dtype=np.float32, mode="r+", shape=(450000 * 10000. / 126964., 128*128*2))
train_neg_memmap = memmap("../data/patchClassification128_neg_train_2.memmap", dtype=np.float32, mode="r+", shape=(450000, 128 * 128 * 2))
val_pos_memmap = memmap("../data/patchClassification128_pos_val_2.memmap", dtype=np.float32, mode="r+", shape=(450000 * 10000. / 126964 * 0.15, 128 * 128 * 2))
val_neg_memmap = memmap("../data/patchClassification128_neg_val_2.memmap", dtype=np.float32, mode="r+", shape=(450000 * 0.15, 128 * 128 * 2))


all_training_losses = []
all_validation_losses = []
all_validation_accuracies = []
n_epochs = 50
for epoch in range(n_epochs):
    print "epoch: ", epoch
    train_loss = 0
    train_acc_tmp = 0
    train_loss_tmp = 0
    batch_ctr = 0
    for data, seg, labels in threaded_generator(memmapGenerator(train_neg_memmap, train_pos_memmap, BATCH_SIZE, n_pos_train, n_neg_train)):
        if np.sum(labels) == 0:
            continue
        if batch_ctr != 0 and batch_ctr % int(np.floor(n_batches_per_epoch/10.)) == 0:
            print "number of batches: ", batch_ctr, "/", n_batches_per_epoch
            print "training_loss since last update: ", train_loss_tmp/np.floor(n_batches_per_epoch/10.), " train accuracy: ", train_acc_tmp/np.floor(n_batches_per_epoch/10.)
            all_training_losses.append(train_loss_tmp/np.floor(n_batches_per_epoch/10.))
            train_loss_tmp = 0
            train_acc_tmp = 0
            printLosses(all_training_losses, all_validation_losses, all_validation_accuracies, "../results/%s.png" % EXPERIMENT_NAME, 10)
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
    printLosses(all_training_losses, all_validation_losses, all_validation_accuracies, "../results/%s.png" % EXPERIMENT_NAME, 10)
    learning_rate *= 0.9

import IPython
IPython.embed()

with open("../results/%s_Params.pkl"%EXPERIMENT_NAME, 'w') as f:
    cPickle.dump(lasagne.layers.get_all_param_values(net['prob']), f)
with open("../results/%s_allLossesNAccur.pkl"%EXPERIMENT_NAME, 'w') as f:
    cPickle.dump([all_training_losses, all_validation_losses, all_validation_accuracies], f)

'''
conv_1_1_layer = net['conv_1_1']
conv_1_1_weights = conv_1_1_layer.get_params()[0].get_value()

plt.figure(figsize=(12, 12))
for i in range(conv_1_1_weights.shape[0]):
    plt.subplot(int(np.floor(conv_1_1_weights.shape[0])), int(np.floor(conv_1_1_weights.shape[0])), i+1)
    plt.imshow(conv_1_1_weights[i, 0, :, :], cmap="gray", interpolation="nearest")
    plt.axis('off')
plt.show()
'''
