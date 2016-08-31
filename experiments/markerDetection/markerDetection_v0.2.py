# now using residual learning

__author__ = 'fabian'
import theano
import lasagne
import theano.tensor as T
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
from lasagne.layers import InputLayer, Pool2DLayer, DenseLayer, NonlinearityLayer, DropoutLayer, BatchNormLayer, GlobalPoolLayer, ElemwiseSumLayer, PadLayer, ExpressionLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import rectify, softmax
from lasagne.layers import Deconv2DLayer, ConcatLayer
import cPickle as pickle
from collections import OrderedDict
import sys
import lmdb
from utils import threaded_generator
from utils_plotting import printLosses, validate_result, plot_layer_weights
from data_generators import *
from data_augmentation_generators import *
import cPickle
from lasagne.layers import batch_norm
from neural_networks import build_residual_net, build_residual_net_noBN
from experimentsWithMPQueue import multi_threaded_generator
from numpy import memmap

EXPERIMENT_NAME = "markerDetection_MGMT_noloccues"
BATCH_SIZE = 100

memmap_name = "patchClassification_ws_resampled_t1km_flair_adc_cbv_markers_allInOne_resized"
with open("/media/fabian/DeepLearningData/datasets/%s_properties.pkl" % (memmap_name), 'r') as f:
    my_dict = cPickle.load(f)

data_ctr = my_dict['n_data']
train_shape = my_dict['train_neg_shape']
info_memmap_shape = my_dict['info_shape']

memmap_data = memmap("/media/fabian/DeepLearningData/datasets/%s.memmap" % (memmap_name), dtype=np.float32, mode="r", shape=train_shape)
memmap_gt = memmap("/media/fabian/DeepLearningData/datasets/%s_info.memmap" % (memmap_name), dtype=np.float32, mode="r", shape=info_memmap_shape)

patient_ids = np.unique(memmap_gt[:, 0]).astype(int)
# validation_patients = np.random.choice(patient_ids, 15)

validation_patients = [ 75,   1,  67,   1, 127, 120,  94, 131,  78,  74,  62,  10,  65, 47, 124]

n_training_samples = int(float(len(patient_ids) - len(validation_patients)) / float(len(patient_ids)) * memmap_data.shape[0])
n_val_samples = int(float(len(validation_patients)) / float(len(patient_ids)) * memmap_data.shape[0])

data_gen_train = memmapGenerator_allInOne_markers(memmap_data, memmap_gt, BATCH_SIZE, validation_patients, marker="MGMT", mode="train", shuffle=False)
data_gen_train = data_channel_selection_generator(data_gen_train, [0, 3, 6, 9])
data_gen_train = seg_channel_selection_generator(data_gen_train, [0])

data_gen_validation = memmapGenerator_allInOne_markers(memmap_data, memmap_gt, BATCH_SIZE, validation_patients, marker="MGMT", mode="test", shuffle=False)
data_gen_validation = data_channel_selection_generator(data_gen_validation, [0, 3, 6, 9])
data_gen_validation = seg_channel_selection_generator(data_gen_validation, [0])

data_gen_train = mirror_axis_generator(data_gen_train)
data_gen_train = elastric_transform_generator(data_gen_train, 100., 10.)

net = build_residual_net_noBN(n=1, BATCH_SIZE=BATCH_SIZE, n_input_channels=4)

'''params_from = "classifyPatches_memmap_v0.7_ws_resample_t1km_flair_adc_cbv_new_Params_ep6.pkl"
with open("../results/%s"%params_from, 'r') as f:
    params = cPickle.load(f)
    lasagne.layers.set_all_param_values(net, params)'''

# override output layer
globalPoolLayer = net.input_layer
outputLayer = DenseLayer(globalPoolLayer, num_units=2, W=lasagne.init.HeNormal(), nonlinearity=softmax)

n_batches_per_epoch = np.floor(n_training_samples/float(BATCH_SIZE))
n_test_batches = np.floor(n_val_samples/float(BATCH_SIZE))

x_sym = T.tensor4()
y_sym = T.ivector()

l2_loss = lasagne.regularization.regularize_network_params(outputLayer, lasagne.regularization.l2) * 5e-4

prediction_train = lasagne.layers.get_output(outputLayer, x_sym, deterministic=False)
loss = lasagne.objectives.categorical_crossentropy(prediction_train, y_sym)
loss = loss.mean()
loss += l2_loss
acc_train = T.mean(T.eq(T.argmax(prediction_train, axis=1), y_sym), dtype=theano.config.floatX)

prediction_test = lasagne.layers.get_output(outputLayer, x_sym, deterministic=True)
loss_val = lasagne.objectives.categorical_crossentropy(prediction_test, y_sym)
loss_val = loss_val.mean()
loss_val += l2_loss
acc = T.mean(T.eq(T.argmax(prediction_test, axis=1), y_sym), dtype=theano.config.floatX)

params = lasagne.layers.get_all_params(outputLayer, trainable=True)
learning_rate = theano.shared(np.float32(0.001))
updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

train_fn = theano.function([x_sym, y_sym], [loss, acc_train], updates=updates)
val_fn = theano.function([x_sym, y_sym], [loss_val, acc])
pred_fn = theano.function([x_sym], prediction_test)


all_training_losses = []
all_validation_losses = []
all_validation_accuracies = []
all_training_accs = []
n_epochs = 10
for epoch in range(n_epochs):
    print "epoch: ", epoch
    train_loss = 0
    train_acc_tmp = 0
    train_loss_tmp = 0
    batch_ctr = 0
    for data, seg, labels in multi_threaded_generator(data_gen_train, num_threads=8, num_cached=50):
        if batch_ctr != 0 and batch_ctr % int(np.floor(n_batches_per_epoch/10.)) == 0:
            print "number of batches: ", batch_ctr, "/", n_batches_per_epoch
            print "training_loss since last update: ", train_loss_tmp/np.floor(n_batches_per_epoch/10.), " train accuracy: ", train_acc_tmp/np.floor(n_batches_per_epoch/10.)
            all_training_losses.append(train_loss_tmp/np.floor(n_batches_per_epoch/10.))
            all_training_accs.append(train_acc_tmp/np.floor(n_batches_per_epoch/10.))
            train_loss_tmp = 0
            train_acc_tmp = 0
            printLosses(all_training_losses, all_training_accs, all_validation_losses, all_validation_accuracies, "../results/%s.png" % EXPERIMENT_NAME, 10)
        loss, acc = train_fn(data, labels.astype(np.int32))
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
    for data, seg, labels in multi_threaded_generator(data_gen_validation, num_threads=8, num_cached=50):
        loss, acc = val_fn(data, labels.astype(np.int32))
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
        cPickle.dump(lasagne.layers.get_all_param_values(outputLayer), f)
    with open("../results/%s_allLossesNAccur_ep%d.pkl"% (EXPERIMENT_NAME, epoch), 'w') as f:
        cPickle.dump([all_training_losses, all_training_accs, all_validation_losses, all_validation_accuracies], f)

import cPickle
with open("../results/%s_Params.pkl"%EXPERIMENT_NAME, 'w') as f:
    cPickle.dump(lasagne.layers.get_all_param_values(outputLayer), f)
with open("../results/%s_allLossesNAccur.pkl"%EXPERIMENT_NAME, 'w') as f:
    cPickle.dump([all_training_losses, all_training_accs, all_validation_losses, all_validation_accuracies], f)

