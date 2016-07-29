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
from utils import threaded_generator, printLosses, validate_result, plot_layer_weights
from memmap_negPos_batchgen import memmapGenerator, memmapGeneratorDataAugm, memmapGenerator_t1km_flair, memmapGeneratorDataAugm_t1km_flair, memmapGeneratorDataAugm_t1km_flair_adc_cbv, memmapGenerator_t1km_flair_adc_cbv
import cPickle
from lasagne.layers import batch_norm


EXPERIMENT_NAME = "classifyPatches_memmap_v0.7_ws_resample_t1km_flair_adc_cbv_new"
memmap_name = "patchClassification_ws_resampled_t1km_flair_adc_cbv_new"
BATCH_SIZE = 70

with open("../data/%s_properties.pkl" % memmap_name, 'r') as f:
    memmap_properties = cPickle.load(f)
n_pos_train = memmap_properties["train_pos"]
n_neg_train = memmap_properties["train_neg"]
n_pos_val = memmap_properties["val_pos"]
n_neg_val = memmap_properties["val_neg"]
n_training_samples = memmap_properties["train_total"]
n_val_samples = memmap_properties["val_total"]


def build_UNet():
    net = OrderedDict()
    net['input'] = InputLayer((BATCH_SIZE, 1, 128, 128))

    net['contr_1_1'] = batch_norm(ConvLayer(net['input'], 64, 3, nonlinearity=lasagne.nonlinearities.elu))
    net['contr_1_2'] = batch_norm(ConvLayer(net['contr_1_1'], 64, 3, nonlinearity=lasagne.nonlinearities.elu))
    net['pool1'] = Pool2DLayer(net['contr_1_2'], 2)

    net['contr_2_1'] = batch_norm(ConvLayer(net['pool1'], 128, 3, nonlinearity=lasagne.nonlinearities.elu))
    net['contr_2_2'] = batch_norm(ConvLayer(net['contr_2_1'], 128, 3, nonlinearity=lasagne.nonlinearities.elu))
    net['pool2'] = Pool2DLayer(net['contr_2_2'], 2)

    net['contr_3_1'] = batch_norm(ConvLayer(net['pool2'], 256, 3, nonlinearity=lasagne.nonlinearities.elu))
    net['contr_3_2'] = batch_norm(ConvLayer(net['contr_3_1'], 256, 3, nonlinearity=lasagne.nonlinearities.elu))
    net['pool3'] = Pool2DLayer(net['contr_3_2'], 2)

    net['contr_4_1'] = batch_norm(ConvLayer(net['pool3'], 512, 3, nonlinearity=lasagne.nonlinearities.elu))
    net['contr_4_2'] = batch_norm(ConvLayer(net['contr_4_1'], 512, 3, nonlinearity=lasagne.nonlinearities.elu))
    net['pool4'] = Pool2DLayer(net['contr_4_2'], 2)

    net['contr_5_1'] = batch_norm(ConvLayer(net['pool4'], 1024, 3, nonlinearity=lasagne.nonlinearities.elu))
    net['contr_5_2'] = batch_norm(ConvLayer(net['contr_5_1'], 1024, 3, nonlinearity=lasagne.nonlinearities.elu))
    net['deconv1'] = Deconv2DLayer(net['contr_5_2'], 512, 2)

    net['concat1'] = ConcatLayer([net['deconv1'], net['contr_4_2']], cropping='center')
    net['expand_1_1'] = batch_norm(ConvLayer(net['concat1'], 512, 3, nonlinearity=lasagne.nonlinearities.elu))
    net['expand_1_2'] = batch_norm(ConvLayer(net['expand_1_1'], 512, 3, nonlinearity=lasagne.nonlinearities.elu))
    net['deconv2'] = Deconv2DLayer(net['expand_1_2'], 256, 2)

    net['concat2'] = ConcatLayer([net['deconv2'], net['contr_3_2']], cropping='center')
    net['expand_2_1'] = batch_norm(ConvLayer(net['concat2'], 256, 3, nonlinearity=lasagne.nonlinearities.elu))
    net['expand_2_2'] = batch_norm(ConvLayer(net['expand_2_1'], 256, 3, nonlinearity=lasagne.nonlinearities.elu))
    net['deconv3'] = Deconv2DLayer(net['expand_2_2'], 128, 2)

    net['concat3'] = ConcatLayer([net['deconv3'], net['contr_2_2']], cropping='center')
    net['expand_3_1'] = batch_norm(ConvLayer(net['concat3'], 128, 3, nonlinearity=lasagne.nonlinearities.elu))
    net['expand_3_2'] = batch_norm(ConvLayer(net['expand_3_1'], 128, 3, nonlinearity=lasagne.nonlinearities.elu))
    net['deconv4'] = Deconv2DLayer(net['expand_3_2'], 64, 2)

    net['concat4'] = ConcatLayer([net['deconv4'], net['contr_1_2']], cropping='center')
    net['expand_4_1'] = batch_norm(ConvLayer(net['concat4'], 64, 3, nonlinearity=lasagne.nonlinearities.elu))
    net['expand_4_2'] = batch_norm(ConvLayer(net['expand_4_1'], 64, 3, nonlinearity=lasagne.nonlinearities.elu))

    net['segLayer'] = ConvLayer(net['expand_4_2'], 2, 1, nonlinearity=lasagne.nonlinearities.softmax)
    return net


net = build_UNet()

'''params_from = "classifyPatches_memmap_v0.3.py"
with open("../results/%s_Params.pkl"%params_from, 'r') as f:
    params = cPickle.load(f)
    lasagne.layers.set_all_param_values(net['prob'], params)'''


n_batches_per_epoch = np.floor(n_training_samples/float(BATCH_SIZE))
n_test_batches = np.floor(n_val_samples/float(BATCH_SIZE))

x_sym = T.tensor4()
y_sym = T.ivector()

l2_loss = lasagne.regularization.regularize_network_params(net, lasagne.regularization.l2) * 5e-4

prediction_train = lasagne.layers.get_output(net, x_sym, deterministic=False)
loss = lasagne.objectives.categorical_crossentropy(prediction_train, y_sym)
loss = loss.mean()
loss += l2_loss
acc_train = T.mean(T.eq(T.argmax(prediction_train, axis=1), y_sym), dtype=theano.config.floatX)

prediction_test = lasagne.layers.get_output(net, x_sym, deterministic=True)
loss_val = lasagne.objectives.categorical_crossentropy(prediction_test, y_sym)
loss_val = loss_val.mean()
loss_val += l2_loss
acc = T.mean(T.eq(T.argmax(prediction_test, axis=1), y_sym), dtype=theano.config.floatX)

params = lasagne.layers.get_all_params(net, trainable=True)
learning_rate = theano.shared(np.float32(0.001))
updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

train_fn = theano.function([x_sym, y_sym], [loss, acc_train], updates=updates)
val_fn = theano.function([x_sym, y_sym], [loss_val, acc])
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
for epoch in range(n_epochs):
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
    for data, seg, labels in threaded_generator(memmapGenerator_t1km_flair_adc_cbv(val_neg_memmap, val_pos_memmap, BATCH_SIZE, n_pos_val, n_neg_val)):
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
        cPickle.dump(lasagne.layers.get_all_param_values(net), f)
    with open("../results/%s_allLossesNAccur_ep%d.pkl"% (EXPERIMENT_NAME, epoch), 'w') as f:
        cPickle.dump([all_training_losses, all_validation_losses, all_validation_accuracies], f)

import cPickle
with open("../results/%s_Params.pkl"%EXPERIMENT_NAME, 'w') as f:
    cPickle.dump(lasagne.layers.get_all_param_values(net), f)
with open("../results/%s_allLossesNAccur.pkl"%EXPERIMENT_NAME, 'w') as f:
    cPickle.dump([all_training_losses, all_validation_losses, all_validation_accuracies], f)

