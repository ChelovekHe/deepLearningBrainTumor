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
import batchGenerator
import cPickle

BATCH_SIZE = 256

def printLosses(all_training_losses, all_validation_losses, all_valid_accur, fname, samplesPerEpoch=10):
    fig, ax1 = plt.subplots()
    trainLoss_x_values = np.arange(1/float(samplesPerEpoch), len(all_training_losses)/float(samplesPerEpoch)+0.000001, 1/float(samplesPerEpoch))
    val_x_values = np.arange(1, len(all_validation_losses)+0.001, 1)
    ax1.plot(trainLoss_x_values, all_training_losses)
    ax1.plot(val_x_values, all_validation_losses)
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')

    ax2 = ax1.twinx()
    ax2.plot(val_x_values, all_valid_accur, color='r')
    ax2.set_ylabel('validAccur')
    for t2 in ax2.get_yticklabels():
        t2.set_color('r')

    ax1.legend(['trainLoss', 'validLoss'])
    #ax2.legend(['valAccuracy'])
    plt.savefig(fname)
    plt.close()

IMAGE_MEAN = 0.282187011048
W_NonTumorSamples = 49038./305657
W_TumorSamples = 1 / W_NonTumorSamples
N_TRAIN_IMAGES = 354695
N_TEST_PATCHES = 62588
n_batches_per_epoch = np.floor(N_TRAIN_IMAGES/float(BATCH_SIZE))
n_validation_batches = np.floor(N_TEST_PATCHES/float(BATCH_SIZE))

'''
def build_net():
    net = dict()

    net['input'] = InputLayer((BATCH_SIZE, 1, 64, 64))

    net['conv_1_1'] = ConvLayer(net['input'], 32, 7, pad=3, stride=1)
    net['conv_1_1_do'] = DropoutLayer(net['conv_1_1'], p=0.1)
    net['conv_1_2'] = ConvLayer(net['conv_1_1_do'], 64, 5, pad=2, stride=1)
    net['conv_1_2_do'] = DropoutLayer(net['conv_1_2'], p=0.1)
    net['maxPool_1'] = Pool2DLayer(net['conv_1_2_do'], 2, mode='max')

    net['conv_2_1'] = ConvLayer(net['maxPool_1'], 64, 3, pad=1, stride=1)
    net['conv_2_1_do'] = DropoutLayer(net['conv_2_1'], p=0.2)
    net['conv_2_2'] = ConvLayer(net['conv_2_1_do'], 64, 3, pad=1, stride=1)
    net['conv_2_2_do'] = DropoutLayer(net['conv_2_2'], p=0.2)
    net['maxPool_2'] = Pool2DLayer(net['conv_2_2_do'], 2, mode='max')

    net['conv_3_1'] = ConvLayer(net['maxPool_2'], 100, 3, pad=1, stride=1)
    net['conv_3_1_do'] = DropoutLayer(net['conv_3_1'], p=0.3)
    net['conv_3_2'] = ConvLayer(net['conv_3_1_do'], 100, 3, pad=1, stride=1)
    net['conv_3_2_do'] = DropoutLayer(net['conv_3_2'], p=0.3)
    net['conv_3_3'] = ConvLayer(net['conv_3_2_do'], 100, 3, pad=1, stride=1)
    net['conv_3_3_do'] = DropoutLayer(net['conv_3_3'], p=0.3)
    net['maxPool_3'] = Pool2DLayer(net['conv_3_3_do'], 2, mode='max')

    net['fc_4'] = DenseLayer(net['maxPool_1'], 320)
    net['fc_4_dropOut'] = DropoutLayer(net['fc_4'], p=0.5)

    net['prob'] = DenseLayer(net['fc_4_dropOut'], 2, nonlinearity=lasagne.nonlinearities.softmax)

    return net

my_net = build_net()

# with open("patchClassifier_v0.1_dropout_Params.pkl", 'r') as f:
#      params = cPickle.load(f)

# lasagne.layers.set_all_param_values(net['prob'], params)

x_sym = T.tensor4()
y_sym = T.ivector()
w_sym = T.vector()

output_train = lasagne.layers.get_output(my_net['prob'], x_sym, deterministic=False)
output_validation = lasagne.layers.get_output(my_net['prob'], x_sym, deterministic=True)

loss = lasagne.objectives.categorical_crossentropy(output_train, y_sym)
loss = T.mean(loss * w_sym)
loss += lasagne.regularization.regularize_network_params(my_net['prob'], lasagne.regularization.l2) * 0.00001

acc = T.mean(T.eq(T.argmax(output_validation, axis=1), y_sym), dtype=theano.config.floatX)

params = lasagne.layers.get_all_params(my_net['prob'], trainable=True)
learning_rate = theano.shared(np.float32(0.0015))
# updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.0001, momentum=0.9)
updates = lasagne.updates.adam(T.grad(loss, params), params, learning_rate=learning_rate)

train_fn = theano.function([x_sym, y_sym, w_sym], loss, updates=updates)
val_fn = theano.function([x_sym, y_sym, w_sym], [loss, acc])
pred_fn = theano.function([x_sym], output_validation)
'''


l_in = lasagne.layers.InputLayer((BATCH_SIZE, 1, 64, 64))

l_conv_1_1 = ConvLayer(l_in, num_filters=25, filter_size=3, pad=1, stride=1)
l_conv_1_1_d = DropoutLayer(l_conv_1_1, p=0.1)
l_conv_1_2 = ConvLayer(l_conv_1_1_d, num_filters=25, filter_size=3, pad=1, stride=1)
l_conv_1_2_d = DropoutLayer(l_conv_1_2, p=0.1)

l_pool_1 = Pool2DLayer(l_conv_1_2_d, pool_size=2, mode='max')

l_conv_2_1 = ConvLayer(l_pool_1, num_filters=50, filter_size=3, pad=1, stride=1)
l_conv_2_1_d = DropoutLayer(l_conv_2_1, p=0.2)
l_conv_2_2 = ConvLayer(l_conv_2_1_d, num_filters=50, filter_size=3, pad=1, stride=1)
l_conv_2_2_d = DropoutLayer(l_conv_2_2, p=0.2)

l_pool_2 = Pool2DLayer(l_conv_2_2_d, pool_size=2, mode='max')

l_conv_3_1 = ConvLayer(l_pool_2, num_filters=100, filter_size=3, pad=1, stride=1)
l_conv_3_1_d = DropoutLayer(l_conv_3_1, p=0.3)
l_conv_3_2 = ConvLayer(l_conv_3_1_d, num_filters=100, filter_size=3, pad=1, stride=1)
l_conv_3_2_d = DropoutLayer(l_conv_3_2, p=0.3)
l_conv_3_3 = ConvLayer(l_conv_3_2_d, num_filters=100, filter_size=3, pad=1, stride=1)
l_conv_3_3_d = DropoutLayer(l_conv_3_3, p=0.3)

l_pool_2 = Pool2DLayer(l_conv_3_3_d, pool_size=2, mode='max')

l_fc_1 = lasagne.layers.DenseLayer(l_pool_2, 1500, nonlinearity=lasagne.nonlinearities.rectify)
#l_fc_1_d = DropoutLayer(l_fc_1, p=0.5)

l_out = lasagne.layers.DenseLayer(l_fc_1,
                                  num_units=2,
                                  nonlinearity=lasagne.nonlinearities.softmax)

# Compile and train the network.
# Accuracy is much better than the single layer network, despite the small number of filters.
X_sym = T.tensor4()
y_sym = T.ivector()
w_sym = T.vector()

output = lasagne.layers.get_output(l_out, X_sym, deterministic=False)
pred = output.argmax(-1)

output_val = lasagne.layers.get_output(l_out, X_sym, deterministic=True)
pred_val = output_val.argmax(-1)

loss = T.mean(lasagne.objectives.categorical_crossentropy(output, y_sym) * w_sym)

l2_penalty = lasagne.regularization.regularize_network_params(l_out, lasagne.regularization.l2) * 5e-4
loss += l2_penalty

acc = T.mean(T.eq(pred_val, y_sym))

params = lasagne.layers.get_all_params(l_out)
grad = T.grad(loss, params)
#updates = lasagne.updates.momentum(grad, params, learning_rate=0.01, momentum=0.9)
updates = lasagne.updates.adam(grad, params, 0.001)
# updates = lasagne.updates.nesterov_momentum(grad, params, learning_rate=0.01)

train_fn = theano.function([X_sym, y_sym, w_sym], [loss, acc], updates=updates)

val_fn = theano.function([X_sym, y_sym, w_sym], [loss, acc])
f_predict = theano.function([X_sym], pred_val)

train_batch_generator = batchGenerator.ImagenetBatchIterator(BATCH_SIZE, "/home/fabian/datasets/Hirntumor_von_David/lmdb_train", use_caffe=False)
validation_batch_generator = batchGenerator.ImagenetBatchIterator(BATCH_SIZE, "/home/fabian/datasets/Hirntumor_von_David/lmdb_test", use_caffe=False, testing=True)

all_training_losses = []
all_validation_losses = []
all_validation_accuracies = []
n_epochs = 15
for epoch in range(n_epochs):
    print "epoch: ", epoch
    train_loss = 0
    train_loss_tmp = 0
    batch_ctr = 0
    tumor_ctr = 0
    nontumor_ctr = 0
    #for i in xrange(int(n_batches_per_epoch)):
    for data, labels, seg in batchGenerator.threaded_generator(train_batch_generator):
        if (batch_ctr+1) % int(np.floor(n_batches_per_epoch/10.)) == 0:
            print "number of batches: ", batch_ctr, "/", n_batches_per_epoch
            print "training_loss since last update: ", train_loss_tmp/np.floor(n_batches_per_epoch/10.)
            all_training_losses.append(train_loss_tmp/np.floor(n_batches_per_epoch/10.))
            train_loss_tmp = 0
            printLosses(all_training_losses, all_validation_losses, all_validation_accuracies, "patchClassifier_v0.1_progress.png", 10)
        data -= IMAGE_MEAN
        w = np.ones(BATCH_SIZE)
        w[labels == 0] = W_NonTumorSamples
        w[labels == 1] = W_TumorSamples
        tmp_loss, tmp_acc = train_fn(data, labels, w.astype(np.float32))

        for i, img in enumerate(data):
            write_me = np.repeat(img, 3, axis=0).transpose((1, 2, 0))
            write_me[:, :, 0][seg[i] > 1] = 1.
            if labels[i] == 0:
                target = "/home/fabian/datasets/Hirntumor_von_David/some_images/non-tumor/"
                plt.imsave(path.join(target, "%d.png"%nontumor_ctr), write_me)
                nontumor_ctr += 1
            else:
                target = "/home/fabian/datasets/Hirntumor_von_David/some_images/tumor/"
                plt.imsave(path.join(target, "%d.png"%tumor_ctr), write_me)
                tumor_ctr += 1
        train_loss += tmp_loss
        train_loss_tmp += tmp_loss
        batch_ctr += 1
        if batch_ctr == n_batches_per_epoch:
            break
    train_loss /= n_batches_per_epoch
    print "training loss average on epoch: ", train_loss

    validation_loss = 0
    accuracies = []
    for i in xrange(int(n_validation_batches)):
        data, labels, seg = validation_batch_generator.next()
        data -= IMAGE_MEAN
        w = np.ones(BATCH_SIZE)
        w[labels == 0] = W_NonTumorSamples
        w[labels == 1] = W_TumorSamples
        tmp_loss, tmp_acc = val_fn(data, labels, w.astype(np.float32))
        validation_loss += tmp_loss
        accuracies.append(tmp_acc)
    validation_loss /= n_validation_batches
    print "test loss: ", validation_loss
    print "test acc: ", np.mean(accuracies), "\n"
    all_validation_losses.append(validation_loss)
    all_validation_accuracies.append(np.mean(accuracies))
    printLosses(all_training_losses, all_validation_losses, all_validation_accuracies, "patchClassifier_v0.1_progress.png", 10)
    #learning_rate *= 0.8

with open("patchClassifier_v0.1_dropout_Params.pkl", 'w') as f:
    cPickle.dump(lasagne.layers.get_all_param_values(my_net['prob']), f)
with open("patchClassifier_v0.1_allLossesNAccur.pkl", 'w') as f:
    cPickle.dump([all_training_losses, all_validation_losses, all_validation_accuracies], f)

'''conv_1_1_layer = net['conv_1_1']
conv_1_1_weights = conv_1_1_layer.get_params()[0].get_value()

plt.figure(figsize=(12, 12))
for i in range(conv_1_1_weights.shape[0]):
    plt.subplot(6, 6, i+1)
    plt.imshow(conv_1_1_weights[i, 0, :, :], cmap="gray", interpolation="nearest")
    plt.axis('off')
plt.show()'''
