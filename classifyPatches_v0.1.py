__author__ = 'fabian'
import theano
import lasagne
import theano.tensor as T
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
from lasagne.layers import InputLayer, Pool2DLayer, DenseLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Deconv2DLayer, ConcatLayer
import batchGenerator

def build_net():
    net = {}

    net['input'] = InputLayer((None, 1, 64, 64))

    net['conv_1_1'] = ConvLayer(net['input'], 32, 3, pad=1)
    net['conv_1_2'] = ConvLayer(net['conv_1_1'], 32, 3, pad=1)
    net['maxPool_1'] = Pool2DLayer(net['conv_1_2'], 2)

    net['conv_2_1'] = ConvLayer(net['maxPool_1'], 64, 3, pad=1)
    net['conv_2_2'] = ConvLayer(net['conv_2_1'], 64, 3, pad=1)
    net['maxPool_2'] = Pool2DLayer(net['conv_2_2'], 2)

    net['conv_3_1'] = ConvLayer(net['maxPool_2'], 128, 3, pad=1)
    net['conv_3_2'] = ConvLayer(net['conv_3_1'], 128, 3, pad=1)
    net['maxPool_3'] = Pool2DLayer(net['conv_3_2'], 2)

    '''net['conv_4_1'] = ConvLayer(net['maxPool_3'], 256, 3, pad=1)
    net['conv_4_2'] = ConvLayer(net['conv_4_1'], 256, 3, pad=1)
    net['maxPool_4'] = Pool2DLayer(net['conv_4_2'], 2)'''

    net['fc_5'] = DenseLayer(net['maxPool_3'], 512)
    net['fc_6'] = DenseLayer(net['fc_5'], 512)
    net['fc_7'] = DenseLayer(net['fc_6'], 2, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc_7'], lasagne.nonlinearities.softmax)

    return net

net = build_net()

IMAGE_MEAN = 0.282187011048
CLASS_IMBALANCE_TRAIN = 46132./287830.
N_TRAIN_IMAGES = 333962
BATCH_SIZE = 256
N_TEST_PATCHES = 83321
n_batches_per_epoch = np.floor(N_TRAIN_IMAGES/float(BATCH_SIZE))
n_test_batches = np.floor(N_TEST_PATCHES/float(BATCH_SIZE))

x_sym = T.tensor4()
y_sym = T.ivector()
w_sym = T.vector()

prediction = lasagne.layers.get_output(net['prob'], x_sym)
loss = lasagne.objectives.categorical_crossentropy(prediction, y_sym)
loss = loss * w_sym
loss = loss.mean()

acc = T.mean(T.eq(T.argmax(prediction, axis=1), y_sym), dtype=theano.config.floatX)

params = lasagne.layers.get_all_params(net['prob'], trainable=True)
# updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.0001, momentum=0.9)
updates = lasagne.updates.adam(loss, params)

train_fn = theano.function([x_sym, y_sym, w_sym], loss, updates=updates)
val_fn = theano.function([x_sym, y_sym, w_sym], [loss, acc])
pred_fn = theano.function([x_sym], prediction)

train_batch_generator = batchGenerator.ImagenetBatchIterator(BATCH_SIZE, "/home/fabian/datasets/Hirntumor_von_David/lmdb_train", use_caffe=False)
test_batch_generator = batchGenerator.ImagenetBatchIterator(BATCH_SIZE, "/home/fabian/datasets/Hirntumor_von_David/lmdb_test", use_caffe=False, testing=True)

for epoch in range(10):
    print "epoch: ", epoch
    train_loss = 0
    train_loss_tmp = 0
    for i in xrange(int(n_batches_per_epoch)):
        if i % int(np.floor(n_batches_per_epoch/10.)) == 0:
            print "number of batches: ", i, "/", n_batches_per_epoch
            print "training_loss since last update: ", train_loss_tmp/np.floor(n_batches_per_epoch/10.)
            train_loss_tmp = 0
        data, labels, keys = train_batch_generator.next()
        data -= IMAGE_MEAN
        w = np.ones(BATCH_SIZE)
        w[labels == 0] = CLASS_IMBALANCE_TRAIN
        loss = train_fn(data, labels, w.astype(np.float32))
        train_loss += loss
        train_loss_tmp += loss
    train_loss /= n_batches_per_epoch
    print "training loss average on epoch: ", train_loss

    test_loss = 0
    accuracies = []
    for i in xrange(int(n_test_batches)):
        data, labels, keys = test_batch_generator.next()
        data -=IMAGE_MEAN
        w = np.ones(BATCH_SIZE)
        w[labels == 0] = CLASS_IMBALANCE_TRAIN
        loss, acc = val_fn(data, labels, w.astype(np.float32))
        test_loss += loss
        accuracies.append(acc)
    test_loss /= n_test_batches
    print "test loss: ", test_loss
    print "test acc: ", np.mean(accuracies), "\n"



conv_1_1_layer = net['conv_1_1']
conv_1_1_weights = conv_1_1_layer.get_params()[0].get_value()

plt.figure(figsize=(12, 12))
for i in range(conv_1_1_weights.shape[0]):
    plt.subplot(6, 6, i+1)
    plt.imshow(conv_1_1_weights[i, 0, :, :], cmap="gray", interpolation="nearest")
    plt.axis('off')
plt.show()