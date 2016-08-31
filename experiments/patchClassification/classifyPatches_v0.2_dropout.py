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
from collections import OrderedDict

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
    # ax2.legend(['valAccuracy'])
    plt.savefig(fname)
    plt.close()

BATCH_SIZE = 64

def build_net():
    net = OrderedDict()

    net['input'] = InputLayer((BATCH_SIZE, 1, 128, 128))

    net['conv_1_1'] = ConvLayer(net['input'], 32, 7, pad=3, stride=1)
    # net['conv_1_1_do'] = DropoutLayer(net['conv_1_1'], p=0.1)
    net['conv_1_2'] = ConvLayer(net['conv_1_1'], 32, 5, pad=2, stride=1)
    # net['conv_1_2_do'] = DropoutLayer(net['conv_1_2'], p=0.1)
    net['maxPool_1'] = Pool2DLayer(net['conv_1_2'], 2, mode='max')

    net['conv_2_1'] = ConvLayer(net['maxPool_1'], 64, 3, pad=1, stride=1)
    # net['conv_2_1_do'] = DropoutLayer(net['conv_2_1'], p=0.2)
    net['conv_2_2'] = ConvLayer(net['conv_2_1'], 64, 3, pad=1, stride=1)
    # net['conv_2_2_do'] = DropoutLayer(net['conv_2_2'], p=0.2)
    net['maxPool_2'] = Pool2DLayer(net['conv_2_2'], 2, mode='max')

    net['conv_3_1'] = ConvLayer(net['maxPool_2'], 128, 3, pad=1, stride=1)
    # net['conv_3_1_do'] = DropoutLayer(net['conv_3_1'], p=0.3)
    net['conv_3_2'] = ConvLayer(net['conv_3_1'], 128, 3, pad=1, stride=1)
    # net['conv_3_2_do'] = DropoutLayer(net['conv_3_2'], p=0.3)
    net['conv_3_3'] = ConvLayer(net['conv_3_2'], 128, 3, pad=1, stride=1)
    # net['conv_3_3_do'] = DropoutLayer(net['conv_3_3'], p=0.3)
    net['maxPool_3'] = Pool2DLayer(net['conv_3_3'], 2, mode='max')

    net['fc_4'] = DenseLayer(net['maxPool_3'], 2048)
    # net['fc_4_dropOut'] = DropoutLayer(net['fc_4'], p=0.5)

    net['prob'] = DenseLayer(net['fc_4'], 2, nonlinearity=lasagne.nonlinearities.softmax)

    return net

'''import nolearn
import nolearn.lasagne

net_nolearn = nolearn.lasagne.NeuralNet(
    [(InputLayer, {'shape': (BATCH_SIZE, 1, 128, 128)}),
    (ConvLayer, {'num_filters': 32, 'filter_size': 3, 'pad': 1}),
    (ConvLayer, {'num_filters': 32, 'filter_size': 3, 'pad': 1}),
    (Pool2DLayer, {'pool_size': 2}),
    (ConvLayer, {'num_filters': 64, 'filter_size': 3, 'pad': 1}),
    (ConvLayer, {'num_filters': 64, 'filter_size': 3, 'pad': 1}),
    (Pool2DLayer, {'pool_size': 2}),
    (ConvLayer, {'num_filters': 128, 'filter_size': 3, 'pad': 1}),
    (ConvLayer, {'num_filters': 128, 'filter_size': 3, 'pad': 1}),
    (ConvLayer, {'num_filters': 128, 'filter_size': 3, 'pad': 1}),
    (Pool2DLayer, {'pool_size': 2}),
    (ConvLayer, {'num_filters': 256, 'filter_size': 3, 'pad': 1}),
    (ConvLayer, {'num_filters': 256, 'filter_size': 3, 'pad': 1}),
    (ConvLayer, {'num_filters': 256, 'filter_size': 3, 'pad': 1}),
    (Pool2DLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 512}),
    (DenseLayer, {'num_units': 512}),
    (DenseLayer, {'num_units': 2, 'nonlinearity': lasagne.nonlinearities.softmax})],
    max_epochs=10,
    update=lasagne.updates.adam,
    update_learning_rate=0.0002,
    objective_l2=0.0025,
    verbose=2
)'''

net = build_net()

# with open("patchClassifier_v0.1_dropout_Params.pkl", 'r') as f:
#      params = cPickle.load(f)

# lasagne.layers.set_all_param_values(net['prob'], params)

IMAGE_MEAN = 0.245669156554
W_NonTumorSamples = 25182./89234.
W_TumorSamples = 1 / W_NonTumorSamples
N_TRAIN_IMAGES = 114416
N_TEST_PATCHES = 20094
n_batches_per_epoch = np.floor(N_TRAIN_IMAGES/float(BATCH_SIZE))
n_test_batches = np.floor(N_TEST_PATCHES/float(BATCH_SIZE))

x_sym = T.tensor4()
y_sym = T.ivector()
w_sym = T.vector()

prediction_train = lasagne.layers.get_output(net['prob'], x_sym)
prediction_test = lasagne.layers.get_output(net['prob'], x_sym)
loss = lasagne.objectives.categorical_crossentropy(prediction_train, y_sym)

# loss = loss * w_sym
loss = loss.mean()

# l2_loss = lasagne.regularization.regularize_network_params(net['prob'], lasagne.regularization.l2) * 5e-4
# loss += l2_loss

acc = T.mean(T.eq(T.argmax(prediction_test, axis=1), y_sym), dtype=theano.config.floatX)

params = lasagne.layers.get_all_params(net['prob'], trainable=True)
# updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.001)
learning_rate = theano.shared(np.float32(0.001))
updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

#train_fn = theano.function([x_sym, y_sym, w_sym], [loss, acc], updates=updates)
#val_fn = theano.function([x_sym, y_sym, w_sym], [loss, acc])
train_fn = theano.function([x_sym, y_sym], [loss, acc], updates=updates)
val_fn = theano.function([x_sym, y_sym], [loss, acc])
pred_fn = theano.function([x_sym], prediction_test)

train_batch_generator = batchGenerator.ImagenetBatchIterator(BATCH_SIZE, "/home/fabian/datasets/Hirntumor_von_David/lmdb_train128", use_caffe=False)
test_batch_generator = batchGenerator.ImagenetBatchIterator(BATCH_SIZE, "/home/fabian/datasets/Hirntumor_von_David/lmdb_test128", use_caffe=False, testing=True)

all_training_losses = []
all_validation_losses = []
all_validation_accuracies = []
n_epochs = 10
for epoch in range(n_epochs):
    print "epoch: ", epoch
    train_loss = 0
    train_acc_tmp = 0
    train_loss_tmp = 0
    batch_ctr = 0
    # for i in xrange(int(n_batches_per_epoch)):
    for data, labels, seg in batchGenerator.threaded_generator(train_batch_generator):
        if epoch == 1:
            import IPython
            IPython.embed()
        if (batch_ctr+1) % int(np.floor(n_batches_per_epoch/10.)) == 0:
            print "number of batches: ", batch_ctr, "/", n_batches_per_epoch
            print "training_loss since last update: ", train_loss_tmp/np.floor(n_batches_per_epoch/10.), " train accuracy: ", train_acc_tmp/np.floor(n_batches_per_epoch/10.)
            all_training_losses.append(train_loss_tmp/np.floor(n_batches_per_epoch/10.))
            train_loss_tmp = 0
            train_acc_tmp = 0
            printLosses(all_training_losses, all_validation_losses, all_validation_accuracies, "patchClassifier_v0.2_progress.png", 10)
        # data, labels, _ = train_batch_generator.next()
        data -= IMAGE_MEAN
        w = np.ones(BATCH_SIZE)
        w[labels == 0] = W_NonTumorSamples
        w[labels == 1] = W_TumorSamples
        loss, acc = train_fn(data, labels)
        train_loss += loss
        train_loss_tmp += loss
        train_acc_tmp += acc
        batch_ctr += 1
        if batch_ctr == n_batches_per_epoch:
            break

    train_loss /= n_batches_per_epoch
    print "training loss average on epoch: ", train_loss

    test_loss = 0
    accuracies = []
    # all_labels = []
    for i in xrange(int(n_test_batches)):
        data, labels, _ = test_batch_generator.next()
        data -= IMAGE_MEAN
        w = np.ones(BATCH_SIZE)
        w[labels == 0] = W_NonTumorSamples
        w[labels == 1] = W_TumorSamples
        loss, acc = val_fn(data, labels)
        test_loss += loss
        accuracies.append(acc)
    test_loss /= n_test_batches
    print "test loss: ", test_loss
    print "test acc: ", np.mean(accuracies), "\n"
    all_validation_losses.append(test_loss)
    all_validation_accuracies.append(np.mean(accuracies))
    printLosses(all_training_losses, all_validation_losses, all_validation_accuracies, "patchClassifier_v0.2_progress.png", 10)
    learning_rate = learning_rate * 0.9

'''with open("patchClassifier_v0.2_dropout_Params.pkl", 'w') as f:
    cPickle.dump(lasagne.layers.get_all_param_values(net['prob']), f)
with open("patchClassifier_v0.2_allLossesNAccur.pkl", 'w') as f:
    cPickle.dump([all_training_losses, all_validation_losses, all_validation_accuracies], f)

conv_1_1_layer = net['conv_1_1']
conv_1_1_weights = conv_1_1_layer.get_params()[0].get_value()

plt.figure(figsize=(12, 12))
for i in range(conv_1_1_weights.shape[0]):
    plt.subplot(6, 6, i+1)
    plt.imshow(conv_1_1_weights[i, 0, :, :], cmap="gray", interpolation="nearest")
    plt.axis('off')
plt.show()
'''