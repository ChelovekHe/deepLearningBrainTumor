__author__ = 'fabian'
import theano
import lasagne
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from lasagne.layers import InputLayer, Pool2DLayer, DenseLayer, NonlinearityLayer, DropoutLayer, BatchNormLayer, GlobalPoolLayer, ElemwiseSumLayer, PadLayer, ExpressionLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import rectify, softmax
from lasagne.layers import Deconv2DLayer, ConcatLayer, DropoutLayer
import cPickle as pickle
from collections import OrderedDict
from lasagne.layers import batch_norm, FlattenLayer, ReshapeLayer, DimshuffleLayer
from subnets import *

def build_residual_net(n_input_channels=1, BATCH_SIZE=None, input_var=None, n=5, patch_size_x=128, patch_size_y=128):
    # Building the network
    l_in = InputLayer(shape=(BATCH_SIZE, n_input_channels, patch_size_x, patch_size_y), input_var=input_var)

    # first layer, output is 16 x 128 x 128
    l = batch_norm(ConvLayer(l_in, num_filters=16, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

    # first stack of residual blocks, output is 16 x 128 x 128
    for _ in range(n):
        l = residual_block(l)

    # second stack of residual blocks, output is 32 x 64 x 64
    l = residual_block(l, increase_dim=True, projection=True)
    for _ in range(1,n):
        l = residual_block(l)

    # third stack of residual blocks, output is 64 x 32 x 32
    l = residual_block(l, increase_dim=True, projection=True)
    for _ in range(1,n):
        l = residual_block(l)

    # fourth stack of residual blocks, output is 128 x 16 x 16
    l = residual_block(l, increase_dim=True, projection=True)
    for _ in range(1,n):
        l = residual_block(l)

    # average pooling
    l = GlobalPoolLayer(l)

    # fully connected layer
    network = DenseLayer(l, num_units=2,
                         W=lasagne.init.HeNormal(),
                         nonlinearity=softmax)

    return network

def build_residual_net_noBN(n_input_channels=1, BATCH_SIZE=None, input_var=None, n=5, patch_size_x=128, patch_size_y=128):
    # Building the network
    l_in = InputLayer(shape=(BATCH_SIZE, n_input_channels, patch_size_x, patch_size_y), input_var=input_var)

    # first layer, output is 16 x 128 x 128
    l = ConvLayer(l_in, num_filters=16, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)

    # first stack of residual blocks, output is 16 x 128 x 128
    for _ in range(n):
        l = residual_block_noBN(l)

    # second stack of residual blocks, output is 32 x 64 x 64
    l = residual_block_noBN(l, increase_dim=True, projection=True)
    for _ in range(1,n):
        l = residual_block_noBN(l)

    # third stack of residual blocks, output is 64 x 32 x 32
    l = residual_block_noBN(l, increase_dim=True, projection=True)
    for _ in range(1,n):
        l = residual_block_noBN(l)

    # fourth stack of residual blocks, output is 128 x 16 x 16
    l = residual_block_noBN(l, increase_dim=True, projection=True)
    for _ in range(1,n):
        l = residual_block_noBN(l)

    # average pooling
    l = GlobalPoolLayer(l)

    # fully connected layer
    network = DenseLayer(l, num_units=2,
                         W=lasagne.init.HeNormal(),
                         nonlinearity=softmax)

    return network
