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


# create a residual learning building block with two stacked 3x3 convlayers as in paper
def residual_block(l, increase_dim=False, projection=False):
    input_num_filters = l.output_shape[1]
    if increase_dim:
        first_stride = (2, 2)
        out_num_filters = input_num_filters*2
    else:
        first_stride = (1, 1)
        out_num_filters = input_num_filters

    stack_1 = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
    stack_2 = batch_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

    # add shortcut connections
    if increase_dim:
        if projection:
            # projection shortcut, as option B in paper
            projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None, flip_filters=False))
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]),nonlinearity=rectify)
        else:
            # identity shortcut, as option A in paper
            identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
            padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]),nonlinearity=rectify)
    else:
        block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=rectify)

    return block

def residual_block_noBN(l, increase_dim=False, projection=False):
    input_num_filters = l.output_shape[1]
    if increase_dim:
        first_stride = (2, 2)
        out_num_filters = input_num_filters*2
    else:
        first_stride = (1, 1)
        out_num_filters = input_num_filters

    stack_1 = ConvLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)
    stack_2 = ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)

    # add shortcut connections
    if increase_dim:
        if projection:
            # projection shortcut, as option B in paper
            projection = ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None, flip_filters=False)
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]),nonlinearity=rectify)
        else:
            # identity shortcut, as option A in paper
            identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
            padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]),nonlinearity=rectify)
    else:
        block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=rectify)

    return block


# create a residual learning building block with two stacked 3x3 convlayers as in paper
def residual_block_noPool(l, increase_dim=False, projection=True):
    input_num_filters = l.output_shape[1]
    first_stride = (1,1)
    if increase_dim:
        out_num_filters = input_num_filters*2
    else:
        out_num_filters = input_num_filters

    stack_1 = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
    stack_2 = batch_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

    # add shortcut connections
    if increase_dim:
        if projection:
            # projection shortcut, as option B in paper
            projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(1,1), nonlinearity=None, pad='same', b=None, flip_filters=False))
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]),nonlinearity=rectify)
        else:
            # identity shortcut, as option A in paper
            identity = ExpressionLayer(l, lambda X: X[:, :, ::1, ::1], lambda s: (s[0], s[1], s[2]//1, s[3]//1))
            padding = PadLayer(identity, [out_num_filters//2,0,0], batch_ndim=1)
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]),nonlinearity=rectify)
    else:
        block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=rectify)

    return block

def residual_block_noPool_noBN(l, increase_dim=False, projection=True):
    input_num_filters = l.output_shape[1]
    first_stride = (1,1)
    if increase_dim:
        out_num_filters = input_num_filters*2
    else:
        out_num_filters = input_num_filters

    stack_1 = ConvLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)
    stack_2 = ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)

    # add shortcut connections
    if increase_dim:
        if projection:
            # projection shortcut, as option B in paper
            projection = ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(1,1), nonlinearity=None, pad='same', b=None, flip_filters=False)
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]),nonlinearity=rectify)
        else:
            # identity shortcut, as option A in paper
            identity = ExpressionLayer(l, lambda X: X[:, :, ::1, ::1], lambda s: (s[0], s[1], s[2]//1, s[3]//1))
            padding = PadLayer(identity, [out_num_filters//2,0,0], batch_ndim=1)
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]),nonlinearity=rectify)
    else:
        block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=rectify)

    return block
