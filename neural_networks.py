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

def build_UNet_old(n_input_channels=1, BATCH_SIZE=None, num_output_classes=2):
    net = OrderedDict()
    net['input'] = InputLayer((BATCH_SIZE, n_input_channels, 128, 128))

    net['contr_1_1'] = batch_norm(ConvLayer(net['input'], 64, 3, nonlinearity=lasagne.nonlinearities.elu, pad='same'))
    net['contr_1_2'] = batch_norm(ConvLayer(net['contr_1_1'], 64, 3, nonlinearity=lasagne.nonlinearities.elu, pad='same'))
    net['pool1'] = Pool2DLayer(net['contr_1_2'], 2)

    net['contr_2_1'] = batch_norm(ConvLayer(net['pool1'], 128, 3, nonlinearity=lasagne.nonlinearities.elu, pad='same'))
    net['contr_2_2'] = batch_norm(ConvLayer(net['contr_2_1'], 128, 3, nonlinearity=lasagne.nonlinearities.elu, pad='same'))
    net['pool2'] = Pool2DLayer(net['contr_2_2'], 2)

    net['contr_3_1'] = batch_norm(ConvLayer(net['pool2'], 256, 3, nonlinearity=lasagne.nonlinearities.elu, pad='same'))
    net['contr_3_2'] = batch_norm(ConvLayer(net['contr_3_1'], 256, 3, nonlinearity=lasagne.nonlinearities.elu, pad='same'))
    net['pool3'] = Pool2DLayer(net['contr_3_2'], 2)

    net['contr_4_1'] = batch_norm(ConvLayer(net['pool3'], 512, 3, nonlinearity=lasagne.nonlinearities.elu, pad='same'))
    net['contr_4_2'] = batch_norm(ConvLayer(net['contr_4_1'], 512, 3, nonlinearity=lasagne.nonlinearities.elu, pad='same'))
    net['pool4'] = Pool2DLayer(net['contr_4_2'], 2)

    net['contr_5_1'] = batch_norm(ConvLayer(net['pool4'], 1024, 3, nonlinearity=lasagne.nonlinearities.elu, pad='same'))
    net['contr_5_2'] = batch_norm(ConvLayer(net['contr_5_1'], 1024, 3, nonlinearity=lasagne.nonlinearities.elu, pad='same'))
    net['deconv1'] = Deconv2DLayer(net['contr_5_2'], 512, 2, 2)

    net['concat1'] = ConcatLayer([net['deconv1'], net['contr_4_2']], cropping=(None, None, "center", "center"))
    net['expand_1_1'] = batch_norm(ConvLayer(net['concat1'], 512, 3, nonlinearity=lasagne.nonlinearities.elu, pad='same'))
    net['expand_1_2'] = batch_norm(ConvLayer(net['expand_1_1'], 512, 3, nonlinearity=lasagne.nonlinearities.elu, pad='same'))
    net['deconv2'] = Deconv2DLayer(net['expand_1_2'], 256, 2, 2)

    net['concat2'] = ConcatLayer([net['deconv2'], net['contr_3_2']], cropping=(None, None, "center", "center"))
    net['expand_2_1'] = batch_norm(ConvLayer(net['concat2'], 256, 3, nonlinearity=lasagne.nonlinearities.elu, pad='same'))
    net['expand_2_2'] = batch_norm(ConvLayer(net['expand_2_1'], 256, 3, nonlinearity=lasagne.nonlinearities.elu, pad='same'))
    net['deconv3'] = Deconv2DLayer(net['expand_2_2'], 128, 2, 2)

    net['concat3'] = ConcatLayer([net['deconv3'], net['contr_2_2']], cropping=(None, None, "center", "center"))
    net['expand_3_1'] = batch_norm(ConvLayer(net['concat3'], 128, 3, nonlinearity=lasagne.nonlinearities.elu, pad='same'))
    net['expand_3_2'] = batch_norm(ConvLayer(net['expand_3_1'], 128, 3, nonlinearity=lasagne.nonlinearities.elu, pad='same'))
    net['deconv4'] = Deconv2DLayer(net['expand_3_2'], 64, 2, 2)

    net['concat4'] = ConcatLayer([net['deconv4'], net['contr_1_2']], cropping=(None, None, "center", "center"))
    net['expand_4_1'] = batch_norm(ConvLayer(net['concat4'], 64, 3, nonlinearity=lasagne.nonlinearities.elu, pad='same'))
    net['expand_4_2'] = batch_norm(ConvLayer(net['expand_4_1'], 64, 3, nonlinearity=lasagne.nonlinearities.elu, pad='same'))

    net['segLayer'] = ConvLayer(net['expand_4_2'], num_output_classes, 1, nonlinearity=softmax_fcn)
    net['dimshuffle'] = DimshuffleLayer(net['segLayer'], (1, 0, 2, 3))
    net['reshapeSeg'] = ReshapeLayer(net['dimshuffle'], (num_output_classes, -1))
    net['dimshuffle2'] = DimshuffleLayer(net['reshapeSeg'], (1, 0))
    net['output'] = NonlinearityLayer(net['dimshuffle2'], nonlinearity=lasagne.nonlinearities.softmax)
    return net

def build_UNet(n_input_channels=1, BATCH_SIZE=None, num_output_classes=2, pad='same', nonlinearity=lasagne.nonlinearities.elu, input_dim=128, base_n_filters=64, do_dropout=False):
    net = OrderedDict()
    net['input'] = InputLayer((BATCH_SIZE, n_input_channels, input_dim, input_dim))

    net['contr_1_1'] = batch_norm(ConvLayer(net['input'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad))
    if do_dropout:
        net['contr_1_1'] = DropoutLayer(net['contr_1_1'], 0.1)
    net['contr_1_2'] = batch_norm(ConvLayer(net['contr_1_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad))
    if do_dropout:
        net['contr_1_2'] = DropoutLayer(net['contr_1_2'], 0.1)
    net['pool1'] = Pool2DLayer(net['contr_1_2'], 2)

    net['contr_2_1'] = batch_norm(ConvLayer(net['pool1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad))
    if do_dropout:
        net['contr_2_1'] = DropoutLayer(net['contr_2_1'], 0.2)
    net['contr_2_2'] = batch_norm(ConvLayer(net['contr_2_1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad))
    if do_dropout:
        net['contr_2_2'] = DropoutLayer(net['contr_2_2'], 0.2)
    net['pool2'] = Pool2DLayer(net['contr_2_2'], 2)

    net['contr_3_1'] = batch_norm(ConvLayer(net['pool2'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad))
    if do_dropout:
        net['contr_3_1'] = DropoutLayer(net['contr_3_1'], 0.3)
    net['contr_3_2'] = batch_norm(ConvLayer(net['contr_3_1'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad))
    if do_dropout:
        net['contr_3_2'] = DropoutLayer(net['contr_3_2'], 0.3)
    net['pool3'] = Pool2DLayer(net['contr_3_2'], 2)

    net['contr_4_1'] = batch_norm(ConvLayer(net['pool3'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad))
    if do_dropout:
        net['contr_4_1'] = DropoutLayer(net['contr_4_1'], 0.4)
    net['contr_4_2'] = batch_norm(ConvLayer(net['contr_4_1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad))
    if do_dropout:
        net['contr_4_2'] = DropoutLayer(net['contr_4_2'], 0.4)
    net['pool4'] = Pool2DLayer(net['contr_4_2'], 2)

    net['contr_5_1'] = batch_norm(ConvLayer(net['pool4'], base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad))
    if do_dropout:
        net['contr_5_1'] = DropoutLayer(net['contr_5_1'], 0.5)
    net['contr_5_2'] = batch_norm(ConvLayer(net['contr_5_1'], base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad))
    if do_dropout:
        net['contr_5_2'] = DropoutLayer(net['contr_5_2'], 0.5)
    net['deconv1'] = Deconv2DLayer(net['contr_5_2'], base_n_filters*8, 2, 2)

    net['concat1'] = ConcatLayer([net['deconv1'], net['contr_4_2']], cropping=(None, None, "center", "center"))
    net['expand_1_1'] = batch_norm(ConvLayer(net['concat1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad))
    if do_dropout:
        net['expand_1_1'] = DropoutLayer(net['expand_1_1'], 0.4)
    net['expand_1_2'] = batch_norm(ConvLayer(net['expand_1_1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad))
    if do_dropout:
        net['expand_1_2'] = DropoutLayer(net['expand_1_2'], 0.4)
    net['deconv2'] = Deconv2DLayer(net['expand_1_2'], base_n_filters*4, 2, 2)

    net['concat2'] = ConcatLayer([net['deconv2'], net['contr_3_2']], cropping=(None, None, "center", "center"))
    net['expand_2_1'] = batch_norm(ConvLayer(net['concat2'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad))
    if do_dropout:
        net['expand_2_1'] = DropoutLayer(net['expand_2_1'], 0.3)
    net['expand_2_2'] = batch_norm(ConvLayer(net['expand_2_1'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad))
    if do_dropout:
        net['expand_2_2'] = DropoutLayer(net['expand_2_2'], 0.3)
    net['deconv3'] = Deconv2DLayer(net['expand_2_2'], base_n_filters*2, 2, 2)

    net['concat3'] = ConcatLayer([net['deconv3'], net['contr_2_2']], cropping=(None, None, "center", "center"))
    net['expand_3_1'] = batch_norm(ConvLayer(net['concat3'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad))
    if do_dropout:
        net['expand_3_1'] = DropoutLayer(net['expand_3_1'], 0.2)
    net['expand_3_2'] = batch_norm(ConvLayer(net['expand_3_1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad))
    if do_dropout:
        net['expand_3_2'] = DropoutLayer(net['expand_3_2'], 0.2)
    net['deconv4'] = Deconv2DLayer(net['expand_3_2'], base_n_filters, 2, 2)

    net['concat4'] = ConcatLayer([net['deconv4'], net['contr_1_2']], cropping=(None, None, "center", "center"))
    net['expand_4_1'] = batch_norm(ConvLayer(net['concat4'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad))
    if do_dropout:
        net['expand_4_1'] = DropoutLayer(net['expand_4_1'], 0.1)
    net['expand_4_2'] = batch_norm(ConvLayer(net['expand_4_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad))
    if do_dropout:
        net['expand_4_2'] = DropoutLayer(net['expand_4_2'], 0.1)

    net['segLayer'] = ConvLayer(net['expand_4_2'], num_output_classes, 1, nonlinearity=None)
    net['dimshuffle'] = DimshuffleLayer(net['segLayer'], (1, 0, 2, 3))
    net['reshapeSeg'] = ReshapeLayer(net['dimshuffle'], (num_output_classes, -1))
    net['dimshuffle2'] = DimshuffleLayer(net['reshapeSeg'], (1, 0))
    net['output'] = NonlinearityLayer(net['dimshuffle2'], nonlinearity=lasagne.nonlinearities.softmax)
    return net

def build_residual_UNet(n_input_channels=1, BATCH_SIZE=None, num_output_classes=2, input_dim=128, base_n_filters=16):
    net = OrderedDict()
    net['input'] = InputLayer((BATCH_SIZE, n_input_channels, input_dim, input_dim))
    net['conv_1'] = batch_norm(ConvLayer(net['input'], num_filters=base_n_filters, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

    # = residual, increase dim, projection true
    net['contr_block_1'] = residual_block_noPool(net['input'], True, True)
    net['contr_block_1_pool'] = Pool2DLayer(net['contr_block_1'], 2)

    # = residual, increase dim, projection true
    net['contr_block_2'] = residual_block_noPool(net['contr_block_1_pool'], True, True)
    net['contr_block_2_pool'] = Pool2DLayer(net['contr_block_2'], 2)

    # = residual, increase dim, projection true
    net['contr_block_3'] = residual_block_noPool(net['contr_block_2_pool'], True, True)
    net['contr_block_3_pool'] = Pool2DLayer(net['contr_block_3'], 2)

    # = residual, increase dim, projection true
    net['contr_block_4'] = residual_block_noPool(net['contr_block_3_pool'], True, True)
    net['contr_block_4_pool'] = Pool2DLayer(net['contr_block_4'], 2)

    # = residual, increase false, projection true
    net['enc'] = residual_block_noPool(net['contr_block_4_pool'], True, True)

    net['deconv1'] = Deconv2DLayer(net['enc'], base_n_filters*8, 2, 2)
    net['concat1'] = ConcatLayer([net['deconv1'], net['contr_block_4']], cropping=(None, None, "center", "center"))
    # = residual, increase false, projection true
    net['expand_1'] = residual_block_noPool(net['concat1'], False, True)

    net['deconv2'] = Deconv2DLayer(net['expand_1'], base_n_filters*4, 2, 2)
    net['concat2'] = ConcatLayer([net['deconv2'], net['contr_block_3']], cropping=(None, None, "center", "center"))
    # = residual, increase false, projection true
    net['expand_2'] = residual_block_noPool(net['concat2'], False, True)

    net['deconv3'] = Deconv2DLayer(net['expand_2'], base_n_filters*2, 2, 2)
    net['concat3'] = ConcatLayer([net['deconv3'], net['contr_block_2']], cropping=(None, None, "center", "center"))
    # = residual, increase false, projection true
    net['expand_3'] = residual_block_noPool(net['concat3'], False, True)

    net['deconv4'] = Deconv2DLayer(net['expand_3'], base_n_filters, 2, 2)
    net['concat4'] = ConcatLayer([net['deconv4'], net['contr_block_1']], cropping=(None, None, "center", "center"))
    # = residual, increase false, projection true
    net['expand_4'] = residual_block_noPool(net['concat4'], False, True)

    net['segLayer'] = ConvLayer(net['expand_4'], num_output_classes, 1, nonlinearity=None)
    net['dimshuffle'] = DimshuffleLayer(net['segLayer'], (1, 0, 2, 3))
    net['reshapeSeg'] = ReshapeLayer(net['dimshuffle'], (num_output_classes, -1))
    net['dimshuffle2'] = DimshuffleLayer(net['reshapeSeg'], (1, 0))
    net['output'] = NonlinearityLayer(net['dimshuffle2'], nonlinearity=lasagne.nonlinearities.softmax)
    return net

def build_deep_residual_UNet(n_input_channels=1, BATCH_SIZE=None, num_output_classes=2, input_dim=128, base_n_filters=16, n_res_blocks=1):
    net = OrderedDict()
    l = net['input'] = InputLayer((BATCH_SIZE, n_input_channels, input_dim, input_dim))
    l = net['conv_1'] = batch_norm(ConvLayer(l, num_filters=base_n_filters, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

    # = residual, increase dim, projection true
    l = net['contr_block_1_1'] = residual_block_noPool(l, True, True)
    for i in xrange(n_res_blocks):
        l = net['contr_block_1_%d'%(i+2)] = residual_block_noPool(l, False, True)
    l = net['contr_block_1_pool'] = Pool2DLayer(l, 2)

    # = residual, increase dim, projection true
    l = net['contr_block_2_1'] = residual_block_noPool(l, True, True)
    for i in xrange(n_res_blocks):
        l = net['contr_block_2_%d'%(i+2)] = residual_block_noPool(l, False, True)
    l = net['contr_block_2_pool'] = Pool2DLayer(l, 2)

    # = residual, increase dim, projection true
    l = net['contr_block_3_1'] = residual_block_noPool(l, True, True)
    for i in xrange(n_res_blocks):
        l = net['contr_block_3_%d'%(i+2)] = residual_block_noPool(l, False, True)
    l = net['contr_block_3_pool'] = Pool2DLayer(l, 2)

    # = residual, increase dim, projection true
    l = net['contr_block_4_1'] = residual_block_noPool(l, True, True)
    for i in xrange(n_res_blocks):
        l = net['contr_block_4_%d'%(i+2)] = residual_block_noPool(l, False, True)
    l = net['contr_block_4_pool'] = Pool2DLayer(l, 2)

    # = residual, increase false, projection true
    l = net['enc_1'] = residual_block_noPool(l, True, True)
    for i in xrange(n_res_blocks):
        l = net['enc_%d'%(i+2)] = residual_block_noPool(l, False, True)

    l = net['deconv1'] = Deconv2DLayer(l, base_n_filters*8, 2, 2)
    l = net['concat1'] = ConcatLayer([l, net['contr_block_4_%d'%(n_res_blocks+1)]], cropping=(None, None, "center", "center"))
    # = residual, increase false, projection true
    l = net['expand_1_1'] = residual_block_noPool(l, False, True)
    for i in xrange(n_res_blocks):
        l = net['expand_1_%d'%(i+2)] = residual_block_noPool(l, False, True)

    l = net['deconv2'] = Deconv2DLayer(l, base_n_filters*4, 2, 2)
    l = net['concat2'] = ConcatLayer([l, net['contr_block_3_%d'%(n_res_blocks+1)]], cropping=(None, None, "center", "center"))
    # = residual, increase false, projection true
    l = net['expand_2_1'] = residual_block_noPool(l, False, True)
    for i in xrange(n_res_blocks):
        l = net['expand_2_%d'%(i+2)] = residual_block_noPool(l, False, True)

    l = net['deconv3'] = Deconv2DLayer(l, base_n_filters*2, 2, 2)
    l = net['concat3'] = ConcatLayer([l, net['contr_block_2_%d'%(n_res_blocks+1)]], cropping=(None, None, "center", "center"))
    # = residual, increase false, projection true
    l = net['expand_3_1'] = residual_block_noPool(l, False, True)
    for i in xrange(n_res_blocks):
        l = net['expand_3_%d'%(i+2)] = residual_block_noPool(l, False, True)

    l = net['deconv4'] = Deconv2DLayer(l, base_n_filters, 2, 2)
    l = net['concat4'] = ConcatLayer([l, net['contr_block_1_%d'%(n_res_blocks+1)]], cropping=(None, None, "center", "center"))
    # = residual, increase false, projection true
    l = net['expand_4_1'] = residual_block_noPool(l, False, True)
    for i in xrange(n_res_blocks):
        l = net['expand_4_%d'%(i+2)] = residual_block_noPool(l, False, True)

    l = net['segLayer'] = ConvLayer(l, num_output_classes, 1, nonlinearity=None)
    l = net['dimshuffle'] = DimshuffleLayer(l, (1, 0, 2, 3))
    l = net['reshapeSeg'] = ReshapeLayer(l, (num_output_classes, -1))
    l = net['dimshuffle2'] = DimshuffleLayer(l, (1, 0))
    net['output'] = NonlinearityLayer(l, nonlinearity=lasagne.nonlinearities.softmax)
    return net


def softmax_fcn(x):
    e_x = theano.tensor.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def categorical_crossentropy_fcn(predicted, target):
    # predicted shape = (BATCH_SIZE, N_CLASSES, X, Y)
    # target shape = (BATCH_SIZE, N_CLASSES, X, Y)
    return -target * T.log(predicted)

def convert_seg_map_for_crossentropy(seg_map, all_classes):
    # we assume that the int's in the seg map are continuous (classes (0, 1, 2, 3, 4) and NOT something like (0, 12, 54, 13))
    # we assume that the seg map has a shape of (BATCH_SIZE, 1, X, Y) and that it stores the int for the correct class
    # we need all_classes because we cannot be sure that every class is represented in every segmentation map
    seg_map_shape = list(seg_map.shape)
    seg_map_shape[1] = len(all_classes)
    new_seg_map = np.zeros(tuple(seg_map_shape), dtype=seg_map.dtype)
    for i, j in enumerate(all_classes):
        new_seg_map[:, i, :, :][seg_map[:, 0, :, :] == j] = 1
    return new_seg_map