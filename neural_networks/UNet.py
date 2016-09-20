__author__ = 'Fabian Isensee'
from collections import OrderedDict
from lasagne.layers import InputLayer, ConcatLayer, Pool2DLayer, ReshapeLayer, DimshuffleLayer, NonlinearityLayer, DropoutLayer, Upscale2DLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
import lasagne
from subnets import *


def build_UNet(n_input_channels=1, BATCH_SIZE=None, num_output_classes=2, pad='same', nonlinearity=lasagne.nonlinearities.elu, input_dim=(128, 128), base_n_filters=64, do_dropout=False):
    net = OrderedDict()
    net['input'] = InputLayer((BATCH_SIZE, n_input_channels, input_dim[0], input_dim[1]))

    net['contr_1_1'] = ConvLayer(net['input'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad)
    net['contr_1_2'] = ConvLayer(net['contr_1_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad)
    net['pool1'] = Pool2DLayer(net['contr_1_2'], 2)

    net['contr_2_1'] = ConvLayer(net['pool1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad)
    net['contr_2_2'] = ConvLayer(net['contr_2_1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad)
    net['pool2'] = Pool2DLayer(net['contr_2_2'], 2)

    net['contr_3_1'] = ConvLayer(net['pool2'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad)
    net['contr_3_2'] = ConvLayer(net['contr_3_1'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad)
    net['pool3'] = Pool2DLayer(net['contr_3_2'], 2)

    net['contr_4_1'] = ConvLayer(net['pool3'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad)
    net['contr_4_2'] = ConvLayer(net['contr_4_1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad)
    l = net['pool4'] = Pool2DLayer(net['contr_4_2'], 2)
    # the paper does not really describe where and how dropout is added. Feel free to try more options
    if do_dropout:
        l = DropoutLayer(l, p=0.4)

    net['encode_1'] = ConvLayer(l, base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad)
    net['encode_2'] = ConvLayer(net['encode_1'], base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad)
    net['deconv1'] = Upscale2DLayer(net['encode_2'], 2)

    net['concat1'] = ConcatLayer([net['deconv1'], net['contr_4_2']], cropping=(None, None, "center", "center"))
    net['expand_1_1'] = ConvLayer(net['concat1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad)
    net['expand_1_2'] = ConvLayer(net['expand_1_1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad)
    net['deconv2'] = Upscale2DLayer(net['expand_1_2'], 2)

    net['concat2'] = ConcatLayer([net['deconv2'], net['contr_3_2']], cropping=(None, None, "center", "center"))
    net['expand_2_1'] = ConvLayer(net['concat2'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad)
    net['expand_2_2'] = ConvLayer(net['expand_2_1'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad)
    net['deconv3'] = Upscale2DLayer(net['expand_2_2'], 2)

    net['concat3'] = ConcatLayer([net['deconv3'], net['contr_2_2']], cropping=(None, None, "center", "center"))
    net['expand_3_1'] = ConvLayer(net['concat3'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad)
    net['expand_3_2'] = ConvLayer(net['expand_3_1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad)
    net['deconv4'] = Upscale2DLayer(net['expand_3_2'], 2)

    net['concat4'] = ConcatLayer([net['deconv4'], net['contr_1_2']], cropping=(None, None, "center", "center"))
    net['expand_4_1'] = ConvLayer(net['concat4'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad)
    net['expand_4_2'] = ConvLayer(net['expand_4_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad)

    net['output_segmentation'] = ConvLayer(net['expand_4_2'], num_output_classes, 1, nonlinearity=None)
    net['dimshuffle'] = DimshuffleLayer(net['output_segmentation'], (1, 0, 2, 3))
    net['reshapeSeg'] = ReshapeLayer(net['dimshuffle'], (num_output_classes, -1))
    net['dimshuffle2'] = DimshuffleLayer(net['reshapeSeg'], (1, 0))
    net['output_flattened'] = NonlinearityLayer(net['dimshuffle2'], nonlinearity=lasagne.nonlinearities.softmax)

    return net



def build_deep_residual_UNet(n_input_channels=1, BATCH_SIZE=None, num_output_classes=2, input_dim=(128, 128), base_n_filters=16, n_res_blocks=1, doBN=False):
    if doBN:
        residualBlock = residual_block_noPool
    else:
        residualBlock = residual_block_noPool_noBN

    net = OrderedDict()
    l = net['input'] = InputLayer((BATCH_SIZE, n_input_channels, input_dim[0], input_dim[1]))
    l = residualBlock(l, False, True, base_n_filters)

    # = residual, increase dim, projection true
    l = net['contr_block_1_1'] = residualBlock(l, False, True)
    for i in xrange(n_res_blocks):
        l = net['contr_block_1_%d'%(i+2)] = residualBlock(l, False, True)
    l = net['contr_block_1_pool'] = Pool2DLayer(l, 2)

    # = residual, increase dim, projection true
    l = net['contr_block_2_1'] = residualBlock(l, True, True)
    for i in xrange(n_res_blocks):
        l = net['contr_block_2_%d'%(i+2)] = residualBlock(l, False, True)
    l = net['contr_block_2_pool'] = Pool2DLayer(l, 2)

    # = residual, increase dim, projection true
    l = net['contr_block_3_1'] = residualBlock(l, True, True)
    for i in xrange(n_res_blocks):
        l = net['contr_block_3_%d'%(i+2)] = residualBlock(l, False, True)
    l = net['contr_block_3_pool'] = Pool2DLayer(l, 2)

    # = residual, increase dim, projection true
    l = net['contr_block_4_1'] = residualBlock(l, True, True)
    for i in xrange(n_res_blocks):
        l = net['contr_block_4_%d'%(i+2)] = residualBlock(l, False, True)
    l = net['contr_block_4_pool'] = Pool2DLayer(l, 2)

    # = residual, increase false, projection true
    l = net['enc_1'] = residualBlock(l, True, True)
    for i in xrange(n_res_blocks):
        l = net['enc_%d'%(i+2)] = residualBlock(l, False, True)
    l = net['upscale1'] = Upscale2DLayer(l, 2)

    l = net['concat1'] = ConcatLayer([l, net['contr_block_4_%d'%(n_res_blocks+1)]], cropping=(None, None, "center", "center"))
    # = residual, increase false, projection true
    l = net['expand_1_1'] = residualBlock(l, False, True)
    for i in xrange(n_res_blocks):
        l = net['expand_1_%d'%(i+2)] = residualBlock(l, False, True)
    l = net['upscale2'] = Upscale2DLayer(l, 2)

    l = net['concat2'] = ConcatLayer([l, net['contr_block_3_%d'%(n_res_blocks+1)]], cropping=(None, None, "center", "center"))
    # = residual, increase false, projection true
    l = net['expand_2_1'] = residualBlock(l, False, True)
    for i in xrange(n_res_blocks):
        l = net['expand_2_%d'%(i+2)] = residualBlock(l, False, True)
    l = net['upscale3'] = Upscale2DLayer(l, 2)

    l = net['concat3'] = ConcatLayer([l, net['contr_block_2_%d'%(n_res_blocks+1)]], cropping=(None, None, "center", "center"))
    # = residual, increase false, projection true
    l = net['expand_3_1'] = residualBlock(l, False, True)
    for i in xrange(n_res_blocks):
        l = net['expand_3_%d'%(i+2)] = residualBlock(l, False, True)
    l = net['upscale4'] = Upscale2DLayer(l, 2)

    l = net['concat4'] = ConcatLayer([l, net['contr_block_1_%d'%(n_res_blocks+1)]], cropping=(None, None, "center", "center"))
    # = residual, increase false, projection true
    l = net['expand_4_1'] = residualBlock(l, False, True)
    for i in xrange(n_res_blocks):
        l = net['expand_4_%d'%(i+2)] = residualBlock(l, False, True)

    l = net['output_segmentation'] = ConvLayer(l, num_output_classes, 1, nonlinearity=None)
    l = net['dimshuffle'] = DimshuffleLayer(l, (1, 0, 2, 3))
    l = net['reshapeSeg'] = ReshapeLayer(l, (num_output_classes, -1))
    l = net['dimshuffle2'] = DimshuffleLayer(l, (1, 0))
    net['output_flattened'] = NonlinearityLayer(l, nonlinearity=lasagne.nonlinearities.softmax)
    return net
