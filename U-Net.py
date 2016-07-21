__author__ = 'fabian'
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, Pool2DLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Deconv2DLayer, ConcatLayer

import matplotlib.pyplot as plt

import skimage.transform
import skimage
import sklearn.cross_validation
import pickle
import os

def build_UNet():
    net = {}
    net['input'] = InputLayer((None, 1, 572, 572))

    net['contr_1_1'] = ConvLayer(net['input'], 64, 3)
    net['contr_1_2'] = ConvLayer(net['contr_1_1'], 64, 3)
    net['pool1'] = Pool2DLayer(net['contr_1_2'], 2)

    net['contr_2_1'] = ConvLayer(net['pool1'], 128, 3)
    net['contr_2_2'] = ConvLayer(net['contr_2_1'], 128, 3)
    net['pool2'] = Pool2DLayer(net['contr_2_2'], 2)

    net['contr_3_1'] = ConvLayer(net['pool2'], 256, 3)
    net['contr_3_2'] = ConvLayer(net['contr_3_1'], 256, 3)
    net['pool3'] = Pool2DLayer(net['contr_3_2'], 2)

    net['contr_4_1'] = ConvLayer(net['pool3'], 512, 3)
    net['contr_4_2'] = ConvLayer(net['contr_4_1'], 512, 3)
    net['pool4'] = Pool2DLayer(net['contr_4_2'], 2)

    net['contr_5_1'] = ConvLayer(net['pool4'], 1024, 3)
    net['contr_5_2'] = ConvLayer(net['contr_5_1'], 1024, 3)
    net['deconv1'] = Deconv2DLayer(net['contr_5_2'], 512, 2)

    net['concat1'] = ConcatLayer([net['deconv1'], net['contr_4_2']], cropping='center')
    net['expand_1_1'] = ConvLayer(net['concat1'], 512, 3)
    net['expand_1_2'] = ConvLayer(net['expand_1_1'], 512, 3)
    net['deconv2'] = Deconv2DLayer(net['expand_1_2'], 256, 2)

    net['concat2'] = ConcatLayer([net['deconv2'], net['contr_3_2']], cropping='center')
    net['expand_2_1'] = ConvLayer(net['concat2'], 256, 3)
    net['expand_2_2'] = ConvLayer(net['expand_2_1'], 256, 3)
    net['deconv3'] = Deconv2DLayer(net['expand_2_2'], 128, 2)

    net['concat3'] = ConcatLayer([net['deconv3'], net['contr_2_2']], cropping='center')
    net['expand_3_1'] = ConvLayer(net['concat3'], 128, 3)
    net['expand_3_2'] = ConvLayer(net['expand_3_1'], 128, 3)
    net['deconv4'] = Deconv2DLayer(net['expand_3_2'], 64, 2)

    net['concat4'] = ConcatLayer([net['deconv4'], net['contr_1_2']], cropping='center')
    net['expand_4_1'] = ConvLayer(net['concat4'], 64, 3)
    net['expand_4_2'] = ConvLayer(net['expand_4_1'], 64, 3)

    net['segLayer'] = ConvLayer(net['expand_4_2'], 2, 1, nonlinearity=lasagne.nonlinearities.softmax)


# loss is crossentropy for each pixel

# loss is multiplied by weight map