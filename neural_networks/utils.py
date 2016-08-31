__author__ = 'fabian'
import numpy as np
import theano

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