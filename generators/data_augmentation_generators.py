__author__ = 'fabian'

import numpy as np
from scipy.ndimage import interpolation
from scipy.ndimage import map_coordinates
from generator_utils import generate_elastic_transform_coordinates
import lasagne

def rotation_generator(generator, angle_range=(-180, 180)):
    '''
    yields rotated data and seg (rotated around center with a uniformly distributed angle between angle_range[0] and angle_range[1])
    '''
    for data, seg, labels in generator:
        seg_min = np.min(seg)
        seg_max = np.max(seg)
        angle = np.random.uniform(angle_range[0], angle_range[1])
        data = interpolation.rotate(data, angle, (2, 3), reshape=False, mode='nearest')
        seg = np.round(interpolation.rotate(seg, angle, (2, 3), reshape=False)).astype(np.int32)
        seg[seg > seg_max] = seg_max
        seg[seg < seg_min] = seg_min
        yield data, seg, labels

def center_crop_generator(generator, output_size):
    '''
    yields center crop of size output_size (may be 1d or 2d) from data and seg
    '''
    center_crop = lasagne.utils.as_tuple(output_size, 2, int)
    for data, seg, labels in generator:
        center = np.array(data.shape[2:])/2
        yield data[:, :, int(center[0]-center_crop[0]/2.):int(center[0]+center_crop[0]/2.), int(center[1]-center_crop[1]/2.):int(center[1]+center_crop[1]/2.)], seg[:, :, int(center[0]-center_crop[0]/2.):int(center[0]+center_crop[0]/2.), int(center[1]-center_crop[1]/2.):int(center[1]+center_crop[1]/2.)], labels

def mirror_axis_generator(generator):
    '''
    yields mirrored data and seg: 25% x axis, 25% y axis, 25% at both axes and 25% unchanged
    '''
    for data, seg, labels in generator:
        BATCH_SIZE = data.shape[0]
        data[:int(BATCH_SIZE/2)] = data[:int(BATCH_SIZE/2)][:, :, ::-1, :]
        data[int(BATCH_SIZE/4):int(BATCH_SIZE*3/4)] = data[int(BATCH_SIZE/4):int(BATCH_SIZE*3/4)][:, :, :, ::-1]
        seg[:int(BATCH_SIZE/2)] = seg[:int(BATCH_SIZE/2)][:, :, ::-1, :]
        seg[int(BATCH_SIZE/4):int(BATCH_SIZE*3/4)] = seg[int(BATCH_SIZE/4):int(BATCH_SIZE*3/4)][:, :, :, ::-1]
        yield data, seg, labels

def random_crop_generator(generator, crop_size=(128, 128)):
    '''
    yields a random crop of size crop_size
    '''
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size, crop_size]
    elif len(crop_size) == 2:
        crop_size = list(crop_size)
    else:
        raise ValueError("invalid crop_size")
    for data, seg, labels in generator:
        lb_x = np.random.randint(0, data.shape[2]-crop_size[0])
        lb_y = np.random.randint(0, data.shape[3]-crop_size[1])
        data = data[:, :, lb_x:lb_x+crop_size[0], lb_y:lb_y+crop_size[1]]
        seg = seg[:, :, lb_x:lb_x+crop_size[0], lb_y:lb_y+crop_size[1]]
        yield data, seg, labels

def elastric_transform_generator(generator, alpha=100, sigma=10):
    '''
    yields elastically transformed data and seg
    :param alpha: magnitude of deformation
    :param sigma: smoothness of deformation
    '''
    for data, seg, labels in generator:
        seg_min = np.min(seg)
        seg_max = np.max(seg)
        data_shape = tuple(list(data.shape)[2:])
        coords = generate_elastic_transform_coordinates(data_shape, alpha, sigma)
        for data_idx in xrange(data.shape[0]):
            for channel_idx in xrange(data.shape[1]):
                data[data_idx, channel_idx] = map_coordinates(data[data_idx, channel_idx], coords, order=3, mode="nearest").reshape(data_shape)
            for seg_channel_idx in xrange(seg.shape[1]):
                seg[data_idx, seg_channel_idx] = map_coordinates(seg[data_idx, seg_channel_idx], coords, order=3, mode="nearest").reshape(data_shape)
        seg = np.round(seg).astype(np.int32)
        seg[seg > seg_max] = seg_max
        seg[seg < seg_min] = seg_min
        yield data, seg, labels

def data_channel_selection_generator(generator, selected_channels):
    '''
    yields selected channels from data
    '''
    for data, seg, labels in generator:
        yield data[:, selected_channels, :, :], seg, labels

def seg_channel_selection_generator(generator, selected_channels):
    '''
    yields selected channels from seg
    '''
    for data, seg, labels in generator:
        yield data, seg[:, selected_channels, :, :], labels

