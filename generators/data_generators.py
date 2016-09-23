__author__ = 'fabian'
import numpy as np
import IPython
import os.path as path
from numpy import memmap
import sys
sys.path.append("../utils")
from general_utils import find_entries_in_array
from copy import deepcopy
from scipy.ndimage import interpolation
from scipy.ndimage import map_coordinates

def memmapGenerator(array_neg, array_pos, BATCH_SIZE, n_elements_pos=None, n_elements_neg=None):
    np.random.seed()
    while True:
        if n_elements_pos is None:
            n_elements_pos = array_pos.shape[0]
        if n_elements_neg is None:
            n_elements_neg = array_neg.shape[0]
        idx_pos = np.random.choice(n_elements_pos, int(BATCH_SIZE/2.))
        idx_neg = np.random.choice(n_elements_neg, int(BATCH_SIZE/2.))
        data = np.zeros((len(idx_pos)+len(idx_neg), 1, 128, 128), dtype=np.float32)
        seg = np.zeros((len(idx_pos)+len(idx_neg), 1, 128, 128), dtype=np.int32)
        labels = np.zeros(len(idx_pos)+len(idx_neg), dtype=np.int32)
        idx_for_shuffle = np.arange(len(idx_neg)+len(idx_neg))
        np.random.shuffle(idx_for_shuffle)
        pos_data = np.array(array_pos[idx_pos])
        neg_data = np.array(array_neg[idx_neg])
        data[:len(idx_pos)] = pos_data[:, 0:128**2].reshape(len(idx_pos), 1, 128, 128).astype(np.float32)
        data[len(idx_pos):] = neg_data[:, 0:128**2].reshape(len(idx_neg), 1, 128, 128).astype(np.float32)
        seg[:len(idx_pos)] = pos_data[:, 128**2:128**2*2].reshape(len(idx_pos), 1, 128, 128).astype(np.int32)
        seg[len(idx_pos):] = neg_data[:, 128**2:128**2*2].reshape(len(idx_neg), 1, 128, 128).astype(np.int32)
        labels[:len(idx_pos)] = 1
        labels[len(idx_pos):] = 0
        yield data[idx_for_shuffle], seg[idx_for_shuffle], labels[idx_for_shuffle]

def memmapGeneratorDataAugm(array_neg, array_pos, BATCH_SIZE, n_elements_pos=None, n_elements_neg=None):
    np.random.seed()
    while True:
        if n_elements_pos is None:
            n_elements_pos = array_pos.shape[0]
        if n_elements_neg is None:
            n_elements_neg = array_neg.shape[0]
        idx_pos = np.random.choice(n_elements_pos, int(BATCH_SIZE/2.))
        idx_neg = np.random.choice(n_elements_neg, int(BATCH_SIZE/2.))
        data = np.zeros((len(idx_pos)+len(idx_neg), 1, 128, 128), dtype=np.float32)
        seg = np.zeros((len(idx_pos)+len(idx_neg), 1, 128, 128), dtype=np.int32)
        labels = np.zeros(len(idx_pos)+len(idx_neg), dtype=np.int32)
        idx_for_shuffle = np.arange(len(idx_neg)+len(idx_neg))
        np.random.shuffle(idx_for_shuffle)
        pos_data = np.array(array_pos[idx_pos])
        neg_data = np.array(array_neg[idx_neg])
        data[:len(idx_pos)] = pos_data[:, 0:128**2].reshape(len(idx_pos), 1, 128, 128).astype(np.float32)
        data[len(idx_pos):] = neg_data[:, 0:128**2].reshape(len(idx_neg), 1, 128, 128).astype(np.float32)
        seg[:len(idx_pos)] = pos_data[:, 128**2:128**2*2].reshape(len(idx_pos), 1, 128, 128).astype(np.int32)
        seg[len(idx_pos):] = neg_data[:, 128**2:128**2*2].reshape(len(idx_neg), 1, 128, 128).astype(np.int32)
        labels[:len(idx_pos)] = 1
        labels[len(idx_pos):] = 0
        data = data[idx_for_shuffle]
        seg = seg[idx_for_shuffle]
        labels = labels[idx_for_shuffle]
        # 25% flipped vertically, 25% flipped horizontically, 25% flipped both ways
        data[:int(len(idx_for_shuffle)/2)] = data[:int(len(idx_for_shuffle)/2)][:, :, ::-1, :]
        data[int(len(idx_for_shuffle)/4):int(len(idx_for_shuffle)*3/4)] = data[int(len(idx_for_shuffle)/4):int(len(idx_for_shuffle)*3/4)][:, :, :, ::-1]
        seg[:int(len(idx_for_shuffle)/2)] = seg[:int(len(idx_for_shuffle)/2)][:, :, ::-1, :]
        seg[int(len(idx_for_shuffle)/4):int(len(idx_for_shuffle)*3/4)] = seg[int(len(idx_for_shuffle)/4):int(len(idx_for_shuffle)*3/4)][:, :, :, ::-1]
        yield data, seg, labels

def loadPatientMemmaps(folder = "/home/fabian/datasets/Hirntumor_von_David/experiments/data/memmap_by_patient/"):
    import cPickle
    with open(path.join(folder, "valid_ids.pkl"), 'r') as f:
        valid_ids = cPickle.load(f)
    all_patients_dict = {}
    for id in valid_ids:
        if not path.isfile(path.join(folder, "patient_%s_data.memmap" % id)) or not path.isfile(path.join(folder, "patient_%s_properties.pkl" % id)):
            print "patient %s not found" % id
            continue
        with open(path.join(folder, "patient_%s_properties.pkl" % id), 'r') as f:
            properties = cPickle.load(f)
        data = memmap(path.join(folder, "patient_%s_data.memmap" % id), dtype=np.float32, mode="r+", shape=properties['shape'])
        all_patients_dict[id] = {"properties" : properties, "data" : data}
    return all_patients_dict

# super slow. this is not going to be practical...
def generate_batches_patientMemmaps_dataAugm_t1km_flair(all_patients_dict, BATCH_SIZE=50, PATCH_SIZE=128):
    np.random.seed()
    patient_ids = all_patients_dict.keys()
    while True:
        data = np.zeros((BATCH_SIZE, 2, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        seg = np.zeros((BATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE), dtype=np.int32)
        labels = np.zeros(BATCH_SIZE, dtype=np.int32) + 9
        n_positive = 0
        n_negative = 0
        while n_positive < BATCH_SIZE/2 or n_negative < BATCH_SIZE/2:
            id = np.random.choice(patient_ids)
            shape = all_patients_dict[id]['properties']['shape']
            z_idx = np.random.choice(np.arange(shape[1]))
            x_idx = np.random.choice(np.arange(shape[2]-PATCH_SIZE))
            y_idx = np.random.choice(np.arange(shape[3]-PATCH_SIZE))
            patch_seg = np.array(all_patients_dict[id]['data'][all_patients_dict[id]['properties']['seg_idx']][z_idx, x_idx:x_idx+PATCH_SIZE, y_idx:y_idx+PATCH_SIZE]).astype(np.int32)
            patch_t1km = np.array(all_patients_dict[id]['data'][all_patients_dict[id]['properties']['t1km_idx']][z_idx, x_idx:x_idx+PATCH_SIZE, y_idx:y_idx+PATCH_SIZE])
            # outside_value = patch_t1km[0,0,0]
            # if np.sum(patch_t1km == outside_value) < (0.6 * PATCH_SIZE**2):
            #     continue
            # find out label:
            if np.sum(patch_seg > 1)/float(PATCH_SIZE**2) > 0.15:
                label = 1
            elif np.sum(patch_seg > 1)/float(PATCH_SIZE**2) == 0:
                label = 0
            else:
                continue

            if label == 1:
                if n_positive == BATCH_SIZE/2:
                    continue
                else:
                    n_positive += 1

            if label == 0:
                if n_negative == BATCH_SIZE/2:
                    continue
                else:
                    n_negative += 1

            labels[n_positive + n_negative - 1] = label

            patch_flair = np.array(all_patients_dict[id]['data'][all_patients_dict[id]['properties']['flair_idx']][z_idx, x_idx:x_idx+PATCH_SIZE, y_idx:y_idx+PATCH_SIZE])
            if np.random.sample() < 0.5:
                patch_t1km = patch_t1km[:, ::-1]
                patch_flair = patch_flair[:, ::-1]
                patch_seg = patch_seg[:, ::-1]
                if np.random.sample() < 0.5:
                    patch_t1km = patch_t1km[::-1, :]
                    patch_flair = patch_flair[::-1, :]
                    patch_seg = patch_seg[::-1, :]
            data[n_positive + n_negative - 1, 0] = patch_t1km
            data[n_positive + n_negative - 1, 1] = patch_flair
            seg[n_positive + n_negative - 1, 0] = patch_seg
        yield data, seg, labels


def memmapGenerator_t1km_flair(array_neg, array_pos, BATCH_SIZE, n_elements_pos=None, n_elements_neg=None):
    np.random.seed()
    while True:
        if n_elements_pos is None:
            n_elements_pos = array_pos.shape[0]
        if n_elements_neg is None:
            n_elements_neg = array_neg.shape[0]
        idx_pos = np.random.choice(n_elements_pos, int(BATCH_SIZE/2.))
        idx_neg = np.random.choice(n_elements_neg, int(BATCH_SIZE/2.))
        data = np.zeros((len(idx_pos)+len(idx_neg), 2, 128, 128), dtype=np.float32)
        seg = np.zeros((len(idx_pos)+len(idx_neg), 1, 128, 128), dtype=np.int32)
        labels = np.zeros(len(idx_pos)+len(idx_neg), dtype=np.int32)
        idx_for_shuffle = np.arange(len(idx_neg)+len(idx_neg))
        np.random.shuffle(idx_for_shuffle)
        pos_data = np.array(array_pos[idx_pos])
        neg_data = np.array(array_neg[idx_neg])
        data[:len(idx_pos)] = pos_data[:, :2].astype(np.float32)
        data[len(idx_pos):] = neg_data[:, :2].astype(np.float32)
        seg[:len(idx_pos)][:, 0, :, :] = pos_data[:, 2].astype(np.int32)
        seg[len(idx_pos):][:, 0, :, :] = neg_data[:, 2].astype(np.int32)
        labels[:len(idx_pos)] = 1
        labels[len(idx_pos):] = 0
        yield data[idx_for_shuffle], seg[idx_for_shuffle], labels[idx_for_shuffle]


def memmapGeneratorDataAugm_t1km_flair(array_neg, array_pos, BATCH_SIZE, n_elements_pos=None, n_elements_neg=None):
    np.random.seed()
    while True:
        if n_elements_pos is None:
            n_elements_pos = array_pos.shape[0]
        if n_elements_neg is None:
            n_elements_neg = array_neg.shape[0]
        idx_pos = np.random.choice(n_elements_pos, int(BATCH_SIZE/2.))
        idx_neg = np.random.choice(n_elements_neg, int(BATCH_SIZE/2.))
        data = np.zeros((len(idx_pos)+len(idx_neg), 2, 128, 128), dtype=np.float32)
        seg = np.zeros((len(idx_pos)+len(idx_neg), 1, 128, 128), dtype=np.int32)
        labels = np.zeros(len(idx_pos)+len(idx_neg), dtype=np.int32)
        idx_for_shuffle = np.arange(len(idx_neg)+len(idx_neg))
        np.random.shuffle(idx_for_shuffle)
        pos_data = np.array(array_pos[idx_pos])
        neg_data = np.array(array_neg[idx_neg])
        data[:len(idx_pos)] = pos_data[:, :2].astype(np.float32)
        data[len(idx_pos):] = neg_data[:, :2].astype(np.float32)
        seg[:len(idx_pos)][:, 0, :, :] = pos_data[:, 2].astype(np.int32)
        seg[len(idx_pos):][:, 0, :, :] = neg_data[:, 2].astype(np.int32)
        labels[:len(idx_pos)] = 1
        labels[len(idx_pos):] = 0
        data = data[idx_for_shuffle]
        seg = seg[idx_for_shuffle]
        labels = labels[idx_for_shuffle]
        # 25% flipped vertically, 25% flipped horizontically, 25% flipped both ways
        data[:int(len(idx_for_shuffle)/2)] = data[:int(len(idx_for_shuffle)/2)][:, :, ::-1, :]
        data[int(len(idx_for_shuffle)/4):int(len(idx_for_shuffle)*3/4)] = data[int(len(idx_for_shuffle)/4):int(len(idx_for_shuffle)*3/4)][:, :, :, ::-1]
        seg[:int(len(idx_for_shuffle)/2)] = seg[:int(len(idx_for_shuffle)/2)][:, :, ::-1, :]
        seg[int(len(idx_for_shuffle)/4):int(len(idx_for_shuffle)*3/4)] = seg[int(len(idx_for_shuffle)/4):int(len(idx_for_shuffle)*3/4)][:, :, :, ::-1]
        yield data, seg, labels


def memmapGeneratorDataAugm_t1km_flair_adc_cbv(array_neg, array_pos, BATCH_SIZE, n_elements_pos=None, n_elements_neg=None):
    np.random.seed()
    while True:
        if n_elements_pos is None:
            n_elements_pos = array_pos.shape[0]
        if n_elements_neg is None:
            n_elements_neg = array_neg.shape[0]
        idx_pos = np.random.choice(n_elements_pos, int(BATCH_SIZE/2.))
        idx_neg = np.random.choice(n_elements_neg, int(BATCH_SIZE/2.))
        data = np.zeros((len(idx_pos)+len(idx_neg), 4, 128, 128), dtype=np.float32)
        seg = np.zeros((len(idx_pos)+len(idx_neg), 1, 128, 128), dtype=np.int32)
        labels = np.zeros(len(idx_pos)+len(idx_neg), dtype=np.int32)
        idx_for_shuffle = np.arange(len(idx_neg)+len(idx_neg))
        np.random.shuffle(idx_for_shuffle)
        pos_data = np.array(array_pos[idx_pos])
        neg_data = np.array(array_neg[idx_neg])
        data[:len(idx_pos)] = pos_data[:, :4].astype(np.float32)
        data[len(idx_pos):] = neg_data[:, :4].astype(np.float32)
        seg[:len(idx_pos)][:, 0, :, :] = pos_data[:, 4].astype(np.int32)
        seg[len(idx_pos):][:, 0, :, :] = neg_data[:, 4].astype(np.int32)
        labels[:len(idx_pos)] = 1
        labels[len(idx_pos):] = 0
        data = data[idx_for_shuffle]
        seg = seg[idx_for_shuffle]
        labels = labels[idx_for_shuffle]
        # 25% flipped vertically, 25% flipped horizontically, 25% flipped both ways
        data[:int(len(idx_for_shuffle)/2)] = data[:int(len(idx_for_shuffle)/2)][:, :, ::-1, :]
        data[int(len(idx_for_shuffle)/4):int(len(idx_for_shuffle)*3/4)] = data[int(len(idx_for_shuffle)/4):int(len(idx_for_shuffle)*3/4)][:, :, :, ::-1]
        seg[:int(len(idx_for_shuffle)/2)] = seg[:int(len(idx_for_shuffle)/2)][:, :, ::-1, :]
        seg[int(len(idx_for_shuffle)/4):int(len(idx_for_shuffle)*3/4)] = seg[int(len(idx_for_shuffle)/4):int(len(idx_for_shuffle)*3/4)][:, :, :, ::-1]
        yield data, seg, labels


def memmapGenerator_t1km_flair_adc_cbv(array_neg, array_pos, BATCH_SIZE, n_elements_pos=None, n_elements_neg=None):
    np.random.seed()
    while True:
        if n_elements_pos is None:
            n_elements_pos = array_pos.shape[0]
        if n_elements_neg is None:
            n_elements_neg = array_neg.shape[0]
        idx_pos = np.random.choice(n_elements_pos, int(BATCH_SIZE/2.))
        idx_neg = np.random.choice(n_elements_neg, int(BATCH_SIZE/2.))
        data = np.zeros((len(idx_pos)+len(idx_neg), 4, 128, 128), dtype=np.float32)
        seg = np.zeros((len(idx_pos)+len(idx_neg), 1, 128, 128), dtype=np.int32)
        labels = np.zeros(len(idx_pos)+len(idx_neg), dtype=np.int32)
        idx_for_shuffle = np.arange(len(idx_neg)+len(idx_neg))
        np.random.shuffle(idx_for_shuffle)
        pos_data = np.array(array_pos[idx_pos])
        neg_data = np.array(array_neg[idx_neg])
        data[:len(idx_pos)] = pos_data[:, :4].astype(np.float32)
        data[len(idx_pos):] = neg_data[:, :4].astype(np.float32)
        seg[:len(idx_pos)][:, 0, :, :] = pos_data[:, 4].astype(np.int32)
        seg[len(idx_pos):][:, 0, :, :] = neg_data[:, 4].astype(np.int32)
        labels[:len(idx_pos)] = 1
        labels[len(idx_pos):] = 0
        data = data[idx_for_shuffle]
        seg = seg[idx_for_shuffle]
        labels = labels[idx_for_shuffle]
        yield data, seg, labels

def memmapGenerator_tumorClassRot(array_neg, array_pos, BATCH_SIZE, n_elements_pos=None, n_elements_neg=None):
    np.random.seed()
    while True:
        if n_elements_pos is None:
            n_elements_pos = array_pos.shape[0]
        if n_elements_neg is None:
            n_elements_neg = array_neg.shape[0]
        idx_pos = np.random.choice(n_elements_pos, int(BATCH_SIZE/2.))
        idx_neg = np.random.choice(n_elements_neg, int(BATCH_SIZE/2.))
        data = np.zeros((len(idx_pos)+len(idx_neg), 15, 128, 128), dtype=np.float32)
        seg = np.zeros((len(idx_pos)+len(idx_neg), 3, 128, 128), dtype=np.int32)
        labels = np.zeros(len(idx_pos)+len(idx_neg), dtype=np.int32)
        idx_for_shuffle = np.arange(len(idx_neg)+len(idx_neg))
        np.random.shuffle(idx_for_shuffle)
        pos_data = np.array(array_pos[idx_pos])
        neg_data = np.array(array_neg[idx_neg])
        data[:len(idx_pos)] = pos_data[:, :15].astype(np.float32)
        data[len(idx_pos):] = neg_data[:, :15].astype(np.float32)
        seg[:len(idx_pos)][:, :, :, :] = pos_data[:, 15:].astype(np.int32)
        seg[len(idx_pos):][:, :, :, :] = neg_data[:, 15:].astype(np.int32)
        labels[:len(idx_pos)] = 1
        labels[len(idx_pos):] = 0
        data = data[idx_for_shuffle]
        seg = seg[idx_for_shuffle]
        labels = labels[idx_for_shuffle]
        yield data, seg, labels


def memmapGeneratorDataAugm_t1km_flair_adc_cbv_markers(array_neg, array_pos, BATCH_SIZE, n_elements_pos=None, n_elements_neg=None):
    np.random.seed()
    while True:
        if n_elements_pos is None:
            n_elements_pos = array_pos.shape[0]
        if n_elements_neg is None:
            n_elements_neg = array_neg.shape[0]
        idx_pos = np.random.choice(n_elements_pos, int(BATCH_SIZE/2.))
        idx_neg = np.random.choice(n_elements_neg, int(BATCH_SIZE/2.))
        data = np.zeros((len(idx_pos)+len(idx_neg), 4, 128, 128), dtype=np.float32)
        seg = np.zeros((len(idx_pos)+len(idx_neg), 1, 128, 128), dtype=np.int32)
        labels = np.zeros(len(idx_pos)+len(idx_neg), dtype=np.int32)
        markers = np.zeros((len(idx_pos)+len(idx_neg), 3))
        idx_for_shuffle = np.arange(len(idx_neg)+len(idx_neg))
        np.random.shuffle(idx_for_shuffle)
        pos_data = np.array(array_pos[idx_pos])
        neg_data = np.array(array_neg[idx_neg])
        data[:len(idx_pos)][:, 0] = pos_data[:, 128**2 * 0 : 128**2 * 1].reshape(len(idx_pos), 128, 128).astype(np.float32)
        data[len(idx_pos):][:, 0] = neg_data[:, 128**2 * 0 : 128**2 * 1].reshape(len(idx_neg), 128, 128).astype(np.float32)
        data[:len(idx_pos)][:, 1] = pos_data[:, 128**2 * 1 : 128**2 * 2].reshape(len(idx_pos), 128, 128).astype(np.float32)
        data[len(idx_pos):][:, 1] = neg_data[:, 128**2 * 1 : 128**2 * 2].reshape(len(idx_neg), 128, 128).astype(np.float32)
        data[:len(idx_pos)][:, 2] = pos_data[:, 128**2 * 2 : 128**2 * 3].reshape(len(idx_pos), 128, 128).astype(np.float32)
        data[len(idx_pos):][:, 2] = neg_data[:, 128**2 * 2 : 128**2 * 3].reshape(len(idx_neg), 128, 128).astype(np.float32)
        data[:len(idx_pos)][:, 3] = pos_data[:, 128**2 * 3 : 128**2 * 4].reshape(len(idx_pos), 128, 128).astype(np.float32)
        data[len(idx_pos):][:, 3] = neg_data[:, 128**2 * 3 : 128**2 * 4].reshape(len(idx_neg), 128, 128).astype(np.float32)
        seg[:len(idx_pos)] = pos_data[:, 128**2 * 4 : 128**2 * 5].reshape(len(idx_pos), 1, 128, 128).astype(np.int32)
        seg[len(idx_pos):] = neg_data[:, 128**2 * 4 : 128**2 * 5].reshape(len(idx_neg), 1, 128, 128).astype(np.int32)
        markers[:len(idx_pos)] = pos_data[:, -3:]
        labels[:len(idx_pos)] = 1
        labels[len(idx_pos):] = 0
        data = data[idx_for_shuffle]
        markers = markers[idx_for_shuffle]
        seg = seg[idx_for_shuffle]
        labels = labels[idx_for_shuffle]
        # 25% flipped vertically, 25% flipped horizontically, 25% flipped both ways
        data[:int(len(idx_for_shuffle)/2)] = data[:int(len(idx_for_shuffle)/2)][:, :, ::-1, :]
        data[int(len(idx_for_shuffle)/4):int(len(idx_for_shuffle)*3/4)] = data[int(len(idx_for_shuffle)/4):int(len(idx_for_shuffle)*3/4)][:, :, :, ::-1]
        seg[:int(len(idx_for_shuffle)/2)] = seg[:int(len(idx_for_shuffle)/2)][:, :, ::-1, :]
        seg[int(len(idx_for_shuffle)/4):int(len(idx_for_shuffle)*3/4)] = seg[int(len(idx_for_shuffle)/4):int(len(idx_for_shuffle)*3/4)][:, :, :, ::-1]
        yield data, seg, labels, markers


def memmapGenerator_t1km_flair_adc_cbv_markers(array_neg, array_pos, BATCH_SIZE, n_elements_pos=None, n_elements_neg=None):
    np.random.seed()
    while True:
        if n_elements_pos is None:
            n_elements_pos = array_pos.shape[0]
        if n_elements_neg is None:
            n_elements_neg = array_neg.shape[0]
        idx_pos = np.random.choice(n_elements_pos, int(BATCH_SIZE/2.))
        idx_neg = np.random.choice(n_elements_neg, int(BATCH_SIZE/2.))
        data = np.zeros((len(idx_pos)+len(idx_neg), 4, 128, 128), dtype=np.float32)
        seg = np.zeros((len(idx_pos)+len(idx_neg), 1, 128, 128), dtype=np.int32)
        labels = np.zeros(len(idx_pos)+len(idx_neg), dtype=np.int32)
        markers = np.zeros((len(idx_pos)+len(idx_neg), 3))
        idx_for_shuffle = np.arange(len(idx_neg)+len(idx_neg))
        np.random.shuffle(idx_for_shuffle)
        pos_data = np.array(array_pos[idx_pos])
        neg_data = np.array(array_neg[idx_neg])
        data[:len(idx_pos)][:, 0] = pos_data[:, 128**2 * 0 : 128**2 * 1].reshape(len(idx_pos), 128, 128).astype(np.float32)
        data[len(idx_pos):][:, 0] = neg_data[:, 128**2 * 0 : 128**2 * 1].reshape(len(idx_neg), 128, 128).astype(np.float32)
        data[:len(idx_pos)][:, 1] = pos_data[:, 128**2 * 1 : 128**2 * 2].reshape(len(idx_pos), 128, 128).astype(np.float32)
        data[len(idx_pos):][:, 1] = neg_data[:, 128**2 * 1 : 128**2 * 2].reshape(len(idx_neg), 128, 128).astype(np.float32)
        data[:len(idx_pos)][:, 2] = pos_data[:, 128**2 * 2 : 128**2 * 3].reshape(len(idx_pos), 128, 128).astype(np.float32)
        data[len(idx_pos):][:, 2] = neg_data[:, 128**2 * 2 : 128**2 * 3].reshape(len(idx_neg), 128, 128).astype(np.float32)
        data[:len(idx_pos)][:, 3] = pos_data[:, 128**2 * 3 : 128**2 * 4].reshape(len(idx_pos), 128, 128).astype(np.float32)
        data[len(idx_pos):][:, 3] = neg_data[:, 128**2 * 3 : 128**2 * 4].reshape(len(idx_neg), 128, 128).astype(np.float32)
        seg[:len(idx_pos)] = pos_data[:, 128**2 * 4 : 128**2 * 5].reshape(len(idx_pos), 1, 128, 128).astype(np.int32)
        seg[len(idx_pos):] = neg_data[:, 128**2 * 4 : 128**2 * 5].reshape(len(idx_neg), 1, 128, 128).astype(np.int32)
        markers[:len(idx_pos)] = pos_data[:, -3:]
        labels[:len(idx_pos)] = 1
        labels[len(idx_pos):] = 0
        data = data[idx_for_shuffle]
        markers = markers[idx_for_shuffle]
        seg = seg[idx_for_shuffle]
        labels = labels[idx_for_shuffle]
        yield data, seg, labels, markers

def memmapGenerator_allInOne_markers(array_data, array_gt, BATCH_SIZE, validation_patients, marker="RTK2", mode="train", shuffle=True):
    np.random.seed()
    assert mode in ["train", "test"]
    assert marker in ["RTK2", "MGMT", "EGFR"]
    if marker == "RTK2":
        marker_idx = 1
    if marker == "MGMT":
        marker_idx = 2
    if marker == "EGFR":
        marker_idx = 3
    patient_ids = np.array(array_gt[:, 0])
    data_idx = find_entries_in_array(validation_patients, patient_ids)
    if mode == "train":
        data_idx = ~data_idx
    data_idx = np.where(data_idx)[0]
    all_idx_pos = data_idx[array_gt[data_idx, marker_idx] == 1]
    all_idx_neg = data_idx[array_gt[data_idx, marker_idx] == 0]
    data_shape = list(array_data.shape)
    data_shape[0] = BATCH_SIZE
    data_shape[1] -= 3
    seg_shape = deepcopy(data_shape)
    seg_shape[1] = 3
    while True:
        idx_pos = np.random.choice(all_idx_pos, int(BATCH_SIZE/2.))
        idx_neg = np.random.choice(all_idx_neg, int(BATCH_SIZE/2.))
        data = np.zeros(data_shape, dtype=np.float32)
        seg = np.zeros(seg_shape, dtype=np.int32)
        labels = np.zeros(BATCH_SIZE)
        data[:len(idx_pos)] = array_data[idx_pos, :-3]
        data[len(idx_pos):] = array_data[idx_neg, :-3]
        seg[:len(idx_pos)] = array_data[idx_pos, -3:]
        seg[len(idx_pos):] = array_data[idx_neg, -3:]
        labels[:len(idx_pos)] = 1
        labels[len(idx_pos):] = 0
        if shuffle:
            idx_for_shuffle = np.arange(len(idx_neg)+len(idx_neg))
            np.random.shuffle(idx_for_shuffle)
            data = data[idx_for_shuffle]
            seg = seg[idx_for_shuffle]
            labels = labels[idx_for_shuffle]
        yield data, seg, labels


def memmapGenerator_allInOne_segmentation(array_data, array_gt, BATCH_SIZE, validation_patients, mode="train", ignore=[], shuffle=True):
    np.random.seed()
    assert mode in ["train", "test"]
    patient_ids = np.array(array_gt[:, 0])
    data_idx = find_entries_in_array(validation_patients, patient_ids)
    if mode == "train":
        data_idx = ~data_idx
    data_idx = np.where(data_idx)[0]
    if len(ignore) > 0:
        data_idx = data_idx[~find_entries_in_array(ignore, array_gt[data_idx, 0])]
    while True:
        idx = np.random.choice(data_idx, BATCH_SIZE)
        data = np.array(array_data[idx, :-5]).astype(np.float32)
        seg = np.array(array_data[idx, -5:]).astype(np.float32)
        if shuffle:
            idx_for_shuffle = np.arange(len(idx))
            np.random.shuffle(idx_for_shuffle)
            data = data[idx_for_shuffle]
            seg = seg[idx_for_shuffle]
        yield data, seg, None

def memmapGenerator_allInOne_segmentation_lossSampling(array_data, array_gt, BATCH_SIZE, validation_patients, mode="train", ignore=[], losses=None, num_batches=None):
    # patches with higher loss are sampled more frequently
    if num_batches is None:
        num_batches = 1e100
    batches_generated = 0
    np.random.seed()
    assert mode in ["train", "test"]
    newshape_gt = list(array_gt.shape)
    newshape_gt[1] += 1
    array_gt2 = np.zeros(tuple(newshape_gt))
    array_gt2[:, :-1] = array_gt
    if losses is None:
        array_gt2[:, -1] = 1. / float(len(array_gt2))
    else:
        assert len(losses) == len(array_gt2)
        losses[losses < np.mean(losses)/10.] = np.mean(losses)/10.
        array_gt2[:, -1] = losses
    patient_ids = np.array(array_gt2[:, 0])
    data_idx = find_entries_in_array(validation_patients, patient_ids)
    if mode == "train":
        data_idx = ~data_idx
    data_idx = np.where(data_idx)[0]
    if len(ignore) > 0:
        data_idx = data_idx[~find_entries_in_array(ignore, array_gt2[data_idx, 0])]
    # probabilities have to sum to one...
    array_gt2[data_idx, -1] = array_gt2[data_idx, -1] / np.sum(array_gt2[data_idx, -1])
    while batches_generated < num_batches:
        idx = np.random.choice(data_idx, BATCH_SIZE, p=array_gt2[data_idx, -1])
        data = np.array(array_data[idx, :-5]).astype(np.float32)
        seg = np.array(array_data[idx, -5:]).astype(np.float32)
        batches_generated += 1
        yield data, seg, idx


def load_all_patients():
    folder = "/media/fabian/DeepLearningData/datasets/Hirntumor_raw_data/"
    all_patient_data = {}
    for i in range(150):
        if not path.isfile(folder + "patient_%03.0d_adc_data.npy" % i):
            continue
        if not path.isfile(folder + "patient_%03.0d_cbv_data.npy" % i):
            continue
        if not path.isfile(folder + "patient_%03.0d_flair_data.npy" % i):
            continue
        if not path.isfile(folder + "patient_%03.0d_segmentation.npy" % i):
            continue
        if not path.isfile(folder + "patient_%03.0d_t1_data.npy" % i):
            continue
        if not path.isfile(folder + "patient_%03.0d_t1km_data.npy" % i):
            continue
        if not path.isfile(folder + "patient_%03.0d_t1km_downsampled_128_data.npy" % i):
            continue
        all_patient_data[i] = {}

        all_patient_data[i]["t1"] = np.load(folder + "patient_%03.0d_t1_data.npy" % i, mmap_mode="r")
        all_patient_data[i]["t1km"] = np.load(folder + "patient_%03.0d_t1km_data.npy" % i, mmap_mode="r")
        all_patient_data[i]["adc"] = np.load(folder + "patient_%03.0d_adc_data.npy" % i, mmap_mode="r")
        all_patient_data[i]["cbv"] = np.load(folder + "patient_%03.0d_cbv_data.npy" % i, mmap_mode="r")
        all_patient_data[i]["flair"] = np.load(folder + "patient_%03.0d_flair_data.npy" % i, mmap_mode="r")
        all_patient_data[i]["seg"] = np.load(folder + "patient_%03.0d_segmentation.npy" % i, mmap_mode="r")
        all_patient_data[i]["t1km_ds"] = np.load(folder + "patient_%03.0d_t1km_downsampled_128_data.npy" % i, mmap_mode="r")
    return all_patient_data


class SegmentationBatchGeneratorFromRawData():
    def __init__(self, all_patient_data, BATCH_SIZE, validation_patients, PATCH_SIZE=(736, 736), mode="train", ignore=[], losses=None, num_batches=None, seed=None):
        assert mode in ["train", "test"]
        self._all_patient_data = all_patient_data
        self.BATCH_SIZE = BATCH_SIZE
        self._validation_patients = validation_patients
        self._mode = mode
        self._ignore = ignore
        self._losses = losses
        self._num_batches = num_batches
        self._seed = None
        self._resetted_rng = False
        self._iter_initialized = False
        self._p = None
        self._PATCH_SIZE = PATCH_SIZE

        # patches with higher loss are sampled more frequently
        if self._num_batches is None:
            self._num_batches = 1e100
        self._batches_generated = 0

        num_slices = 0
        for patient in self._all_patient_data.keys():
            shape = self._all_patient_data[patient]["t1km"].shape
            num_slices += (shape[0] - 4)
        self._loss_sampling_data = np.zeros((num_slices, 3)) # [identifier, patient id, slice id, loss of that slice]
        pos = 0
        for patient in self._all_patient_data.keys():
            num_slices_this_patient = self._all_patient_data[patient]["t1km"].shape[0] - 4
            self._loss_sampling_data[pos:pos+num_slices_this_patient, 0] = patient
            self._loss_sampling_data[pos:pos+num_slices_this_patient, 1] = np.arange(2, num_slices_this_patient+2)
            self._loss_sampling_data[pos:pos+num_slices_this_patient, 2] = 100
            pos += num_slices_this_patient

    def get_losses(self):
        return self._loss_sampling_data[:, -1]

    def set_losses(self, new_losses):
        assert len(new_losses) == self._loss_sampling_data.shape[0]
        self._loss_sampling_data.shape[:, -1] = new_losses

    def _initialize_iter(self):
        self._data_idx = find_entries_in_array(self._validation_patients, np.unique(self._loss_sampling_data[:, 0]))
        if self._mode == "train":
            self._data_idx = ~self._data_idx
        self._data_idx = np.where(self._data_idx)[0]
        if len(self._ignore) > 0:
            self._data_idx = self._data_idx[~find_entries_in_array(self._ignore, self._loss_sampling_data[self._data_idx, 0])]
        self._p = self._loss_sampling_data[self._data_idx, 2] / np.sum(self._loss_sampling_data[self._data_idx, 2])
        self._iter_initialized = True

    def __iter__(self):
        return self

    def next(self):
        if not self._iter_initialized:
            self._initialize_iter()
        if self._batches_generated >= self._num_batches:
            raise StopIteration
        idx = np.random.choice(self._data_idx, self.BATCH_SIZE, p=self._p)
        data = np.zeros((self.BATCH_SIZE, 25, self._PATCH_SIZE[0], self._PATCH_SIZE[1]), dtype=np.float32)
        seg = np.zeros((self.BATCH_SIZE, 5, self._PATCH_SIZE[0], self._PATCH_SIZE[1]), dtype=np.float32)
        for j, i in enumerate(idx):
            patient = self._loss_sampling_data[i, 0]
            slice = self._loss_sampling_data[i, 1]
            data_t1km = self._all_patient_data[patient]["t1km"][slice-2:slice+3]
            data_adc = self._all_patient_data[patient]["adc"][slice-2:slice+3]
            data_flair = self._all_patient_data[patient]["flair"][slice-2:slice+3]
            data_cbv = self._all_patient_data[patient]["cbv"][slice-2:slice+3]
            data_t1 = self._all_patient_data[patient]["t1"][slice-2:slice+3]
            data_seg = self._all_patient_data[patient]["seg"][slice-2:slice+3]

            data_t1km = self.resize_image_by_padding(np.array(data_t1km), self._PATCH_SIZE)
            data_adc = self.resize_image_by_padding(np.array(data_adc), self._PATCH_SIZE)
            data_flair = self.resize_image_by_padding(np.array(data_flair), self._PATCH_SIZE)
            data_cbv = self.resize_image_by_padding(np.array(data_cbv), self._PATCH_SIZE)
            data_t1 = self.resize_image_by_padding(np.array(data_t1), self._PATCH_SIZE)
            data_seg = self.resize_image_by_padding(np.array(data_seg), self._PATCH_SIZE)

            data[j, :5] = data_t1km
            data[j, 5:10] = data_adc
            data[j, 10:15] = data_flair
            data[j, 15:20] = data_cbv
            data[j, 20:25] = data_t1
            seg[j] = data_seg
        return data, seg, idx

    @staticmethod
    def resize_image_by_padding(image, new_shape, pad_value=None):
        shape = tuple(list(image.shape)[1:])
        new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape(2,2), axis=0))
        if pad_value is None:
            pad_value = image[0, 0, 0]
        pad_x = [(new_shape[0]-shape[0])/2, (new_shape[0]-shape[0])/2]
        pad_y = [(new_shape[1]-shape[1])/2, (new_shape[1]-shape[1])/2]
        if (new_shape[0]-shape[0])%2 == 1:
            pad_x[1] += 1
        if (new_shape[1]-shape[1])%2 == 1:
            pad_y[1] += 1
        res = np.ones([image.shape[0]]+list(new_shape), dtype=image.dtype) * pad_value
        start = np.array(new_shape)/2. - np.array(shape)/2.
        res[:, start[0]:start[0]+shape[0], start[1]:start[1]+shape[1]] = image
        return res

#if __name__ == "__main__":
#     a = load_all_patients()
#     b = SegmentationBatchGeneratorFromRawData(a, 1, [1, 2, 3], (256, 256))
#     _ = b.next()