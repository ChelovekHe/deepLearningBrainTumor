__author__ = 'fabian'
import numpy as np
import IPython
import os.path as path
from numpy import memmap


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

