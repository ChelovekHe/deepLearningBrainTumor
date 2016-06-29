__author__ = 'fabian'
import numpy as np
import IPython

def memmapGenerator(array_neg, array_pos, BATCH_SIZE, n_elements_pos=None, n_elements_neg=None):
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
