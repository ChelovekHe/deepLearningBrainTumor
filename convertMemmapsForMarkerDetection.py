__author__ = 'fabian'
import numpy as np
from numpy import memmap
import matplotlib.pyplot as plt
import cPickle

memmap_name = "patchClassification_ws_resampled_t1km_flair_adc_cbv_markers_new"
new_memmap_name = "patchClassification_ws_res_t1km_flair_adc_cbv_EGFR"

with open("../data/%s_properties.pkl" % memmap_name, 'r') as f:
    memmap_properties = cPickle.load(f)
n_pos_train = memmap_properties["train_pos"]
n_neg_train = memmap_properties["train_neg"]
n_pos_val = memmap_properties["val_pos"]
n_neg_val = memmap_properties["val_neg"]
n_training_samples = memmap_properties["train_total"]
n_val_samples = memmap_properties["val_total"]

train_pos_memmap = memmap("../data/%s_train_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["train_pos_shape"])
val_pos_memmap = memmap("../data/%s_val_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["val_pos_shape"])

train_pos_new_memmap = memmap("../data/%s_train_pos.memmap" % new_memmap_name, dtype=np.float32, mode="w+", shape=(n_pos_train, 5, 128, 128))
train_neg_new_memmap = memmap("../data/%s_train_neg.memmap" % new_memmap_name, dtype=np.float32, mode="w+", shape=(n_pos_train, 5, 128, 128))
val_pos_new_memmap = memmap("../data/%s_val_pos.memmap" % new_memmap_name, dtype=np.float32, mode="w+", shape=(n_pos_val, 5, 128, 128))
val_neg_new_memmap = memmap("../data/%s_val_neg.memmap" % new_memmap_name, dtype=np.float32, mode="w+", shape=(n_pos_val, 5, 128, 128))

n_train_pos = 0
n_train_neg = 0
for i in xrange(n_pos_train):
    if train_pos_memmap[i, -1] == 0:
        train_pos_new_memmap[n_train_pos][0] = train_pos_memmap[i][0 * 128**2 : 1 * 128**2].reshape((128, 128))
        train_pos_new_memmap[n_train_pos][1] = train_pos_memmap[i][1 * 128**2 : 2 * 128**2].reshape((128, 128))
        train_pos_new_memmap[n_train_pos][2] = train_pos_memmap[i][2 * 128**2 : 3 * 128**2].reshape((128, 128))
        train_pos_new_memmap[n_train_pos][3] = train_pos_memmap[i][3 * 128**2 : 4 * 128**2].reshape((128, 128))
        train_pos_new_memmap[n_train_pos][4] = train_pos_memmap[i][4 * 128**2 : 5 * 128**2].reshape((128, 128))
        n_train_pos += 1
    elif train_pos_memmap[i, -1] == 1:
        train_neg_new_memmap[n_train_neg][0] = train_pos_memmap[i][0 * 128**2 : 1 * 128**2].reshape((128, 128))
        train_neg_new_memmap[n_train_neg][1] = train_pos_memmap[i][1 * 128**2 : 2 * 128**2].reshape((128, 128))
        train_neg_new_memmap[n_train_neg][2] = train_pos_memmap[i][2 * 128**2 : 3 * 128**2].reshape((128, 128))
        train_neg_new_memmap[n_train_neg][3] = train_pos_memmap[i][3 * 128**2 : 4 * 128**2].reshape((128, 128))
        train_neg_new_memmap[n_train_neg][4] = train_pos_memmap[i][4 * 128**2 : 5 * 128**2].reshape((128, 128))
        n_train_neg += 1

n_val_pos = 0
n_val_neg = 0
for i in xrange(n_pos_val):
    if train_pos_memmap[i, -1] == 0:
        val_pos_new_memmap[n_val_pos][0] = val_pos_memmap[i][0 * 128**2 : 1 * 128**2].reshape((128, 128))
        val_pos_new_memmap[n_val_pos][1] = val_pos_memmap[i][1 * 128**2 : 2 * 128**2].reshape((128, 128))
        val_pos_new_memmap[n_val_pos][2] = val_pos_memmap[i][2 * 128**2 : 3 * 128**2].reshape((128, 128))
        val_pos_new_memmap[n_val_pos][3] = val_pos_memmap[i][3 * 128**2 : 4 * 128**2].reshape((128, 128))
        val_pos_new_memmap[n_val_pos][4] = val_pos_memmap[i][4 * 128**2 : 5 * 128**2].reshape((128, 128))
        n_val_pos += 1
    elif train_pos_memmap[i, -1] == 1:
        val_neg_new_memmap[n_val_neg][0] = val_pos_memmap[i][0 * 128**2 : 1 * 128**2].reshape((128, 128))
        val_neg_new_memmap[n_val_neg][1] = val_pos_memmap[i][1 * 128**2 : 2 * 128**2].reshape((128, 128))
        val_neg_new_memmap[n_val_neg][2] = val_pos_memmap[i][2 * 128**2 : 3 * 128**2].reshape((128, 128))
        val_neg_new_memmap[n_val_neg][3] = val_pos_memmap[i][3 * 128**2 : 4 * 128**2].reshape((128, 128))
        val_neg_new_memmap[n_val_neg][4] = val_pos_memmap[i][4 * 128**2 : 5 * 128**2].reshape((128, 128))
        n_val_neg += 1


my_dict = {
    "train_total" : n_train_pos + n_train_neg,
    "train_pos": n_train_pos,
    "train_neg": n_train_neg,
    "val_total" : n_val_pos + n_val_neg,
    "val_pos": n_val_pos,
    "val_neg": n_val_neg,
    "train_neg_shape": (n_pos_train, 5, 128, 128),
    "train_pos_shape": (n_pos_train, 5, 128, 128),
    "val_neg_shape": (n_pos_val, 5, 128, 128),
    "val_pos_shape": (n_pos_val, 5, 128, 128)

}
with open("../data/%s_properties.pkl" % new_memmap_name, 'w') as f:
    cPickle.dump(my_dict, f)