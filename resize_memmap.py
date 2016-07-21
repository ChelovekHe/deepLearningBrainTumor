__author__ = 'fabian'
import numpy as np
from numpy import memmap
import cPickle

# memmap_name = "patchClassification_ws_resampled_t1km_flair_adc_cbv"
new_memmap_name = memmap_name + "_new"

with open("%s_properties.pkl" % memmap_name, 'r') as f:
    memmap_properties = cPickle.load(f)
n_pos_train = memmap_properties["train_pos"]
n_neg_train = memmap_properties["train_neg"]
n_pos_val = memmap_properties["val_pos"]
n_neg_val = memmap_properties["val_neg"]
n_training_samples = memmap_properties["train_total"]
n_val_samples = memmap_properties["val_total"]

train_pos_memmap = memmap("%s_train_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["train_pos_shape"])
train_neg_memmap = memmap("%s_train_neg.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["train_neg_shape"])
val_pos_memmap = memmap("%s_val_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["val_pos_shape"])
val_neg_memmap = memmap("%s_val_neg.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["val_neg_shape"])


'''new_train_pos_shape = (n_pos_train, memmap_properties["train_pos_shape"][1])
new_train_neg_shape = (n_neg_train, memmap_properties["train_neg_shape"][1])
new_val_pos_shape = (n_pos_val, memmap_properties["val_pos_shape"][1])
new_val_neg_shape = (n_neg_val, memmap_properties["val_neg_shape"][1])'''
new_train_pos_shape = (n_pos_train, memmap_properties["train_pos_shape"][1], memmap_properties["train_pos_shape"][2], memmap_properties["train_pos_shape"][3])
new_train_neg_shape = (n_neg_train, memmap_properties["train_neg_shape"][1], memmap_properties["train_neg_shape"][2], memmap_properties["train_neg_shape"][3])
new_val_pos_shape = (n_pos_val, memmap_properties["val_pos_shape"][1], memmap_properties["val_pos_shape"][2], memmap_properties["val_pos_shape"][3])
new_val_neg_shape = (n_neg_val, memmap_properties["val_neg_shape"][1], memmap_properties["val_neg_shape"][2], memmap_properties["val_neg_shape"][3])

train_pos_memmap_new = memmap("%s_train_pos.memmap" % new_memmap_name, dtype=np.float32, mode="w+", shape=new_train_pos_shape)
train_neg_memmap_new = memmap("%s_train_neg.memmap" % new_memmap_name, dtype=np.float32, mode="w+", shape=new_train_neg_shape)
val_pos_memmap_new = memmap("%s_val_pos.memmap" % new_memmap_name, dtype=np.float32, mode="w+", shape=new_val_pos_shape)
val_neg_memmap_new = memmap("%s_val_neg.memmap" % new_memmap_name, dtype=np.float32, mode="w+", shape=new_val_neg_shape)

train_pos_memmap_new[:] = train_pos_memmap[:n_pos_train]
train_neg_memmap_new[:] = train_neg_memmap[:n_neg_train]
val_pos_memmap_new[:] = val_pos_memmap[:n_pos_val]
val_neg_memmap_new[:] = val_neg_memmap[:n_neg_val]


memmap_properties["train_pos_shape"] = new_train_pos_shape
memmap_properties["train_neg_shape"] = new_train_neg_shape
memmap_properties["val_pos_shape"] = new_val_pos_shape
memmap_properties["val_neg_shape"] = new_val_neg_shape

with open("%s_properties.pkl" % new_memmap_name, 'w') as f:
    cPickle.dump(memmap_properties, f)