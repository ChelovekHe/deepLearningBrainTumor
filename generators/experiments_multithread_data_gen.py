__author__ = 'fabian'
import numpy as np
import cPickle
from numpy import memmap
from multithreaded_generators import Multithreaded_Generator
from data_generators import memmapGenerator_allInOne_segmentation_lossSampling

dataset_folder = "/media/fabian/DeepLearningData/datasets/"
EXPERIMENT_NAME = "segment_tumor_v0.1_Unet"
memmap_name = "patchSegmentation_allInOne_ws_t1km_flair_adc_cbv_resized"

BATCH_SIZE = 10
PATCH_SIZE = 15

with open(dataset_folder + "%s_properties.pkl" % (memmap_name), 'r') as f:
    my_dict = cPickle.load(f)

data_ctr = my_dict['n_data']
train_shape = my_dict['train_neg_shape']
info_memmap_shape = my_dict['info_shape']
memmap_data = memmap(dataset_folder + "%s.memmap" % (memmap_name), dtype=np.float32, mode="r", shape=train_shape)
memmap_gt = memmap(dataset_folder + "%s_info.memmap" % (memmap_name), dtype=np.float32, mode="r", shape=info_memmap_shape)


data_gen = memmapGenerator_allInOne_segmentation_lossSampling(memmap_data, memmap_gt, 1, [0, 1], num_batches=10)

for data, seg, ids in data_gen:
    print ids[0]

for _ in range(5):
    data_gen = memmapGenerator_allInOne_segmentation_lossSampling(memmap_data, memmap_gt, 1, [0, 1])
    data_gen_mt = Multithreaded_Generator(data_gen, 8, 30)
    ctr = 0
    for data, seg, ids in data_gen_mt:
        print ids[0]
        ctr += 1
        if ctr > 10:
            break
    data_gen_mt._finish()
