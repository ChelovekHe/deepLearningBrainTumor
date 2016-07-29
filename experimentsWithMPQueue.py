__author__ = 'fabian'
from multiprocessing import Process
from multiprocessing import Queue as MPQueue
import numpy as np
from numpy import memmap
import cPickle
from utils import memmapGenerator_t1km_flair_adc_cbv
from os import times
import matplotlib.pyplot as plt
from time import sleep


def multi_threaded_generator(generator, num_cached=10, num_threads=4):
    queue = MPQueue(maxsize=num_cached)

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
            # pretend we are doing some calculations
            sleep(0.5)
        queue.put('DONE')

    # start producer (in a background thread)
    for _ in xrange(num_threads):
        thread = Process(target=producer)
        thread.daemon = True
        thread.start()

    # run as consumer (read items from queue, in current thread)
    res = queue.get()
    while res is not 'DONE':
        yield res
        res = queue.get()


EXPERIMENT_NAME = "classifyPatches_memmap_v0.7_ws_resample_t1km_flair_adc_cbv_new"
memmap_name = "patchClassification_ws_resampled_t1km_flair_adc_cbv_new"
BATCH_SIZE = 70

with open("../data/%s_properties.pkl" % memmap_name, 'r') as f:
    memmap_properties = cPickle.load(f)
n_pos_train = memmap_properties["train_pos"]
n_neg_train = memmap_properties["train_neg"]
n_pos_val = memmap_properties["val_pos"]
n_neg_val = memmap_properties["val_neg"]
n_training_samples = memmap_properties["train_total"]
n_val_samples = memmap_properties["val_total"]


for num_threads in [1, 2, 4, 8]:
    train_pos_memmap = memmap("../data/%s_train_pos.memmap" % memmap_name, dtype=np.float32, mode="r", shape=memmap_properties["train_pos_shape"])
    train_neg_memmap = memmap("../data/%s_train_neg.memmap" % memmap_name, dtype=np.float32, mode="r", shape=memmap_properties["train_neg_shape"])
    val_pos_memmap = memmap("../data/%s_val_pos.memmap" % memmap_name, dtype=np.float32, mode="r", shape=memmap_properties["val_pos_shape"])
    val_neg_memmap = memmap("../data/%s_val_neg.memmap" % memmap_name, dtype=np.float32, mode="r", shape=memmap_properties["val_neg_shape"])
    start_time = times()[4]
    ctr = 0
    for data, seg, labels in multi_threaded_generator(memmapGenerator_t1km_flair_adc_cbv(train_neg_memmap, train_pos_memmap, BATCH_SIZE, n_pos_train, n_neg_train), num_threads=num_threads):
        if ctr < 100:
            ctr += 1
            continue
        else:
            break

    print num_threads, " thread(s) took ", times()[4] - start_time, "s"