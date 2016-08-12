__author__ = 'fabian'
import numpy as np
import matplotlib.pyplot as plt
import lasagne
from memmap_negPos_batchgen import memmapGenerator, memmapGenerator_t1km_flair
from memmap_negPos_batchgen import memmapGenerator_t1km_flair_adc_cbv, memmapGenerator_t1km_flair_adc_cbv_markers, memmapGenerator_tumorClassRot
from numpy import memmap
import cPickle
from multiprocessing import Process
from multiprocessing import Queue as MPQueue
from time import sleep
import theano
import theano.tensor as T

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

def threaded_generator(generator, num_cached=10):
    # this code is written by jan Schluter
    # copied from https://github.com/benanne/Lasagne/issues/12
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()

def printLosses(all_training_losses, all_training_accs, all_validation_losses, all_valid_accur, fname, samplesPerEpoch=10):
    fig, ax1 = plt.subplots()
    trainLoss_x_values = np.arange(1/float(samplesPerEpoch), len(all_training_losses)/float(samplesPerEpoch)+0.000001, 1/float(samplesPerEpoch))
    val_x_values = np.arange(1, len(all_validation_losses)+0.001, 1)
    ax1.plot(trainLoss_x_values, all_training_losses, 'b--')
    ax1.plot(val_x_values, all_validation_losses, color='b')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')

    ax2 = ax1.twinx()
    ax2.plot(trainLoss_x_values, all_training_accs, 'r--')
    ax2.plot(val_x_values, all_valid_accur, color='r')
    ax2.set_ylabel('accuracy')
    for t2 in ax2.get_yticklabels():
        t2.set_color('r')

    ax1.legend(['trainLoss', 'validLoss'], loc=0)
    ax2.legend(['trainAcc', 'validAcc'], loc=2)
    # ax2.legend(['valAccuracy'])
    plt.savefig(fname)
    plt.close()


def validate_result(img, convLayer):
    img_for_cnn = img[np.newaxis, np.newaxis, :, :]
    filtered_by_cnn = lasagne.layers.get_output(convLayer, img_for_cnn).eval()
    plt.figure(figsize=(12, 12))
    for i in xrange(filtered_by_cnn.shape[1]):
        plt.subplot(int(np.ceil(np.sqrt(filtered_by_cnn.shape[1]))), int(np.ceil(np.sqrt(filtered_by_cnn.shape[1]))), i+1)
        plt.imshow(filtered_by_cnn[0, i, :, :], cmap="gray", interpolation="nearest")
        plt.axis('off')
    plt.savefig("../results/filtered_by_cnn.png")
    plt.close()

    weights_w = convLayer.get_params()[0].get_value()
    weights_b = convLayer.get_params()[1].get_value()

    from scipy.signal import convolve2d
    res_scipy = []
    numFilters = weights_w.shape[0]
    for i in xrange(numFilters):
        weights = weights_w[i, 0, :, :]
        res_scipy.append(lasagne.nonlinearities.rectify(convolve2d(img, weights, mode='same') + weights_b[i]))

    plt.figure(figsize=(12, 12))
    for i in xrange(numFilters):
        plt.subplot(int(np.ceil(np.sqrt(numFilters))), int(np.ceil(np.sqrt(numFilters))), i+1)
        plt.imshow(res_scipy[i], cmap="gray", interpolation="nearest")
        plt.axis('off')
    plt.savefig("../results/filtered_by_scipy.png")
    plt.close()

def plot_some_data():
    memmap_name = "patchClassification_ws_resampled"
    with open("../data/%s_properties.pkl" % memmap_name, 'r') as f:
        memmap_properties = cPickle.load(f)
    n_pos_train = memmap_properties["train_pos"]
    n_neg_train = memmap_properties["train_neg"]
    n_pos_val = memmap_properties["val_pos"]
    n_neg_val = memmap_properties["val_neg"]
    train_pos_memmap = memmap("../data/%s_train_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["train_pos_shape"])
    train_neg_memmap = memmap("../data/%s_train_neg.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["train_neg_shape"])
    val_pos_memmap = memmap("../data/%s_val_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["val_pos_shape"])
    val_neg_memmap = memmap("../data/%s_val_neg.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["val_neg_shape"])
    i = 0
    ctr = 0
    for data, seg, labels in memmapGenerator(val_neg_memmap, val_pos_memmap, 128, n_pos_val, n_neg_val):
        if i == 2:
            break
        data2 = np.array(data)
        for img, segm, lab in zip(data2, seg, labels):
            img -= img.min()
            img /= img.max()
            plt.figure(figsize=(12,12))
            img = np.array(img[0]) # dont write into memmap
            img = np.repeat(img[np.newaxis, :, :], 3, 0)
            img = img.transpose((1, 2, 0))
            img[:, :, 0][segm[0] > 1] *= 1.0
            plt.imshow(img, interpolation='nearest')
            if lab == 0:
                color = 'green'
            else:
                color = 'red'
            plt.text(0, 0, lab, color=color, bbox=dict(facecolor='white', alpha=1))
            plt.savefig("../some_images/img_%04.0f.png"%ctr)
            plt.close()
            ctr += 1
        i += 1

def plot_some_data_t1km_flair():
    memmap_name = "patchClassification_ws_resampled_t1km_flair_new"
    with open("../data/%s_properties.pkl" % memmap_name, 'r') as f:
        memmap_properties = cPickle.load(f)
    n_pos_train = memmap_properties["train_pos"]
    n_neg_train = memmap_properties["train_neg"]
    n_pos_val = memmap_properties["val_pos"]
    n_neg_val = memmap_properties["val_neg"]
    train_pos_memmap = memmap("../data/%s_train_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["train_pos_shape"])
    train_neg_memmap = memmap("../data/%s_train_neg.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["train_neg_shape"])
    val_pos_memmap = memmap("../data/%s_val_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["val_pos_shape"])
    val_neg_memmap = memmap("../data/%s_val_neg.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["val_neg_shape"])
    i = 0
    ctr = 0
    for data, seg, labels in memmapGenerator_t1km_flair(val_neg_memmap, val_pos_memmap, 128, n_pos_val, n_neg_val):
        if i == 2:
            break
        data2 = np.array(data)
        for img, segm, lab in zip(data2, seg, labels):
            t1km_img = img[0]
            flair_img = img[1]
            plt.figure(figsize=(24,8))
            plt.subplot(1, 3, 1)
            plt.imshow(t1km_img, interpolation='nearest', cmap="gray")
            if lab == 0:
                color = 'green'
            else:
                color = 'red'
            plt.text(0, 0, lab, color=color, bbox=dict(facecolor='white', alpha=1))

            plt.subplot(1, 3, 2)
            plt.imshow(flair_img, interpolation='nearest', cmap="gray")
            plt.subplot(1, 3, 3)
            plt.imshow(segm[0], cmap="jet")
            plt.savefig("../some_images/img_%04.0f.png"%ctr)
            plt.close()
            ctr += 1
        i += 1


def plot_some_data_t1km_flair_adc_cbv():
    memmap_name = "patchClassification_ws_resampled_t1km_flair_adc_cbv_new"
    with open("../data/%s_properties.pkl" % memmap_name, 'r') as f:
        memmap_properties = cPickle.load(f)
    n_pos_train = memmap_properties["train_pos"]
    n_neg_train = memmap_properties["train_neg"]
    n_pos_val = memmap_properties["val_pos"]
    n_neg_val = memmap_properties["val_neg"]
    train_pos_memmap = memmap("../data/%s_train_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["train_pos_shape"])
    train_neg_memmap = memmap("../data/%s_train_neg.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["train_neg_shape"])
    val_pos_memmap = memmap("../data/%s_val_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["val_pos_shape"])
    val_neg_memmap = memmap("../data/%s_val_neg.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["val_neg_shape"])
    i = 0
    ctr = 0
    for data, seg, labels in memmapGenerator_t1km_flair_adc_cbv(val_neg_memmap, val_pos_memmap, 128, n_pos_val, n_neg_val):
        if i == 2:
            break
        data2 = np.array(data)
        for img, segm, lab in zip(data2, seg, labels):
            t1km_img = img[0]
            flair_img = img[1]
            adc_img = img[2]
            cbv_img = img[3]
            plt.figure(figsize=(24,8))
            plt.subplot(1, 5, 1)
            plt.imshow(t1km_img, interpolation='nearest', cmap="gray")
            if lab == 0:
                color = 'green'
            else:
                color = 'red'
            plt.text(0, 0, lab, color=color, bbox=dict(facecolor='white', alpha=1))

            plt.subplot(1, 5, 2)
            plt.imshow(flair_img, interpolation='nearest', cmap="gray")
            plt.subplot(1, 5, 3)
            plt.imshow(adc_img, interpolation='nearest', cmap="gray")
            plt.subplot(1, 5, 4)
            plt.imshow(cbv_img, interpolation='nearest', cmap="gray")
            plt.subplot(1, 5, 5)
            plt.imshow(segm[0], cmap="jet")
            plt.savefig("../some_images/img_%04.0f.png"%ctr)
            plt.close()
            ctr += 1
        i += 1


def plot_some_data_varNumChannels(memmap_gen = memmapGenerator_tumorClassRot):
    memmap_name = "patchClassification_ws_resampled_t1km_flair_adc_cbv_new"
    with open("../data/%s_properties.pkl" % memmap_name, 'r') as f:
        memmap_properties = cPickle.load(f)
    n_pos_train = memmap_properties["train_pos"]
    n_neg_train = memmap_properties["train_neg"]
    n_pos_val = memmap_properties["val_pos"]
    n_neg_val = memmap_properties["val_neg"]
    train_pos_memmap = memmap("../data/%s_train_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["train_pos_shape"])
    train_neg_memmap = memmap("../data/%s_train_neg.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["train_neg_shape"])
    val_pos_memmap = memmap("../data/%s_val_pos.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["val_pos_shape"])
    val_neg_memmap = memmap("../data/%s_val_neg.memmap" % memmap_name, dtype=np.float32, mode="r+", shape=memmap_properties["val_neg_shape"])
    i = 0
    ctr = 0
    for data, seg, labels in memmap_gen(val_neg_memmap, val_pos_memmap, 128, n_pos_val, n_neg_val):
        if i == 2:
            break
        data2 = np.array(data)
        for img, segm, lab in zip(data2, seg, labels):
            num_subplots = img.shape[0] + segm.shape[0]
            subplot_ctr = 1
            plt.figure(figsize=(12, 12))
            for x in xrange(img.shape[0]):
                plt.subplot(int(np.ceil(num_subplots**0.5)), int(np.ceil(num_subplots**0.5)), subplot_ctr)
                plt.imshow(img[x], interpolation='nearest', cmap="gray")
                if x == 0:
                    if lab == 0:
                        color = 'green'
                    else:
                        color = 'red'
                    plt.text(0, 0, lab, color=color, bbox=dict(facecolor='white', alpha=1))
                subplot_ctr += 1
            for x in xrange(segm.shape[0]):
                plt.subplot(int(np.ceil(num_subplots**0.5)), int(np.ceil(num_subplots**0.5)), subplot_ctr)
                plt.imshow(segm[x], interpolation='nearest', cmap="gray")
                subplot_ctr += 1
            plt.savefig("../some_images/img_%04.0f.png"%ctr)
            plt.close()
            ctr += 1
        i += 1


def plot_layer_weights(layer):
    conv_1_1_weights = layer.get_params()[0].get_value()

    plt.figure(figsize=(12, 12))
    for i in range(conv_1_1_weights.shape[0]):
        plt.subplot(int(np.ceil(np.sqrt(conv_1_1_weights.shape[0]))), int(np.ceil(np.sqrt(conv_1_1_weights.shape[0]))), i+1)
        plt.imshow(conv_1_1_weights[i, 0, :, :], cmap="gray", interpolation="nearest")
        plt.axis('off')
    plt.show()


def imgSaveFalsePositiveFalseNegativeCorrectPositiveCorrectNegative(pred_fn, n_images=16, BATCH_SIZE = 50):
    with open("../data/patchClassification_memmap_properties.pkl", 'r') as f:
        memmap_properties = cPickle.load(f)
    n_pos_val = memmap_properties["val_pos"]
    n_neg_val = memmap_properties["val_neg"]
    val_pos_memmap = memmap("../data/patchClassification_val_pos.memmap", dtype=np.float32, mode="r+", shape=memmap_properties["val_pos_shape"])
    val_neg_memmap = memmap("../data/patchClassification_val_neg.memmap", dtype=np.float32, mode="r+", shape=memmap_properties["val_neg_shape"])
    n_fpos = 0
    n_fneg = 0
    n_tpos = 0
    n_tneg = 0
    # it is simpler to just extract the fpos, fneg, tpos and tneg images one cathegory after the other. speed doesnt matter here
    plt.figure(figsize=(16,16))
    for data, seg, labels in memmapGenerator(val_neg_memmap, val_pos_memmap, BATCH_SIZE, n_pos_val, n_neg_val):
        if n_fpos < n_images:
            pred = pred_fn(data).argmax(-1)
            idx = np.where((pred == 1) & (labels == 0))[0]
            for id in idx:
                plt.subplot(int(np.ceil(np.sqrt(n_images))), int(np.ceil(np.sqrt(n_images))), n_fpos)
                plt.imshow(data[id, 0, :, :], cmap="gray", interpolation="nearest")
                plt.text(0, 0, labels[id], color='red', bbox=dict(facecolor='white', alpha=1))
                plt.text(0, 12, pred[id], color='green', bbox=dict(facecolor='white', alpha=1))
                plt.text(8, 0, 'true', color='blue')
                plt.text(8, 12, 'predicted', color='blue')
                n_fpos += 1
                if n_fpos >= n_images:
                    break
        else:
            break
    plt.savefig("../results/falsePositives.png")
    plt.close()

    plt.figure(figsize=(16,16))
    for data, seg, labels in memmapGenerator(val_neg_memmap, val_pos_memmap, BATCH_SIZE, n_pos_val, n_neg_val):
        if n_fneg < n_images:
            pred = pred_fn(data).argmax(-1)
            idx = np.where((pred == 0) & (labels == 1))[0]
            for id in idx:
                plt.subplot(int(np.ceil(np.sqrt(n_images))), int(np.ceil(np.sqrt(n_images))), n_fneg)
                plt.imshow(data[id, 0, :, :], cmap="gray", interpolation="nearest")
                plt.text(0, 0, labels[id], color='green', bbox=dict(facecolor='white', alpha=1))
                plt.text(0, 12, pred[id], color='red', bbox=dict(facecolor='white', alpha=1))
                plt.text(8, 0, 'true', color='blue')
                plt.text(8, 12, 'predicted', color='blue')
                n_fneg += 1
                if n_fneg >= n_images:
                    break
        else:
            break
    plt.savefig("../results/falseNegatives.png")
    plt.close()

    plt.figure(figsize=(16,16))
    for data, seg, labels in memmapGenerator(val_neg_memmap, val_pos_memmap, BATCH_SIZE, n_pos_val, n_neg_val):
        if n_tpos < n_images:
            pred = pred_fn(data).argmax(-1)
            idx = np.where((pred == 1) & (labels == 1))[0]
            for id in idx:
                plt.subplot(int(np.ceil(np.sqrt(n_images))), int(np.ceil(np.sqrt(n_images))), n_tpos)
                plt.imshow(data[id, 0, :, :], cmap="gray", interpolation="nearest")
                plt.text(0, 0, labels[id], color='green', bbox=dict(facecolor='white', alpha=1))
                plt.text(0, 12, pred[id], color='green', bbox=dict(facecolor='white', alpha=1))
                plt.text(8, 0, 'true', color='blue')
                plt.text(8, 12, 'predicted', color='blue')
                n_tpos += 1
                if n_tpos >= n_images:
                    break
        else:
            break
    plt.savefig("../results/truePositives.png")
    plt.close()

    plt.figure(figsize=(16,16))
    for data, seg, labels in memmapGenerator(val_neg_memmap, val_pos_memmap, BATCH_SIZE, n_pos_val, n_neg_val):
        if n_tneg < n_images:
            pred = pred_fn(data).argmax(-1)
            idx = np.where((pred == 0) & (labels == 0))[0]
            for id in idx:
                plt.subplot(int(np.ceil(np.sqrt(n_images))), int(np.ceil(np.sqrt(n_images))), n_tneg)
                plt.imshow(data[id, 0, :, :], cmap="gray", interpolation="nearest")
                plt.text(0, 0, labels[id], color='red', bbox=dict(facecolor='white', alpha=1))
                plt.text(0, 12, pred[id], color='red', bbox=dict(facecolor='white', alpha=1))
                plt.text(8, 0, 'true', color='blue')
                plt.text(8, 12, 'predicted', color='blue')
                n_tneg += 1
                if n_tneg >= n_images:
                    break
        else:
            break
    plt.savefig("../results/trueNegatives.png")
    plt.close()


def create_matrix_rotation_x(angle, matrix = None):
    rotation_x = np.array([[1,              0,              0],
                           [0,              np.cos(angle),  -np.sin(angle)],
                           [0,              np.sin(angle),  np.cos(angle)]])
    if matrix is None:
        return rotation_x

    return np.dot(matrix, rotation_x)


def create_matrix_rotation_y(angle, matrix = None):
    rotation_y = np.array([[np.cos(angle),  0,              np.sin(angle)],
                           [0,              1,              0],
                           [-np.sin(angle), 0,              np.cos(angle)]])
    if matrix is None:
        return rotation_y

    return np.dot(matrix, rotation_y)


def create_matrix_rotation_z(angle, matrix = None):
    rotation_z = np.array([[np.cos(angle),  -np.sin(angle), 0],
                           [np.sin(angle),  np.cos(angle),  0],
                           [0,              0,              1]])
    if matrix is None:
        return rotation_z

    return np.dot(matrix, rotation_z)


def create_random_rotation():
    return create_matrix_rotation_x(np.random.uniform(0.0, 2*np.pi), create_matrix_rotation_y(np.random.uniform(0.0, 2*np.pi), create_matrix_rotation_z(np.random.uniform(0.0, 2*np.pi))))

def show_segmentation_results(data, seg_true, seg_pred, img_ctr=0):
    n_channels = data.shape[1]
    n_images_in_figure = float(n_channels + 3)
    n_cols_and_rows = int(np.ceil(n_images_in_figure**0.5))
    seg_diff = np.zeros(seg_true.shape)
    seg_diff[seg_true!=seg_pred] = 1
    for x in range(0, data.shape[0]):
        plt.figure(figsize=(10,10))
        for i in range(1, n_channels+1):
            plt.subplot(n_cols_and_rows, n_cols_and_rows, i)
            plt.imshow(data[x, i-1, :, :], cmap="gray", interpolation="none")
        plt.subplot(n_cols_and_rows, n_cols_and_rows, n_channels+1)
        plt.imshow(seg_true[x, 0, :, :], cmap="jet", interpolation="none")
        plt.subplot(n_cols_and_rows, n_cols_and_rows, n_channels+2)
        plt.imshow(seg_pred[x, 0, :, :], cmap="jet", interpolation="none")
        plt.subplot(n_cols_and_rows, n_cols_and_rows, n_channels+3)
        plt.imshow(seg_diff[x, 0, :, :], cmap="gray", interpolation="none")
        plt.savefig("../some_images/seg_res_%04.0d.png"%(img_ctr+x))
        plt.close()
    return img_ctr+data.shape[0]




from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

def elastic_transform(image, alpha=100, sigma=10, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape)==2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    return map_coordinates(image, indices, order=3).reshape(shape)

def elastic_transform_3d(image, alpha=100, sigma=10, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape)==3

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1)), np.reshape(z+dz, (-1, 1))

    return map_coordinates(image, indices, order=3).reshape(shape)

def generate_elastic_deform_coordinates(shape, alpha, sigma):
    random_state = np.random.RandomState(None)
    n_dim = len(shape)
    offsets = []
    for _ in range(n_dim):
        offsets.append(gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha)

    tmp = tuple([np.arange(i) for i in shape])
    coords = np.meshgrid(*tmp, indexing='ij')
    indices = [np.reshape(i+j, (-1, 1)) for i,j in zip(offsets, coords)]
    return indices


def test_some_deformations():
    from skimage import data
    img = data.camera()
    plt.figure(figsize=(10,10))
    for i, alpha in enumerate([0, 0.5, 1., 2., 5.]):
        for j, sigma in enumerate([0, 50., 100., 200., 500.]):
            img2 = elastic_transform(img, alpha, sigma)
            plt.subplot(5, 5, i*5 + (j+1))
            plt.imshow(img2)
    plt.show()


def plot_layer_activations(layer, data, output_fname="../results/layerActivation.png"):
    pred = lasagne.layers.get_output(layer, data).eval()
    n_channels = pred.shape[1]
    plt.figure(figsize=(12, 12))
    plots_per_axis = int(np.ceil(np.sqrt(n_channels)))
    for i in xrange(n_channels):
        plt.subplot(plots_per_axis, plots_per_axis, i+1)
        plt.imshow(pred[0, i, :, :], cmap="gray", interpolation="nearest")
    plt.savefig(output_fname)
    plt.close()

