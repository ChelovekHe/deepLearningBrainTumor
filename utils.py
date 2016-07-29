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



def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

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
