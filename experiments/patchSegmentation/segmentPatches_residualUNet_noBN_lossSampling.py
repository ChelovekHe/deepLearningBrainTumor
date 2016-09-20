__author__ = 'fabian'
import theano
import lasagne
import theano.tensor as T
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import cPickle as pickle
import sys
sys.path.append("../../neural_networks/")
sys.path.append("../../generators/")
sys.path.append("../../utils/")
sys.path.append("../../dataset_utils/")
import cPickle
from copy import deepcopy
from numpy import memmap

from UNet import build_deep_residual_UNet
from data_generators import memmapGenerator_allInOne_segmentation_lossSampling
from data_augmentation_generators import seg_channel_selection_generator, center_crop_generator, rotation_generator, elastric_transform_generator
from multithreaded_generators import multi_threaded_generator
from utils_plotting import show_segmentation_results, printLosses, plot_all_layer_activations
from generator_utils import elastic_transform_2d
from multithreaded_generators import Multithreaded_Generator
from sklearn.metrics import roc_auc_score

sys.setrecursionlimit(2000)

dataset_folder = "/media/fabian/DeepLearningData/datasets/"
EXPERIMENT_NAME = "segment_tumor_resudialUnet_noBN_lossSampling"
memmap_name = "patchSegmentation_allInOne_ws_t1km_flair_adc_cbv_resized"

BATCH_SIZE = 1
PATCH_SIZE = 256

with open(dataset_folder + "%s_properties.pkl" % (memmap_name), 'r') as f:
    my_dict = cPickle.load(f)

data_ctr = my_dict['n_data']
train_shape = my_dict['train_neg_shape']
info_memmap_shape = my_dict['info_shape']

class_frequencies = np.zeros(5, dtype=np.float32)
for i in range(5):
    class_frequencies[i] = my_dict['class_frequencies'][i]
class_frequencies = np.sqrt(class_frequencies)**0.5
class_frequencies2 = deepcopy(class_frequencies)
for i in range(5):
    class_frequencies2[i] = class_frequencies[range(5) != i] / class_frequencies[i]
class_frequencies2 /= np.sum(class_frequencies2)
class_frequencies2 *= 5

memmap_data = memmap(dataset_folder + "%s.memmap" % (memmap_name), dtype=np.float32, mode="r", shape=train_shape)
memmap_gt = memmap(dataset_folder + "%s_info.memmap" % (memmap_name), dtype=np.float32, mode="r", shape=info_memmap_shape)

patient_ids = np.unique(memmap_gt[:, 0]).astype(int)
#validation_patients = np.random.choice(patient_ids, 15, False)

# I should have used replace=False. Whatever...
validation_patients = [ 75,   1,  67,   1, 127, 120,  94, 131,  78,  74,  62,  10,  65, 47, 124]

n_training_samples = int(float(len(patient_ids) - len(validation_patients)) / float(len(patient_ids)) * memmap_data.shape[0])
n_val_samples = int(float(len(validation_patients)) / float(len(patient_ids)) * memmap_data.shape[0])

'''d, s, l = data_gen_train.next()
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(d[0,0], cmap="gray")
plt.subplot(1, 3, 2)
d1=elastic_transform_2d(d[0,0], 550., 20.)
plt.imshow(d1, cmap="gray")
plt.subplot(1, 3, 3)
plt.imshow(d[0,0]-d1)
plt.show()
plt.close()'''

data_gen_validation = memmapGenerator_allInOne_segmentation_lossSampling(memmap_data, memmap_gt, BATCH_SIZE, validation_patients, mode="test", ignore=[40])
data_gen_validation = center_crop_generator(data_gen_validation, (PATCH_SIZE, PATCH_SIZE))
data_gen_validation = seg_channel_selection_generator(data_gen_validation, [2])
data_gen_validation = multi_threaded_generator(data_gen_validation, num_threads=4, num_cached=10)
net = build_deep_residual_UNet(20, BATCH_SIZE, num_output_classes=5, base_n_filters=16, input_dim=(PATCH_SIZE, PATCH_SIZE), n_res_blocks=1, doBN=False)
output_layer_for_loss = net["output_flattened"]
'''with open("../../../results/segment_tumor_v0.2_UNet_lossSampling/segment_tumor_v0.2_Unet_lossSampling_Params_ep30.pkl", 'r') as f:
    params = cPickle.load(f)
    lasagne.layers.set_all_param_values(output_layer_for_loss, params)
with open("../../../results/segment_tumor_v0.2_UNet_lossSampling/segment_tumor_v0.2_Unet_lossSampling_allLossesNAccur_ep30.pkl", 'r') as f:
    # [all_training_losses, all_training_accuracies, all_validation_losses, all_validation_accuracies, auc_all] = cPickle.load(f)
    [all_training_losses, all_training_accuracies, all_validation_losses, all_validation_accuracies] = cPickle.load(f)'''

n_batches_per_epoch = 5000
# n_batches_per_epoch = np.floor(n_training_samples/float(BATCH_SIZE))
n_test_batches = 500
# n_test_batches = np.floor(n_val_samples/float(BATCH_SIZE))

x_sym = T.tensor4()
seg_sym = T.ivector()
w_sym = T.vector()

# add some weight decay
l2_loss = lasagne.regularization.regularize_network_params(output_layer_for_loss, lasagne.regularization.l2) * 5e-5

# the distinction between prediction_train and test is important only if we enable dropout
prediction_train = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=False)
# we could use a binary loss but I stuck with categorical crossentropy so that less code has to be changed if your
# application has more than two classes
loss_vec = lasagne.objectives.categorical_crossentropy(prediction_train, seg_sym)
loss_vec *= w_sym
loss = loss_vec.mean()
loss += l2_loss
acc_train = T.mean(T.eq(T.argmax(prediction_train, axis=1), seg_sym), dtype=theano.config.floatX)

prediction_test = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=True)
loss_val = lasagne.objectives.categorical_crossentropy(prediction_test, seg_sym)

# we multiply our loss by a weight map. In this example the weight map only increases the loss for road pixels and
# decreases the loss for other pixels. We do this to ensure that the network puts more focus on getting the roads
# right
loss_val *= w_sym
loss_val = loss_val.mean()
loss_val += l2_loss
acc = T.mean(T.eq(T.argmax(prediction_test, axis=1), seg_sym), dtype=theano.config.floatX)

# learning rate has to be a shared variablebecause we decrease it with every epoch
params = lasagne.layers.get_all_params(output_layer_for_loss, trainable=True)
learning_rate = theano.shared(np.float32(0.001))
updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)
# updates = lasagne.updates.nesterov_momentum(loss, params, 0.005, 0.99)

# create a convenience function to get the segmentation
seg_output = lasagne.layers.get_output(net["output_segmentation"], x_sym, deterministic=True)
seg_output = seg_output.argmax(1)

train_fn = theano.function([x_sym, seg_sym, w_sym], [loss_vec, acc_train], updates=updates)
val_fn = theano.function([x_sym, seg_sym, w_sym], [loss_val, acc])
get_segmentation = theano.function([x_sym], seg_output)
# we need this for calculating the AUC score
get_class_probas = theano.function([x_sym], prediction_test)

n_feedbacks_per_epoch = 100.

all_training_losses = []
all_validation_losses = []
all_validation_accuracies = []
all_training_accuracies = []
auc_all = []

'''def update_loss_per_sample(loss_old, losses):
    # computes a moving average on the losses
    avg_loss = np.mean(losses[:, 1])
    loss_new = (loss_old + losses/avg_loss) / 2.
    return loss_new'''

losses = np.ones(len(memmap_gt))
def update_losses(losses, idx, loss):
    losses[idx] = (losses[idx] + loss*2.) / 3.
    return losses

def convert_seg_flat_to_binary_label_indicator_array(seg_flat, num_classes=5):
    seg2 = np.zeros((len(seg_flat), num_classes))
    for i in xrange(seg2.shape[0]):
        seg2[i, int(seg_flat[i])] = 1
    return seg2

n_epochs = 80
auc_scores=None
for epoch in range(0,n_epochs):
    data_gen_train = memmapGenerator_allInOne_segmentation_lossSampling(memmap_data, memmap_gt, BATCH_SIZE, validation_patients, mode="train", ignore=[40], losses=losses)
    data_gen_train = seg_channel_selection_generator(data_gen_train, [2])
    data_gen_train = rotation_generator(data_gen_train)
    data_gen_train = center_crop_generator(data_gen_train, (PATCH_SIZE, PATCH_SIZE))
    data_gen_train = elastric_transform_generator(data_gen_train, 550., 20.)
    data_gen_train = Multithreaded_Generator(data_gen_train, 4, 20)
    data_gen_train._start()
    print "epoch: ", epoch
    train_loss = 0
    train_acc_tmp = 0
    train_loss_tmp = 0
    batch_ctr = 0
    for data, seg, idx in data_gen_train:
        if batch_ctr != 0 and batch_ctr % int(np.floor(n_batches_per_epoch/n_feedbacks_per_epoch)) == 0:
            print "number of batches: ", batch_ctr, "/", n_batches_per_epoch
            print "training_loss since last update: ", train_loss_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch), " train accuracy: ", train_acc_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch)
            all_training_losses.append(train_loss_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch))
            all_training_accuracies.append(train_acc_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch))
            train_loss_tmp = 0
            train_acc_tmp = 0
            printLosses(all_training_losses, all_training_accuracies, all_validation_losses, all_validation_accuracies, "../../../results/%s.png" % EXPERIMENT_NAME, n_feedbacks_per_epoch, auc_scores=auc_scores, auc_labels=["bg", "brain", "edema", "ce_tumor", "necrosis"], ylim_score=(0,1.5))
        # loss, acc = train_fn(data, convert_seg_map_for_crossentropy(seg, range(4)).astype(np.float32))
        seg_flat = seg.flatten().astype(np.int32)
        w = class_frequencies2[seg_flat]
        loss_vec, acc = train_fn(data, seg_flat, w) #class_weights[seg_flat]
        loss = loss_vec.mean()
        loss_per_sample = loss_vec.reshape(BATCH_SIZE, -1).mean(axis=1)
        losses = update_losses(losses, idx, loss_per_sample)
        train_loss += loss
        train_loss_tmp += loss
        train_acc_tmp += acc
        batch_ctr += 1
        if batch_ctr > n_batches_per_epoch:
            break

    data_gen_train._finish()

    train_loss /= n_batches_per_epoch
    print "training loss average on epoch: ", train_loss
    if epoch > 2:
        losses[:] = train_loss
    elif epoch <= 2:
        losses = np.ones(len(memmap_gt))

    y_true = []
    y_pred = []
    test_loss = 0
    accuracies = []
    valid_batch_ctr = 0
    for data, seg, labels in data_gen_validation:
        # loss, acc = val_fn(data, convert_seg_map_for_crossentropy(seg, range(4)).astype(np.float32))
        seg_flat = seg.flatten().astype(np.int32)
        w = class_frequencies2[seg_flat]
        loss, acc = val_fn(data, seg_flat, w) #, class_weights[seg_flat]
        test_loss += loss
        accuracies.append(acc)
        valid_batch_ctr += 1
        y_true.append(convert_seg_flat_to_binary_label_indicator_array(seg_flat))
        y_pred.append(get_class_probas(data))
        if valid_batch_ctr > n_test_batches:
            break
    test_loss /= n_test_batches
    print "test loss: ", test_loss
    print "test acc: ", np.mean(accuracies), "\n"
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    scores = roc_auc_score(y_true, y_pred, None)
    auc_all.append(scores)
    all_validation_losses.append(test_loss)
    all_validation_accuracies.append(np.mean(accuracies))
    auc_scores = np.concatenate(auc_all, axis=0).reshape(-1, 5)
    printLosses(all_training_losses, all_training_accuracies, all_validation_losses, all_validation_accuracies, "../../../results/%s.png" % EXPERIMENT_NAME, n_feedbacks_per_epoch, auc_scores=auc_scores, auc_labels=["bg", "brain", "edema", "ce_tumor", "necrosis"], ylim_score=(0,1.5))
    learning_rate *= 0.62
    with open("../../../results/%s_Params_ep%d.pkl" % (EXPERIMENT_NAME, epoch), 'w') as f:
        cPickle.dump(lasagne.layers.get_all_param_values(output_layer_for_loss), f)
    with open("../../../results/%s_allLossesNAccur_ep%d.pkl"% (EXPERIMENT_NAME, epoch), 'w') as f:
        cPickle.dump([all_training_losses, all_training_accuracies, all_validation_losses, all_validation_accuracies, auc_all], f)
    with open("../../../results/%s_lossPerPatch_ep%d.pkl"% (EXPERIMENT_NAME, epoch), 'w') as f:
        cPickle.dump(losses, f)

'''
import cPickle
with open("../../../results/%s_Params.pkl"%EXPERIMENT_NAME, 'w') as f:
    cPickle.dump(lasagne.layers.get_all_param_values(output_layer_for_loss), f)
with open("../../../results/%s_allLossesNAccur.pkl"%EXPERIMENT_NAME, 'w') as f:
    cPickle.dump([all_training_losses, all_training_accuracies, all_validation_losses, all_validation_accuracies], f)

img_ctr = 0
batch_ctr2 = 0
for data, seg, labels in data_gen_validation:
    pred = get_segmentation(data)
    img_ctr = show_segmentation_results(data, seg, pred, img_ctr=img_ctr)
    batch_ctr2 += 1
    if batch_ctr2 >= 3:
        break'''