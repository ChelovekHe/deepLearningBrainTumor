__author__ = 'fabian'
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from dataset_utility import load_patient_with_t1


out_dir = "/media/fabian/DeepLearningData/datasets/Hirntumor_David_raw_highRes"
out_dir_resampled = "/media/fabian/My Book/datasets/Hirntumor_resampled_Highres"

def cut_off_values_upper_lower_percentile(image, percentile_lower=0.2, percentile_upper=99.8):
    cut_off_lower = np.percentile(image[image!=image[0,0,0]].ravel(), percentile_lower)
    cut_off_upper = np.percentile(image[image!=image[0,0,0]].ravel(), percentile_upper)
    image[image < cut_off_lower] = cut_off_lower
    image[image > cut_off_upper] = cut_off_upper
    return image

percentile_target = [0., 100.]

for i in range(999):
    print i
    t1_img, t1km_img, flair_img, adc_img, cbv_img, _, seg_combined = load_patient_with_t1(i, [1., 0.5, 0.5], 'new')
    if t1_img is not None:
        np.save(os.path.join(out_dir_resampled, "%03.0d_t1.npy" % i), t1_img)
        np.save(os.path.join(out_dir_resampled, "%03.0d_t1km.npy" % i), t1km_img)
        np.save(os.path.join(out_dir_resampled, "%03.0d_flair.npy" % i), flair_img)
        np.save(os.path.join(out_dir_resampled, "%03.0d_adc.npy" % i), adc_img)
        np.save(os.path.join(out_dir_resampled, "%03.0d_cbv.npy" % i), cbv_img)
        np.save(os.path.join(out_dir_resampled, "%03.0d_seg.npy" % i), seg_combined)


for i in range(119, 150):
    if not os.path.isfile(os.path.join(out_dir_resampled, "%03.0d_t1.npy" % i)):
        continue

    t1_img = np.load(os.path.join(out_dir_resampled, "%03.0d_t1.npy" % i))
    t1km_img = np.load(os.path.join(out_dir_resampled, "%03.0d_t1km.npy" % i))
    flair_img = np.load(os.path.join(out_dir_resampled, "%03.0d_flair.npy" % i))
    adc_img = np.load(os.path.join(out_dir_resampled, "%03.0d_adc.npy" % i))
    cbv_img = np.load(os.path.join(out_dir_resampled, "%03.0d_cbv.npy" % i))
    seg_combined = np.load(os.path.join(out_dir_resampled, "%03.0d_seg.npy" % i))

    if t1_img is not None:
        lower_percentile = 20.
        upper_percentile = 80.

        percentile_actual = [np.percentile(t1_img[t1_img!=t1_img[0,0,0]].ravel(), lower_percentile), np.percentile(t1_img[t1_img!=t1_img[0,0,0]].ravel(), upper_percentile)]
        t1_img -= percentile_actual[0]
        t1_img *= (percentile_target[1] - percentile_target[0]) / (percentile_actual[1] - percentile_actual[0])
        t1_img = cut_off_values_upper_lower_percentile(t1_img, 0.1, 99.9)

        percentile_actual = [np.percentile(t1km_img[t1km_img!=t1km_img[0,0,0]].ravel(), lower_percentile), np.percentile(t1km_img[t1km_img!=t1km_img[0,0,0]].ravel(), upper_percentile)]
        t1km_img -= percentile_actual[0]
        t1km_img *= (percentile_target[1] - percentile_target[0]) / (percentile_actual[1] - percentile_actual[0])
        t1km_img = cut_off_values_upper_lower_percentile(t1km_img, 0.1, 99.9)

        percentile_actual = [np.percentile(flair_img[flair_img!=flair_img[0,0,0]].ravel(), lower_percentile), np.percentile(flair_img[flair_img!=flair_img[0,0,0]].ravel(), upper_percentile)]
        flair_img -= percentile_actual[0]
        flair_img *= (percentile_target[1] - percentile_target[0]) / (percentile_actual[1] - percentile_actual[0])
        flair_img = cut_off_values_upper_lower_percentile(flair_img, 0.1, 99.9)

        percentile_actual = [np.percentile(adc_img[adc_img!=adc_img[0,0,0]].ravel(), lower_percentile), np.percentile(adc_img[adc_img!=adc_img[0,0,0]].ravel(), upper_percentile)]
        adc_img -= percentile_actual[0]
        adc_img *= (percentile_target[1] - percentile_target[0]) / (percentile_actual[1] - percentile_actual[0])
        adc_img = cut_off_values_upper_lower_percentile(adc_img, 0.1, 99.9)

        percentile_actual = [np.percentile(cbv_img[cbv_img!=cbv_img[0,0,0]].ravel(), lower_percentile), np.percentile(cbv_img[cbv_img!=cbv_img[0,0,0]].ravel(), upper_percentile)]
        cbv_img -= percentile_actual[0]
        cbv_img *= (percentile_target[1] - percentile_target[0]) / (percentile_actual[1] - percentile_actual[0])
        cbv_img = cut_off_values_upper_lower_percentile(cbv_img, 0.1, 99.9)

        all_data = np.zeros([6] + list(t1_img.shape), dtype=np.float32)
        all_data[0] = t1_img
        all_data[1] = t1km_img
        all_data[2] = flair_img
        all_data[3] = adc_img
        all_data[4] = cbv_img
        all_data[5] = seg_combined
        np.save(os.path.join(out_dir, "%03.0d.npy"%i), all_data)

        xlim = (-150, 250)
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 5, 1)
        plt.hist(t1_img[t1_img!=t1_img[0,0,0]], bins=100)
        plt.xlim(xlim)
        plt.title("t1")
        plt.subplot(1, 5, 2)
        plt.hist(t1km_img[t1km_img!=t1km_img[0,0,0]], bins=100)
        plt.xlim(xlim)
        plt.title("t1km")
        plt.subplot(1, 5, 3)
        plt.hist(flair_img[flair_img!=flair_img[0,0,0]], bins=100)
        plt.xlim(xlim)
        plt.title("flair")
        plt.subplot(1, 5, 4)
        plt.hist(adc_img[adc_img!=adc_img[0,0,0]], bins=100)
        plt.xlim(xlim)
        plt.title("adc")
        plt.subplot(1, 5, 5)
        plt.hist(cbv_img[cbv_img!=cbv_img[0,0,0]], bins=100)
        plt.xlim(xlim)
        plt.title("cbv")
        plt.savefig(os.path.join(out_dir, "%03.0d_hist.png"%i))
        plt.close()

