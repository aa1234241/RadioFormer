import numpy as np
import os
from datasets.transforms import load_nii_file
from sklearn.metrics import mutual_info_score

small_field_path = '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/images_mycrop_8/'
large_field_path = '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/images_mycrop_8_new/'

label_path = '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/labels/'
all_train_cases = np.loadtxt(os.path.join(label_path, 'labels-validation.txt'), dtype=np.str_)

phase_list = ['T2WI', 'DWI', 'In Phase', 'Out Phase',
              'C-pre', 'C+A', 'C+V', 'C+Delay']

fix_phase = 'C-pre'
phase_list = ['C+A']

train_name = list(all_train_cases[:, 0])

count = 0
for case in train_name:
    flag = 0
    for phase in phase_list:
        small = load_nii_file(os.path.join(small_field_path, case, fix_phase + '.nii.gz'))
        large = load_nii_file(os.path.join(large_field_path, case, phase + '.nii.gz'))
        large_shape = large.shape
        small_shape = small.shape
        common_shape = np.min(
            np.concatenate([np.array(large_shape).reshape(-1, 1), np.array(small_shape).reshape(-1, 1)], axis=1),
            axis=1)
        # mse = large[:common_shape[0], :common_shape[1], :common_shape[2]] - small[:common_shape[0], :common_shape[1],
        #                                                                     :common_shape[2]]

        # mi = mutual_info_score(large[:common_shape[0], :common_shape[1], :common_shape[2]].reshape(-1),
        #                        small[:common_shape[0], :common_shape[1],
        #                        :common_shape[2]].reshape(-1))

        hist, x_edges, y_edges = np.histogram2d(large[:common_shape[0], :common_shape[1], :common_shape[2]].reshape(-1),
                                                small[:common_shape[0], :common_shape[1], :common_shape[2]].reshape(-1),
                                                bins=(20, 20))
        p_xy = hist / np.sum(hist)
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)
        p_xy[p_xy == 0] = 1e-10
        p_x[p_x == 0] = 1e-10
        p_y[p_y == 0] = 1e-10
        mi = np.sum(p_xy * np.log2(p_xy / (np.outer(p_x, p_y))))
        # if mi < 0.6:
        print(case, phase, mi)
        # print(case, phase, mi)
        # if np.abs(mse.mean()) > 10:
        #     flag = 1
        #     print(case, phase, mse.mean())

        if flag == 1:
            count += 1

