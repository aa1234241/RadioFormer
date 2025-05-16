import numpy as np
import os

label_path = '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/labels/'
all_train_cases = np.loadtxt(os.path.join(label_path, 'train_all.txt'), dtype=np.str_)

train_fold5_cases = np.loadtxt(os.path.join(label_path, 'train_fold5.txt'), dtype=np.str_)
val_fold5_cases = np.loadtxt(os.path.join(label_path, 'val_fold5.txt'), dtype=np.str_)


train_fold1_cases = np.loadtxt(os.path.join(label_path, 'train_fold1.txt'), dtype=np.str_)
val_fold1_cases = np.loadtxt(os.path.join(label_path, 'val_fold1.txt'), dtype=np.str_)


train_fold5_name = train_fold5_cases[:,0]
train_fold1_name = train_fold1_cases[:,0]
val_fold5_name = val_fold5_cases[:,0]
val_fold1_name = val_fold1_cases[:,0]

train_fold5_name_set = set(train_fold5_name)
train_fold1_name_set = set(train_fold1_name)
val_fold5_name_set = set(val_fold5_name)
val_fold1_name_set = set(val_fold1_name)
print('test')
