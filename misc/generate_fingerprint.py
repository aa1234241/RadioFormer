import os
import torchio as tio
import numpy as np
import matplotlib.pyplot as plt

all_cropped_path = '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/images_mycrop_8/'
label_path = '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/labels/'
figure_save_path = '/media/xiaoyubai/raw_data/lld_mmri/data/figure/'
if not os.path.isdir(figure_save_path):
    os.mkdir(figure_save_path)
all_anno = np.loadtxt(os.path.join(label_path,'train_all.txt'), dtype=np.str_)
all_cases = os.listdir(all_cropped_path)
all_validate_cases = np.loadtxt(os.path.join(label_path,'labels-validation.txt'), dtype=np.str_)
all_validate_shape = []
all_validate_label = []
all_validate_volume = []

for case in all_validate_cases:
    case_name = case[0]
    case_label = case[1]
    t2 = tio.ScalarImage(os.path.join(all_cropped_path, case_name, 'T2WI.nii.gz'))
    # print(t2.shape[1:],case_label)
    all_validate_shape.append(t2.shape[1:])
    all_validate_label.append(int(case_label))
    all_validate_volume.append(t2.shape[1]*t2.shape[2]*t2.shape[3])
all_validate_shape = np.array(all_validate_shape)
all_validate_label = np.array(all_validate_label)
all_validate_volume = np.array(all_validate_volume)

for i in range(7):
    plt.hist(all_validate_volume[all_validate_label==i],bins=20)
    plt.title(i)
    plt.savefig(os.path.join(figure_save_path,str(i)+'val_all.png'))
    plt.close()
    # print(i,all_validate_shape[all_validate_label==i,:].mean(axis=0))
    # print(i,all_validate_shape[all_validate_label == i, :].min(axis=0))



# all_shape = []
# for case in all_cases:
#     t2 = tio.ScalarImage(os.path.join(all_cropped_path,case,'T2WI.nii.gz'))
#     all_shape.append(t2.shape[1:])
# all_shape = np.array(all_shape)
# print(all_shape)
# print(all_shape.mean(axis=0))
# print(all_shape.std(axis=0))
# print(all_shape.max(axis=0))
# print(all_shape.min(axis=0))
