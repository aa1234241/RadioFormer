import os
import numpy as np
import subprocess

validate_ratio = 0.2

all_anno_name = 'train_and_val.txt'
anno_path = '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/labels/'
save_path = '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/labels-new-4/'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

save_name = 'random_split_'
for jj in range(20):
    all_anno = np.loadtxt(os.path.join(anno_path, all_anno_name), dtype=np.str_)
    trains = []
    vals = []
    for i in range(7):
        anno_single_lesion_type = all_anno[all_anno[:, 1] == str(i)]
        num_cases = anno_single_lesion_type.shape[0]
        ind = np.random.permutation(num_cases)
        train_len = int(np.floor(num_cases * (1 - validate_ratio)))
        train_cases = anno_single_lesion_type[ind[:train_len], :]
        val_cases = anno_single_lesion_type[ind[train_len:], :]
        trains.append(train_cases)
        vals.append(val_cases)

    for i, s in enumerate(trains):
        if i == 0:
            all_trains = s
        else:
            all_trains = np.concatenate([all_trains, s], axis=0)
    for i, s in enumerate(vals):
        if i == 0:
            all_vals = s
        else:
            all_vals = np.concatenate([all_vals, s], axis=0)
    np.savetxt(os.path.join(save_path, save_name + 'train_' + str(jj) + '.txt'), all_trains, fmt='%s')
    np.savetxt(os.path.join(save_path, save_name + 'val_' + str(jj) + '.txt'), all_vals, fmt='%s')
    print('training  '+str(jj)+' model....')

    cmd = ['/home/xiaoyubai/anaconda3/envs/sam/bin/python',
           '/media/userdisk0/code/lmmmeng/LLD-MMRI2023/main/train_new.py',
           '--data_dir', '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/images_mycrop_8/',
           '--train_anno_file',
           save_path + save_name + 'train_' + str(
               jj) + '.txt',
           '--val_anno_file',
           save_path + save_name + 'val_' + str(
               jj) + '.txt',
           '--model', 'resnet18m','--batch-size', '4','--validation-batch-size','1',
           '--num-classes','7',
           '--lr', '1e-4', '--warmup-epochs', '5', '--epochs', '300', '--output',
           '/media/xiaoyubai/output_new_2/'
           ]
    # labels-validation.txt save_name + 'val_' + str(jj) + '.txt'
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stderr.decode())

print('test')
