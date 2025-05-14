import yaml
import os
import pandas as pd
import subprocess

runed_path = '/media/xiaoyubai/output_new_2/'

all_exams = os.listdir(runed_path)
all_exams = [exam for exam in all_exams if os.path.isdir(os.path.join(runed_path, exam))]
# all_exams = ['20230801-062821-uniformer_small_IL', '20230801-183201-uniformer_small_IL',
#              '20230801-071902-uniformer_small_IL', '20230801-192754-uniformer_small_IL',
#              '20230731-211030-uniformer_small_IL', '20230801-030509-uniformer_small_IL',
#              '20230801-210916-uniformer_small_IL', '20230801-104202-uniformer_small_IL',
#              '20230802-030533-uniformer_small', '20230801-003413-uniformer_small_IL',
#              ]

all_exams = ['20250115-191538-hilverformer_vit_small']

for exam in all_exams:
    print(exam + '________start__________________')
    with open(os.path.join(runed_path, exam, 'args.yaml'), 'r') as file:
        all_args = yaml.safe_load(file)

    val_fold = all_args['val_anno_file'].split('/', -1)[-1]
    train_fold = all_args['train_anno_file'].split('/', -1)[-1]
    num = train_fold.split('_', -1)[-1].split('.', 1)[0]
    print(train_fold)

    print(val_fold)
    # if '2' not in val_fold:
    #        continue
    summary = pd.read_csv(os.path.join(runed_path, exam, 'summary.csv'))
    f1_decrease = sorted(summary['eval_f1'], reverse=False)
    import numpy as np

    for i in range(1):
        print(f1_decrease[-(i + 1)], int(np.where(summary['eval_f1'] == f1_decrease[-(i + 1)])[0][0]))

    # print(summary['eval_f1'].max(),summary['eval_kappa'][summary['eval_f1'] == summary['eval_f1'].max()])

    contents = os.listdir(os.path.join(runed_path, exam))
    f1_model_names = [name for name in contents if 'best_f1' in name]
    for f1_model_name in f1_model_names:
        print(f1_model_name)
        cmd = ['/home/xiaoyubai/anaconda3/envs/sam/bin/python',
               '/media/userdisk0/code/lmmmeng/LLD-MMRI2023/main/validate_new.py',
               '--data_dir',
               '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/images_mycrop_8_new/',
               # '/media/xiaoyubai/raw_data/lld_mmri/lldmmri_test_set/classification_dataset/images_mycrop_8/',
               '--val_anno_file',
               '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/labels/labels_test.txt',
               # '/media/xiaoyubai/raw_data/lld_mmri/lldmmri_test_set/classification_dataset/labels/labels_test.txt',
               '--model', 'hilverformer_vit_small', '--num-classes', '7',
               '--batch-size', '1', '--checkpoint',
               runed_path + exam + '/' + f1_model_name, '--results-dir',
               runed_path + exam + '/', '--score-dir', runed_path + exam + '/']
        # labels-validation  random_split_val_1 uniformer_small_IL
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(stdout.decode())
        print(stderr.decode())
    print(print(exam + '________end__________________'))
