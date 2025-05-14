import os
import cmd
import subprocess

all_exams_path = '/media/userdisk0/code/lmmmeng/LLD-MMRI2023/main/output_new_1/'
all_exams = os.listdir(all_exams_path)
all_exams = [exam for exam in all_exams if os.path.isdir(os.path.join(all_exams_path, exam)) ]

metric = 'kappa'
for exam in all_exams:
    contents = os.listdir(os.path.join(all_exams_path, exam))
    f1_model_name = [name for name in contents if 'best_'+metric in name][0]
    print(f1_model_name)
    cmd = ['/home/xiaoyubai/anaconda3/envs/sam/bin/python','/media/userdisk0/code/lmmmeng/LLD-MMRI2023/main/predict_new.py',
           '--data_dir','/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/images_mycrop_4/',
           '--val_anno_file', '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/labels/labels_val_inaccessible.txt',
           '--model', 'uniformer_small_IL','--num-classes','7',
           '--batch-size', '1', '--checkpoint',
           all_exams_path+exam+'/'+f1_model_name, '--results-dir',
           all_exams_path+exam+'/', '--team_name', 'NPUBXY_'+metric]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stderr.decode())
print('test')
