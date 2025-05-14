import yaml
import os
import pandas as pd
import subprocess
runed_path = '/media/xiaoyubai/output_new_2/'


all_exams = os.listdir(runed_path)
all_exams = [exam for exam in all_exams if os.path.isdir(os.path.join(runed_path, exam))]

for exam in all_exams:
       print(exam+'________start__________________')
       with open(os.path.join(runed_path,exam,'args.yaml'), 'r') as file:
           all_args = yaml.safe_load(file)

       val_fold = all_args['val_anno_file'].split('/',-1)[-1]
       print(val_fold)

       summary = pd.read_csv(os.path.join(runed_path,exam,'summary.csv'))
       print(summary['eval_f1'].max(),summary['eval_kappa'][summary['eval_f1'] == summary['eval_f1'].max()])

       contents = os.listdir(os.path.join(runed_path,exam))
       f1_model_name = [name for name in contents if 'best_kappa' in name][0]
       print(f1_model_name)
       cmd = ['/home/xiaoyubai/anaconda3/envs/sam/bin/python', '/media/userdisk0/code/lmmmeng/LLD-MMRI2023/main/validate_new.py',
              '--data_dir', '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/images_mycrop_8/',
              '--val_anno_file',
              '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/labels/labels-validation.txt',
              '--model', 'resnet18m', '--num-classes','7',
              '--batch-size', '1', '--checkpoint',
              runed_path + exam + '/' + f1_model_name, '--results-dir',
              runed_path + exam + '/', '--score-dir',  runed_path + exam + '/']
       process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
       stdout, stderr = process.communicate()
       print(stdout.decode())
       print(stderr.decode())
       print( print(exam+'________end__________________'))