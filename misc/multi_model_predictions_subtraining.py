import os
import cmd
import subprocess

all_exams_path = '/media/xiaoyubai/output_new_2/'
all_exams = os.listdir(all_exams_path)
all_exams = [exam for exam in all_exams if os.path.isdir(os.path.join(all_exams_path, exam))]
# all_exams = ['20240202-104237-hilverformer_vit_small_fold_1_new_regis_m',
#              '20240203-193827-hilverformer_vit_small_fold_2_new_regis_m',
#              '20240203-212145-hilverformer_vit_small_fold_3_new_regis_m',
#              '20240131-013356-hilverformer_vit_small_fold_4_new_regis_m',
#              '20240130-234122-hilverformer_vit_small_fold_5_new_regis_m']
all_exams = ['20240203-234924-hilverformer_vit_small']
metric = 'f1'
for exam in all_exams:
    contents = os.listdir(os.path.join(all_exams_path, exam))
    f1_model_names = [name for name in contents if 'best_' + metric in name]
    for i, f1_model_name in enumerate(f1_model_names):
        print(f1_model_name)
        cmd = ['/home/xiaoyubai/anaconda3/envs/sam/bin/python',
               '/media/userdisk0/code/lmmmeng/LLD-MMRI2023/main/predict_new.py',
               '--data_dir',
               '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/images_mycrop_8_new/',
               '--val_anno_file',
               '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/labels/labels-validation.txt',
               '--model', 'hilverformer_vit_small', '--num-classes', '7',
               '--batch-size', '1', '--checkpoint',
               all_exams_path + exam + '/' + f1_model_name, '--results-dir',
               all_exams_path + exam + '/', '--team_name', 'NPUBXY_' + metric + '_' + str(i)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(stderr.decode())
print('test')
