import os
all_folder_path = '/media/xiaoyubai/output_new_2/'
all_folder = os.listdir(all_folder_path)

all_folder = [folder for folder in all_folder if os.path.isdir(os.path.join(all_folder_path,folder))]

for folder in all_folder:
    all_files = os.listdir(os.path.join(all_folder_path,folder))
    precison_files = [file for file in all_files if 'loss' in file]
    recall_files = [file for file in all_files if 'recall' in file]
    ckpt_files = [file for file in all_files if 'checkpoint' in file and 'best' not in file]
    # f1_files = [file for file in all_files if 'f1' in file]
    # kappa_files = [file for file in all_files if 'kappa' in file]

    delete_files = []
    for file in precison_files:
        delete_files.append(file)
    for file in recall_files:
        delete_files.append(file)
    for file in ckpt_files:
        delete_files.append(file)
    # for file in f1_files:
    #     delete_files.append(file)
    # for file in kappa_files:
    #     delete_files.append(file)
    for file in delete_files:
        cmd = 'rm -rf %s/%s' %(os.path.join(all_folder_path,folder),file)
        os.system(cmd)
    print('test')
