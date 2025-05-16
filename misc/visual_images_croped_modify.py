import os
import numpy as np
import matplotlib.pyplot as plt
import torchio as tio
import json
from datasets.transforms import resize3D,load_nii_file
import tqdm
f = open('/media/xiaoyubai/raw_data/lld_mmri/lldmmri_test_set/labels/Annotation_mycrop.json', 'r')
annos = json.load(f)

all_cases_path = '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/images_mycrop_8_2/'
all_cases = os.listdir(all_cases_path)
save_path = '/media/xiaoyubai/raw_data/lld_mmri/data/visual-cropped_8_2_new_full/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

label_path = '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/labels/'
all_train_cases = np.loadtxt(os.path.join(label_path,'train_all.txt'), dtype=np.str_)


img_list = []
lab_list = []
for item in all_train_cases:
    img_list.append(item[0])
    lab_list.append(int(item[1]))


# all_cases = ['MR198335', 'MR220058', 'MR112315', 'MR230228', 'MR227553', 'MR208089', 'MR196089', 'MR194843',
#                  'MR221970', 'MR156441', 'MR207486']

# all_cases = ['MR156441','MR193842','MR196089','MR200585','MR202035','MR224866','MR235786','MR236008','MR237152']
# all_cases = ['MR94719']

for caseinfo,label in tqdm.tqdm(zip(img_list,lab_list)):
    case = caseinfo
    # if not label == 3:
    #     continue
    if case not in all_cases:
        continue
    # anno = annos[case]
    # label = anno['lesion']['0']['category']
    # phase_images = os.listdir(os.path.join(all_cases_path,case))

    phase_list = ['T2WI', 'DWI', 'In Phase', 'Out Phase',
                  'C-pre', 'C+A', 'C+V', 'C+Delay']
    # all_phase_name = [name.split('.',1)[0] for name in phase_images]
    image_lists = []
    image_spacings = []
    mp_image = []
    for image_name in phase_list:
        image = load_nii_file(os.path.join(all_cases_path,case,image_name+'.nii.gz'))
        mp_image.append(image[None, ...])
    mp_image = np.concatenate(mp_image, axis=0)
    mp_image = resize3D(mp_image,(16,128,128))


    for j in range(16):
        i = 0
        fig, ax = plt.subplots(2, 4, figsize=(40, 20))
        for name in phase_list:
            ax[i//4, i%4].set_title(name)
            image_slice = mp_image[i,j,:,:]
            # image_slice = image_slice[:,::-1]
            # image_slice = np.transpose(image_slice,(1,0))
            ax[i//4, i%4].imshow(image_slice, cmap='gray')
            i = i+1
        plt.savefig(
            f'{save_path}/{case}_{label}_{j}.png')
        plt.close()

