import os
import numpy as np
import matplotlib.pyplot as plt
import torchio as tio
import json

f = open('/media/xiaoyubai/raw_data/lld_mmri/lldmmri_test_set/labels/Annotation_mycrop.json', 'r')
annos = json.load(f)

all_cases_path = '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/images_mycrop_8_2/'
all_cases = os.listdir(all_cases_path)
save_path = '/media/xiaoyubai/raw_data/lld_mmri/data/visual-cropped_8_2_key/'
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
# all_cases = ['MR203544']

for caseinfo,label in zip(img_list,lab_list):
    case = caseinfo
    # if not label == 3:
    #     continue
    # if case not in all_cases:
    #     continue
    # anno = annos[case]
    # label = anno['lesion']['0']['category']
    phase_images = os.listdir(os.path.join(all_cases_path,case))
    all_phase_name = [name.split('.',1)[0] for name in phase_images]
    image_lists = []
    image_spacings = []
    for image_name in phase_images:
        image = tio.ScalarImage(os.path.join(all_cases_path,case,image_name))
        image_numpy = image.data[0,:,:,:].numpy()
        image_spacing = image.spacing
        image_lists.append(image_numpy)
        image_spacings.append(image_spacing)
    fig, ax = plt.subplots(2, 4, figsize=(40, 20))
    i = 0
    for name,image,spacing in zip(all_phase_name,image_lists,image_spacings):
        ax[i//4, i%4].set_title(name)
        image_slice = image[:,:,image.shape[2]//2]
        # image_slice = image_slice[:,::-1]
        # image_slice = np.transpose(image_slice,(1,0))
        ax[i//4, i%4].imshow(image_slice, cmap='gray', aspect=spacing[2] / spacing[1])
        i = i+1
    plt.savefig(
        f'{save_path}/{case}_{label}.png')
    plt.close()
    print('test')
