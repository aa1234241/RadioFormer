import json
import os
import numpy as np
from metrics import ACC, Recall, F1_score, Cohen_Kappa, Precision, confusion_matrix, cls_report

runed_path = '/media/xiaoyubai/output_new_2/'

predicion_file = 'NPUBXY_now.json'

val_anno_file = '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/labels/labels-validation.txt'

prediction = json.load(open(os.path.join(runed_path, predicion_file), 'r'))

pre_dict = {}
for pre in prediction:
    pre_dict[pre['image_id']] = pre['score']
anno = np.loadtxt(val_anno_file, dtype=np.str_)
img_list = []
lab_list = []
for item in anno:
    img_list.append(item[0])
    # if item[1] in ['1','3','6']:
    #     item[1] = 1
    # else:
    #     item[1] = 0

    lab_list.append(item[1])

prediction_list = []
for img_name in img_list:
    prediction_list.append(pre_dict[img_name])

prediction_list = np.array(prediction_list)
lab_list = [int(label) for label in lab_list]
lab_list = np.array(lab_list)

for i, img_name in enumerate(img_list):
    if not np.argmax(prediction_list[i]) == lab_list[i]:
        print(img_name, np.argmax(prediction_list[i]), lab_list[i])

acc = ACC(prediction_list, lab_list)
f1 = F1_score(prediction_list, lab_list)
kappa = Cohen_Kappa(prediction_list, lab_list)
precision = Precision(prediction_list, lab_list)
confu = confusion_matrix(prediction_list, lab_list)
report = cls_report(prediction_list, lab_list)

print('acc:', acc, 'f1:', f1, 'kappa:', kappa, '\n', confu)
print(report)
print('test')
