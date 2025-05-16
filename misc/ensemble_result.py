import json
import os
import numpy as np


all_exams_path = '/media/userdisk0/code/lmmmeng/LLD-MMRI2023/main/output_new/'
all_exams = os.listdir(all_exams_path)

all_exams = [exam for exam in all_exams if os.path.isdir(os.path.join(all_exams_path,exam))]

all_predictions = []
for exam in all_exams:
    # prdiction_single = json.load(open(os.path.join(all_exams_path,exam,'NPUBXY_f1.json'),'r'))
    # all_predictions.append(prdiction_single)
    prdiction_single = json.load(open(os.path.join(all_exams_path, exam, 'NPUBXY_kappa.json'), 'r'))
    all_predictions.append(prdiction_single)
    # prdiction_single = json.load(open(os.path.join(all_exams_path, exam, 'NPUBXY_acc.json'), 'r'))
    # all_predictions.append(prdiction_single)
all_cases = dict()

for prediction_single in all_predictions:
    for case in prediction_single:
        if case['image_id'] not in all_cases.keys():
            all_cases[case['image_id']] = {}
            all_cases[case['image_id']]['prediction'] = []
            all_cases[case['image_id']]['score'] = []
            all_cases[case['image_id']]['prediction'].append(case['prediction'])
            all_cases[case['image_id']]['score'].append(case['score'])
        else:
            all_cases[case['image_id']]['prediction'].append(case['prediction'])
            all_cases[case['image_id']]['score'].append(case['score'])
score_list = []
for case in all_cases:
    score = np.array(all_cases[case]['score'])
    score = score.sum(0)/all_predictions.__len__()
    prediction = np.argmax(score)

    pred_info = {
        'image_id': case,
        'prediction': int(prediction),
        'score': score.tolist(),
    }
    score_list.append(pred_info)

json_data = json.dumps(score_list, indent=4)
save_name = os.path.join(all_exams_path, 'NPUBXY_now'+'.json')
file = open(save_name, 'w')
file.write(json_data)
file.close()
print('test')


print('test')