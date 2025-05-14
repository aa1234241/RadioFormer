import os

import numpy as np
from datasets.transforms import load_nii_file

data_path = '/media/userdisk0/processed_data/lld_mmri/registered-affine/'

all_cases = os.listdir(data_path)

x = 0
y = 0
z = 0
count = 0
for case in all_cases:
    print(count)
    count+=1
    case_pre = load_nii_file(os.path.join(data_path, case, 'C-pre.nii.gz'))
    shape = case_pre.shape
    print(shape)

    if shape[0] > z:
        z = shape[0]

    if shape[1] > y:
        y = shape[1]
    if shape[2] > x:
        x = shape[2]
        cname = case

print(z,y,x,cname)

