import os
import sys
import json
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def crop_lesion(data_dir, json_path, save_dir, xy_extension=8, z_extension=1):
    '''
    Args:
        data_dir: path to original dataset
        json_path: path to annotation file
        save_dir: save_dir of classification dataset
        xy_extension: spatail extension when cropping lesion ROI
        z_extension: slice extension when cropping lesion ROI
    '''

    all_phases = ['C-pre.nii.gz', 'C+A.nii.gz', 'C+V.nii.gz', 'C+Delay.nii.gz', 'DWI.nii.gz', 'In Phase.nii.gz',
                  'Out Phase.nii.gz',
                  'T2WI.nii.gz']

    all_m_cases = ['MR94719']
    #MR123583 97,258,31,124,284,36 DWI 109,285,36  T2WI 111,274,34
    #MR208363 145,165,32,171,188,35 DWI 107,177,30 In Phase 118,170,31 Out Phase 118,170,32 T2WI 101,175,30
    #MR132907 DWI 132,225,38 In Phase 133,197,38 Out Phase 133,197,38 T2WI 130,222,38
    #MR172667 200,150,51,251,203,56
    f = open(json_path, 'r')
    data = json.load(f)
    data = data['Annotation_info']
    for patientID in tqdm(data):
        if patientID not in all_m_cases:
            continue
        for item in data[patientID]:
            phase = item['phase']
            print(phase)
            if not phase == 'C-pre':
                continue
            # spacing = item['pixel_spacing']
            # slice_thickness = item['slice_thickness']
            # src_spacing = (slice_thickness, spacing[0], spacing[1])
            annotation = item['annotation']['lesion']

        for phase in all_phases:
            print(phase)
            image_path = os.path.join(data_dir, patientID, phase)
            try:
                image = sitk.ReadImage(image_path)
            except KeyboardInterrupt:
                exit()
            except:
                print(sys.exc_info())
                print('Countine Processing')
                continue

            image_array = sitk.GetArrayFromImage(image)
            z_shape = image_array.shape[0]
            for ann_idx in annotation:
                ann = annotation[ann_idx]
                lesion_cls = ann['category']
                bbox_info = ann['bbox']['3D_box']

                x_min_pre = int(bbox_info['x_min'])
                y_min_pre = int(bbox_info['y_min'])
                x_max_pre = int(bbox_info['x_max'])
                y_max_pre = int(bbox_info['y_max'])
                z_min_pre = int(bbox_info['z_min'])
                z_max_pre = int(bbox_info['z_max'])
                # bbox = (x_min, y_min, z_min, x_max, y_max, z_max)
                z_tmp_min = z_shape - z_max_pre
                z_tmp_max = z_shape - z_min_pre
                z_min_pre = z_tmp_min
                z_max_pre = z_tmp_max


                # x_min_pre = 200
                # y_min_pre = 150
                # z_min_pre = 51
                # x_max_pre = 251
                # y_max_pre = 203
                # z_max_pre = 56
                print(x_min_pre, y_min_pre, z_min_pre, x_max_pre, y_max_pre, z_max_pre)

                temp_image = image_array

                if phase == 'C-pre.nii.gz':
                    x_min = x_min_pre
                    y_min = y_min_pre
                    z_min = z_min_pre
                    x_max = x_max_pre
                    y_max = y_max_pre
                    z_max = z_max_pre
                else:
                    print(phase)
                    tmp = input('input: (a,b,c)')
                    tmp_c = tmp.split(',',-1)
                    print(tmp_c)
                    x_min = int(tmp_c[0])
                    y_min = int(tmp_c[1])
                    z_min = int(tmp_c[2])
                    x_max = x_min + x_max_pre - x_min_pre
                    y_max = y_min + y_max_pre - y_min_pre
                    z_max = z_min + z_max_pre - z_min_pre

                if xy_extension is not None:
                    x_padding_min = int(abs(x_min - xy_extension)) if x_min - xy_extension < 0 else 0
                    y_padding_min = int(abs(y_min - xy_extension)) if y_min - xy_extension < 0 else 0
                    x_padding_max = int(abs(x_max + xy_extension - temp_image.shape[1])) if x_max + xy_extension > \
                                                                                            temp_image.shape[1] else 0
                    y_padding_max = int(abs(y_max + xy_extension - temp_image.shape[2])) if y_max + xy_extension > \
                                                                                            temp_image.shape[2] else 0

                    x_min = max(x_min - xy_extension, 0)
                    y_min = max(y_min - xy_extension, 0)
                    x_max = min(x_max + xy_extension, temp_image.shape[1])
                    y_max = min(y_max + xy_extension, temp_image.shape[2])
                if z_extension is not None:
                    z_min = max(z_min - z_extension, 0)
                    z_max = min(z_max + z_extension, temp_image.shape[0])

                if temp_image.shape[0] == 1:
                    roi = temp_image[0, y_min:y_max, x_min:x_max]
                    roi = np.expand_dims(roi, axis=0)
                elif z_min == z_max:
                    roi = temp_image[z_min, y_min:y_max, x_min:x_max]
                    roi = np.expand_dims(roi, axis=0)
                else:
                    roi = temp_image[z_min:(z_max + 1), y_min:y_max, x_min:x_max]

                if xy_extension is not None:
                    roi = np.pad(roi, ((0, 0), (y_padding_min, y_padding_max), (x_padding_min, x_padding_max)),
                                 'constant')

                nii_file = sitk.GetImageFromArray(roi)
                if int(ann_idx) == 0:
                    save_folder = os.path.join(save_dir, f'{patientID}')
                else:
                    save_folder = os.path.join(save_dir, f'{patientID}_{ann_idx}')
                os.makedirs(save_folder, exist_ok=True)
                sitk.WriteImage(nii_file, save_folder + f'/{phase}.nii.gz')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Data preprocessing Config', add_help=False)
    parser.add_argument('--data-dir', default='', type=str)
    parser.add_argument('--anno-path', default='', type=str)
    parser.add_argument('--save-dir', default='', type=str)
    args = parser.parse_args()
    data_dir = '/media/userdisk0/processed_data/lld_mmri/registered-new/'
    anno_path = '/media/xiaoyubai/raw_data/lld_mmri/data/labels/Annotation.json'
    save_dir = '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/image_manual_crop/'
    crop_lesion(data_dir, anno_path, save_dir)
