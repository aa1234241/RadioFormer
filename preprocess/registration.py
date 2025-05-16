import SimpleITK as sitk
import os

from timm.models.layers import to_3tuple
import numpy as np
import torch.nn.functional as F
import torch


origin_path = '/media/xiaoyubai/raw_data/lld_mmri/lldmmri_test_set/classification_dataset/images/'
save_path ='/media/xiaoyubai/raw_data/lld_mmri/lldmmri_test_set/classification_dataset/images_regis_deeds/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
all_cases = os.listdir(origin_path)
fixed_phase = 'C-pre.nii.gz'
all_phases= ['C+A.nii.gz','C+Delay.nii.gz','DWI.nii.gz','In Phase.nii.gz','Out Phase.nii.gz','T2WI.nii.gz','C+V.nii.gz']

def load_nii_file(nii_image):
    image = sitk.ReadImage(nii_image)
    image_array = sitk.GetArrayFromImage(image)
    return image_array


def resize3D(image, size):
    size = to_3tuple(size)
    image = image.astype(np.float32)
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    x = F.interpolate(image, size=size, mode='trilinear', align_corners=True).squeeze(0)
    x = x.cpu().numpy()
    return x


for case in all_cases:
    if not os.path.exists(os.path.join(save_path,case)):
        os.mkdir(os.path.join(save_path,case))
    fixed_image = load_nii_file(os.path.join(origin_path,case,fixed_phase))
    fixed_image.shape
    fixed_image = resize3D(fixed_image,(16,64,64))
    nii = sitk.GetImageFromArray(fixed_image[0,:,:,:])
    sitk.WriteImage(nii, os.path.join(save_path,
                                          case, fixed_phase))
    for phase in all_phases:
        moving_image = load_nii_file(os.path.join(origin_path,case,phase))
        moving_image = resize3D(moving_image, (16,64,64))
        nii = sitk.GetImageFromArray(moving_image[0, :, :, :])
        sitk.WriteImage(nii, os.path.join(save_path,
                                          case, phase))

        deeds_path = '/media/userdisk0/code/DEEDSBCV/'
        cmd = '%s/deedsBCV -F %s/%s -M %s/%s -l 2 -G 4x3 -L 4x3 -Q 2x1 -O %s/%s' % (
            deeds_path, os.path.join(save_path,case),fixed_phase, os.path.join(save_path,case),phase, os.path.join(save_path,case),phase)
        os.system(cmd)
