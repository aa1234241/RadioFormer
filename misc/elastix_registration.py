import SimpleITK as sitk
import os
from skimage import morphology
import numpy as np

root_path = '/media/userdisk0/processed_data/lld_mmri/registered-affine/'
save_path = '/media/userdisk0/processed_data/lld_mmri/registered-elastix/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

all_cases = os.listdir(root_path)
# all_cases = [name.split('_',1)[0] for name in all_cases if '0000_mask.nii.gz' in name]
# all_cases = [name.split('_',1)[0] for name in all_cases if '0000.nii.gz' in name]
fix_name = 'C-pre.nii.gz'
moving_list = ['T2WI.nii.gz', 'DWI.nii.gz', 'In Phase.nii.gz', 'Out Phase.nii.gz', 'C+A.nii.gz',
               'C+V.nii.gz', 'C+Delay.nii.gz']
ignores = ['20161669', '05670694']
all_cases = [case for case in all_cases if case not in ignores]
for case in all_cases:
    print(case)
    elastixImageFilter = sitk.ElastixImageFilter()
    fixed = sitk.ReadImage(os.path.join(root_path, case, fix_name))
    fixed_img = sitk.GetArrayFromImage(fixed)
    fixed_img_fore_ground = fixed_img > 0
    fixed_img_fore_ground = fixed_img_fore_ground.astype(np.uint8)
    fixed_foreground_mask = sitk.GetImageFromArray(fixed_img_fore_ground)
    fixed_foreground_mask.CopyInformation(fixed)
    # fix_mask = sitk.ReadImage(os.path.join(root_path,case+'_0000_mask.nii.gz'))
    # fix_mask.CopyInformation(fixed) # this is necessary as I found mask and image head info may different
    moving = sitk.ReadImage(os.path.join(root_path, case, moving_list[0]))
    # get error with moving mask
    # moving_img = sitk.GetArrayFromImage(moving)
    # moving_img_fore_ground = (moving_img>10).astype(np.uint8)
    # moving_img_foreground_mask = sitk.GetImageFromArray(moving_img_fore_ground)
    # moving_img_foreground_mask.CopyInformation(moving)
    # moving_mask = sitk.ReadImage(os.path.join(root_path,case+'_0001_mask.nii.gz'))
    # moving_mask.CopyInformation(moving)
    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetMovingImage(moving)
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("bspline"))
    # elastixImageFilter.SetFixedMask(fixed_foreground_mask)
    # elastixImageFilter.SetMovingMask(moving_img_foreground_mask)
    elastixImageFilter.Execute()
    moved_img = elastixImageFilter.GetResultImage()
    if not os.path.exists(os.path.join(save_path,case)):
        os.mkdir(os.path.join(save_path,case))
    sitk.WriteImage(moved_img, os.path.join(save_path,case,moving_list[0]))
    # transform_parameter_map = elastixImageFilter.GetTransformParameterMap()
    # transformixImageFilter = sitk.TransformixImageFilter()
    # transformixImageFilter.SetMovingImage(moving_mask)
    #
    # transform_parameter_map[0]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    # transformixImageFilter.SetTransformParameterMap(transform_parameter_map)
    # # Perform warp
    # transformixImageFilter.Execute()
    # moved_mask = transformixImageFilter.GetResultImage()
    # sitk.WriteImage(moved_mask, root_path + case + '_0001_rigid_mask_elastix.nii.gz')
    # sitk.WriteImage(fix_mask,root_path+case+'_0000_copy_mask.nii.gz')
