import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from collections import OrderedDict
from math import log10, floor
import argparse


def unify_direction(data_path):
    # the target direction is 'RSA'
    command = ''
    case_path = data_path

    # set global path to determine if we should skip certain cases
    file_3t_t1 = 'image_3tt1.nii.gz'
    file_3t_t2 = 'image_3tt2.nii.gz'
    file_7t_t1_inv1 = 'image_7tt1_inv1.nii.gz'
    file_7t_t1_inv2 = 'image_7tt1_inv2.nii.gz'
    file_7t_t2 = 'image_7tt2.nii.gz'
    path_3tt1 = os.path.join(case_path, file_3t_t1)
    path_3tt2 = os.path.join(case_path, file_3t_t2)
    path_7tt2 = os.path.join(case_path, file_7t_t2)
    path_7tt1_inv1 = os.path.join(case_path, file_7t_t1_inv1)
    path_7tt1_inv2 = os.path.join(case_path, file_7t_t1_inv2)

    command = command + 'cd {}'.format(case_path) + ' && '
    command = command + 'pwd' + ' && '

    # 7T T2 should be RSA direction
    command = command + 'c3d image_7tt2.nii.gz -swapdim RSA -info -o image_7tt2.nii.gz' + ' && '

    # 7T T1
    if os.path.exists(path_7tt1_inv1):
        command = command + 'c3d image_7tt1_inv1.nii.gz -swapdim RSA -info -o image_7tt1_inv1.nii.gz' + ' && '
    if os.path.exists(path_7tt1_inv2):
        command = command + 'c3d image_7tt1_inv2.nii.gz -swapdim RSA -info -o image_7tt1_inv2.nii.gz' + ' && '

    # 3T T1 and 3T T2 also should be same
    if os.path.exists(path_3tt2):
        command = command + 'c3d image_3tt2.nii.gz -swapdim RSA -info -o image_3tt2.nii.gz' + ' && '
    if os.path.exists(path_3tt1):
        command = command + 'c3d image_3tt1.nii.gz -swapdim RSA -info -o image_3tt1.nii.gz' + ' && '
        
    final_command = '#!/bin/bash\n' + command.strip(' && ')
    with open(os.path.join(data_path, 'convert_direction.sh'), 'w') as f_:
        f_.write(final_command)


def global_registration(case_path):
    all_command = ''

    file_3t_t1 = 'image_3tt1.nii.gz'
    file_3t_t2 = 'image_3tt2.nii.gz'
    file_7t_t1_inv1 = 'image_7tt1_inv1.nii.gz'
    file_7t_t1_inv2 = 'image_7tt1_inv2.nii.gz'
    file_7t_t2 = 'image_7tt2.nii.gz'

    # set global path to determine if we should skip certain cases
    path_3tt1 = os.path.join(case_path, file_3t_t1)
    path_3tt2 = os.path.join(case_path, file_3t_t2)
    path_7tt2 = os.path.join(case_path, file_7t_t2)
    path_7tt1_inv1 = os.path.join(case_path, file_7t_t1_inv1)
    path_7tt1_inv2 = os.path.join(case_path, file_7t_t1_inv2)
    
    # start to make command
    # 1. ------- 7T T1 to 7T T2 -------
    if os.path.exists(path_7tt1_inv1) and os.path.exists(path_7tt1_inv2):
        command_7tt1_to_7tt2 = 'greedy -d 3 -a -m NMI -i {} {} -i {} {} -dof 6 -o {} -ia-image-centers -n 100x50x10'.format(file_7t_t2, file_7t_t1_inv1, file_7t_t2, file_7t_t1_inv2, 'zmatrix_7t_t1_to_7t_t2.mat')
        command_apply_7tt1_inv1_to_7tt2 = 'greedy -d 3 -rf {} -rm {} {} -r {}'.format(file_7t_t2, file_7t_t1_inv1, 'img_7t_t1_inv1_to_7t_t2.nii.gz', 'zmatrix_7t_t1_to_7t_t2.mat')
        command_apply_7tt1_inv2_to_7tt2 = 'greedy -d 3 -rf {} -rm {} {} -r {}'.format(file_7t_t2, file_7t_t1_inv2, 'img_7t_t1_inv2_to_7t_t2.nii.gz', 'zmatrix_7t_t1_to_7t_t2.mat')
    else:
        command_7tt1_to_7tt2 = 'echo not_exist'
        command_apply_7tt1_inv1_to_7tt2 = 'echo not_exist'
        command_apply_7tt1_inv2_to_7tt2 = 'echo not_exist'

    # 3. ------- 3T T1 to 7T T2 -------
    if os.path.exists(path_3tt1) and os.path.exists(path_3tt2):
        command_3tt1_to_only_7tt2_nmi = 'greedy -d 3 -a -m NMI -i {} {} -dof 6 -o zmatrix_3t_t1_to_7t_t2_only_nmi.mat -ia-image-centers -n 100x50x10'.format(file_7t_t2, file_3t_t1)
        command_apply_3tt2_to_only_7tt2_nmi = 'greedy -d 3 -rf {} -rm {} img_3t_t2_to_7t_t2.nii.gz -r zmatrix_3t_t1_to_7t_t2_only_nmi.mat'.format(file_7t_t2, file_3t_t2)
        command_apply_3tt1_to_only_7tt2_nmi = 'greedy -d 3 -rf {} -rm {} img_3t_t1_to_7t_t2.nii.gz -r zmatrix_3t_t1_to_7t_t2_only_nmi.mat'.format(file_7t_t2, file_3t_t1)
    else:
        command_3tt1_to_only_7tt2_nmi = 'echo not_exist'
        command_apply_3tt2_to_only_7tt2_nmi = 'echo not_exist'
        command_apply_3tt1_to_only_7tt2_nmi = 'echo not_exist'

    # final command
    command_case_folder = 'cd {}'.format(case_path)

    # make case command
    case_command = command_case_folder + ' && ' + 'pwd' + ' && ' + command_7tt1_to_7tt2 + ' && ' + command_apply_7tt1_inv1_to_7tt2 + ' && ' + command_apply_7tt1_inv2_to_7tt2 + ' && ' + command_3tt1_to_only_7tt2_nmi + ' && ' + command_apply_3tt2_to_only_7tt2_nmi + ' && ' + command_apply_3tt1_to_only_7tt2_nmi + ' && '
    all_command = all_command + case_command
    
    final_command = '#!/bin/bash\n' + all_command.strip(' && ')

    with open(os.path.join(case_path, 'global_registration.sh'), 'w') as f_:
        f_.write(final_command)


def trim_neck_for_original_3tt1(case_path):
    shell_folder_path = '/home/liyue7/data/Data/D14_3T_Mix_7T/Clinical_Exp/Rerun/'

    all_command = ''
    source_3tt1_image = os.path.join(case_path, 'image_3tt1.nii.gz')
    target_3tt1_image = os.path.join(case_path, 'image_3tt1_trim_neck.nii.gz')

    command = 'echo {} && ./trim_neck.sh {} {}'.format('start trim', source_3tt1_image, target_3tt1_image)
    all_command = all_command + command + ' && '
    final_command = '#!/bin/bash\n' + 'cd {}'.format(shell_folder_path) + ' && ' + all_command.strip(' && ')

    with open(os.path.join(case_path, 'command_trim_neck.sh'), 'w') as f_:
        f_.write(final_command)


def register_template_to_original_3tt1_trimed(case_path):
    # set template global image path
    template_3tt1 = 'template/template.nii.gz'
    template_round_left = 'template/left_round_in_global_space.nii.gz'
    template_round_right = 'template/right_round_in_global_space.nii.gz'
   
    all_command = ''
    # 1. affine registration from template to current registered 3tt1 (using NCC)
    command_template_to_3tt1_affine_ncc = 'greedy -d 3 -a -m NCC 2x2x2 -i {} {} -o zmatrix_template_to_registered_original_3tt1_trimed_ncc.mat -ia-image-centers -n 100x50x10'.format('image_3tt1_trim_neck.nii.gz', template_3tt1)
    # 2. deformable registration (using the affine results as beginning)
    command_template_to_3tt1_deform_ncc = 'greedy -d 3 -m NCC 2x2x2 -i {} {} -it zmatrix_template_to_registered_original_3tt1_trimed_ncc.mat -o zdeform_template_to_registered_original_3tt1_trimed_ncc.nii.gz -oinv zdeform_inverse_warp.nii.gz -n 100x50x10'.format('image_3tt1_trim_neck.nii.gz', template_3tt1)

    # 3. apply the deformable registration to all template (left first)
    command_apply_to_template = 'greedy -d 3 -rf {} -rm {} template_to_3tt1.nii.gz -r zmatrix_3t_t1_to_7t_t2_only_nmi.mat zdeform_template_to_registered_original_3tt1_trimed_ncc.nii.gz zmatrix_template_to_registered_original_3tt1_trimed_ncc.mat'.format('image_7tt2.nii.gz', template_3tt1)
    command_apply_to_round_left = 'greedy -d 3 -rf {} -rm {} left_temlate_round_to_3tt1.nii.gz -r zmatrix_3t_t1_to_7t_t2_only_nmi.mat zdeform_template_to_registered_original_3tt1_trimed_ncc.nii.gz zmatrix_template_to_registered_original_3tt1_trimed_ncc.mat'.format('image_7tt2.nii.gz', template_round_left)
    command_apply_to_round_right = 'greedy -d 3 -rf {} -rm {} right_temlate_round_to_3tt1.nii.gz -r zmatrix_3t_t1_to_7t_t2_only_nmi.mat zdeform_template_to_registered_original_3tt1_trimed_ncc.nii.gz zmatrix_template_to_registered_original_3tt1_trimed_ncc.mat'.format('image_7tt2.nii.gz', template_round_right)

    # write these commands
    command_case_folder = 'cd {}'.format(case_path)
    case_command = command_case_folder + ' && ' + 'pwd' + ' && ' +\
                command_template_to_3tt1_affine_ncc + ' && ' +\
                command_template_to_3tt1_deform_ncc + ' && ' +\
                command_apply_to_template + ' && ' +\
                command_apply_to_round_left + ' && ' +\
                command_apply_to_round_right + ' && '

    all_command = all_command + case_command
    final_command = '#!/bin/bash\n' + all_command.strip(' && ')

    with open(os.path.join(case_path, 'command_template_registration.sh'), 'w') as f_:
        f_.write(final_command)


def crop_patch_using_registered_round(case_path):
    command = ''
    for side_ in ['left', 'right']:
        command_0 = 'cd {}'.format(case_path)
        command_1 = 'c3d {}_temlate_round_to_3tt1.nii.gz -trim 0vox -o patch_{}_roi.nii.gz'.format(side_, side_)

        command_2 = 'c3d patch_{}_roi.nii.gz image_7tt2.nii.gz -reslice-identity -o patch_{}_7tt2.nii.gz'.format(side_, side_)
        command_3 = 'c3d patch_{}_roi.nii.gz img_7t_t1_inv1_to_7t_t2.nii.gz -reslice-identity -o patch_{}_7tt1_inv1.nii.gz'.format(side_, side_)
        command_4 = 'c3d patch_{}_roi.nii.gz img_7t_t1_inv2_to_7t_t2.nii.gz -reslice-identity -o patch_{}_7tt1_inv2.nii.gz'.format(side_, side_)
        command_5 = 'c3d patch_{}_roi.nii.gz img_3t_t2_to_7t_t2.nii.gz -reslice-identity -o patch_{}_3tt2.nii.gz'.format(side_, side_)
        command_6 = 'c3d patch_{}_roi.nii.gz img_3t_t1_to_7t_t2.nii.gz -reslice-identity -o patch_{}_3tt1.nii.gz'.format(side_, side_)

        command = command + command_0 + ' && ' + command_1 + ' && ' + command_2 + ' && ' + command_3 + ' && ' +\
                  command_4 + ' && ' + command_5 + ' && ' + command_6 + ' && '
    
    final_command = '#!/bin/bash\n' + command.strip(' && ')
    with open(os.path.join(case_path, 'command_crop_patch.sh'), 'w') as f_:
        f_.write(final_command)


def make_local_registration_command_without_mask(case_path):
    command_registration = ''
    command_cd = 'cd {}'.format(case_path)
    command_registration = command_registration + command_cd + ' && '

    for side_ in ['left', 'right']:
        # 1. ------- 7T T1 to 7T T2 -------
        # NCC only inv1
        command_7tt1_to_7tt2 = 'greedy -d 3 -a -m NCC 2x2x2 -i patch_SIDE_7tt2.nii.gz patch_SIDE_7tt1_inv1.nii.gz -dof 6 -o SIDE_zwmatrix_7t_t1_to_7t_t2.mat -ia-identity -n 100x50'.replace('SIDE', side_)

        # apply
        command_apply_7tt1_inv1_to_7tt2 = 'greedy -d 3 -rf patch_SIDE_7tt2.nii.gz -rm patch_SIDE_7tt1_inv1.nii.gz SIDE_patch_7t_t1_inv1_to_7t_t2.nii.gz -r SIDE_zwmatrix_7t_t1_to_7t_t2.mat'.replace('SIDE', side_)
        command_apply_7tt1_inv2_to_7tt2 = 'greedy -d 3 -rf patch_SIDE_7tt2.nii.gz -rm patch_SIDE_7tt1_inv2.nii.gz SIDE_patch_7t_t1_inv2_to_7t_t2.nii.gz -r SIDE_zwmatrix_7t_t1_to_7t_t2.mat'.replace('SIDE', side_)

        # 2. ------- 3T T2 to 7T T2 -------
        command_3tt2_to_7tt2 = 'greedy -d 3 -a -m WNCC 2x2x2 -gm-trim 5x5x5 -i patch_SIDE_7tt2.nii.gz patch_SIDE_3tt2.nii.gz -dof 6 -o SIDE_zwmatrix_3t_t2_to_7t_t2.mat -ia-identity -n 100x50'.replace('SIDE', side_)
        command_3tt1_to_3tt2 = 'greedy -d 3 -a -m NMI -i patch_SIDE_3tt2.nii.gz patch_SIDE_3tt1.nii.gz -dof 6 -o SIDE_zwmatrix_3t_t1_to_3t_t2.mat -ia-identity -n 100x50'.replace('SIDE', side_)

        # apply
        command_apply_3tt2_to_7tt2 = 'greedy -d 3 -rf patch_SIDE_7tt2.nii.gz -rm patch_SIDE_3tt2.nii.gz SIDE_patch_3t_t2_to_7t_t2.nii.gz -r SIDE_zwmatrix_3t_t2_to_7t_t2.mat'.replace('SIDE', side_)
        command_apply_3tt1_to_7tt2 = 'greedy -d 3 -rf patch_SIDE_7tt2.nii.gz -rm patch_SIDE_3tt1.nii.gz SIDE_patch_3t_t1_to_7t_t2.nii.gz -r SIDE_zwmatrix_3t_t2_to_7t_t2.mat SIDE_zwmatrix_3t_t1_to_3t_t2.mat'.replace('SIDE', side_)

        curr_side_command = command_7tt1_to_7tt2 + ' && ' + command_apply_7tt1_inv1_to_7tt2 + ' && ' + command_apply_7tt1_inv2_to_7tt2 + ' && ' + command_3tt2_to_7tt2 + ' && ' + command_3tt1_to_3tt2 + ' && ' + command_apply_3tt2_to_7tt2 + ' && ' + command_apply_3tt1_to_7tt2
        command_registration = command_registration + curr_side_command + ' && '
        
    shell_command = '#!/bin/bash\n' + command_registration.strip(' && ')
    with open(os.path.join(case_path, 'command_local_registration.sh'), 'w') as f_:
        f_.write(shell_command)

def make_nnunet_input_folder(data_path):
    nnunet_path = os.path.join(data_path, 'nnunet')
    input_path = os.path.join(nnunet_path, 'input')
    output_path = os.path.join(nnunet_path, 'output')
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    ii = 1
    for side_ in ['left', 'right']:
        shutil.copyfile(os.path.join(data_path, 'patch_{}_7tt2.nii.gz'.format(side_)), os.path.join(input_path, "MTL_%03.0d_0000.nii.gz" % ii))
        shutil.copyfile(os.path.join(data_path, '{}_patch_7t_t1_inv1_to_7t_t2.nii.gz'.format(side_)), os.path.join(input_path, "MTL_%03.0d_0001.nii.gz" % ii))
        shutil.copyfile(os.path.join(data_path, '{}_patch_7t_t1_inv2_to_7t_t2.nii.gz'.format(side_)), os.path.join(input_path, "MTL_%03.0d_0002.nii.gz" % ii))
        shutil.copyfile(os.path.join(data_path, '{}_patch_3t_t2_to_7t_t2.nii.gz'.format(side_)), os.path.join(input_path, "MTL_%03.0d_0003.nii.gz" % ii))
        shutil.copyfile(os.path.join(data_path, '{}_patch_3t_t1_to_7t_t2.nii.gz'.format(side_)), os.path.join(input_path, "MTL_%03.0d_0004.nii.gz" % ii))
        ii = ii + 1