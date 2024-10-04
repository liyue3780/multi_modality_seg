This is a step-by-step introduction of the code
- Environmental requirements:\
  This code uses Convert3D and greedy tools frequently. They are embedded in ITK-SNAP (Linux version). Users need to install ITK-SNAP before running the pipeline of this study.
  
1. Preprocess
   - Step 1: Unify data format
     
     For one subject with multi-modality data, these modalities should be named as follow:
     - image_3tt1.nii.gz
     - image_3tt2.nii.gz
     - image_7tt1_inv1.nii.gz
     - image_7tt1_inv2.nii.gz
     - image_7tt2.nii.gz
       
   - Step 2: Unify image direction
     
     In order to finish the following registration, cropping, and flipping, we unify the direction of all images. We used RSA direction in this study.
     - Python function: unify_direction(data_path)
     - Input: data_path: the folder that contains multi-modality data
     - Output: a shell script named "convert_direction.sh"
     - Execuate the shell script "convert_direction.sh", images' directions will be unified to RSA
       
   - Step 3: Whole-brain registration
     
     This is a coarse registration globally.
     - Python function: global_registration(case_path)
     - Input: case_path: the folder that contains multi-modality data
     - Output: a shell script in the case_path named "global_registration.sh"
     - Execuate the shell script "global_registration.sh", whole-brain registration will be implemented

  - Step 4: Trim neck for 3T-T1w image
    
    Because the 3T-T1w template does not contain the neck, we need to trim the neck region from our 3T-T1w data to reduce the registration noisy.
    - Python function: trim_neck_for_original_3tt1(case_path)
    - Input: case_path: the folder that contains multi-modality data
    - Output: a shell script in the case_path named "command_trim_neck.sh"
    - Execuate the shell script "command_trim_neck.sh", 3T-T1w image neck-trimming will be implemented

  - Step 5: Template registration (from 3T-T1w template to individule 3T-T1w image)
    
    - Python function: register_template_to_original_3tt1_trimed(case_path)
    - Input:
      - case_path: the folder that contains multi-modality data
      
      Three specific paths should be set in the function
      - template_3tt1: the path of template of 3T-T1w image
      - template_round_left: the binary left ROI in whole-brain image space
      - template_round_right: the binary right ROI in whole-brain image space
     
    - Output: a shell script in the case_path named "command_template_registration.sh"
    - Execuate the shell script "command_template_registration.sh", template registration will be implemented

  - Step 6: Cropping the ROI

    After registering ROI from template to individule space, we know where the MTL region is. This step crops the whole-brain image into left and right patch
    # here, here
    
      
