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
     - Python function: crop_patch_using_registered_round(case_path)
     - Input:
       - case_path: the folder that contains multi-modality data
     - Output: a shell script in the case_path named "command_crop_patch.sh"
     - Execuate the shell script "command_crop_patch.sh", ROIs will be cropped according to registered template
    
   - Step 7: Local registration in each ROI

     In local space, another rigid registration is implemented to make the alignment among different modality more accurate.
     - Python function: make_local_registration_command_without_mask(case_path)
     - Input:
       - case_path: the folder that contains multi-modality data
     - Output: a shell script in the case_path named "command_local_registration.sh"
     - Execuate the shell script "command_local_registration.sh", other modalities will be registered to the primary modality (7T-T2) in local space

   - Step 8: Make nnUNet input for inference
  
     nnU-Net requires specific input format for image data. This step will create a new folder and copy the registered ROI into it with required name
     - Python function: make_nnunet_input_folder(data_path)
     - Input:
       - case_path: the folder that contains multi-modality data

2. Model training:
   In the project, there is a python script named "modAugAllFourTrainer.py". Please put it into the nnUNet's package under the folder 'nnunetv2/training/nnUNetTrainer'. Then, when training data is prepared, run the nnUNet training command and set trainer as 'ModalityAugmentTransform'.

   Please remember that the left and right patch for the same subject should be put into the same fold while training.

4. Run nnUNet inference
   Just run the command in the command line:

   "nnUNetv2_predict -i $DATAPATH/nnunet/input -o $DATAPATH/nnunet/output -d 600 -c 3d_fullres -tr ModAugAllFourUNetTrainer", where $DATAPATH is the folder that contains multi-modality data (same as 'case_path' above), the '600' is the model ID. We used 600 in this study. You can change it with any ID when you are training the model.
    
    
      
