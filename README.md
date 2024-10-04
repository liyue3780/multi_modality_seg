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
     - Execuate the shell script, images' directions will be unified to RSA
   - Step 3: Whole-brain registration
     This is a coarse registration globally.
      
