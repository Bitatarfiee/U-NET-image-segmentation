import os
import re
import numpy as np
import nibabel as nib

# Creat Mask including OARs and CTV, and GTV 
# 'OpticNerve_R', 'OpticNerve_L', 'Eye_R', 'Eye_L', 'BrainStem', 'Chiasm', 'CTV', 'GTV' 

# Function to load the list of patient IDs from a text file
def load_patient_ids(file_path):
    with open(file_path, 'r') as file:
        patient_ids = [line.strip() for line in file]
    return patient_ids

# Load the list of patient IDs
patient_ids_file_path = "unique_patient_ids.txt"
patient_ids = load_patient_ids(patient_ids_file_path)

# Base directory where the mask files are stored
base_dir = '/local/data1/bitta693/dataset/'

# List of descriptive names for the mask files
mask_names = ['OpticNerve_R', 'OpticNerve_L', 'Eye_R', 'Eye_L', 'BrainStem', 'Chiasm', 'CTV', 'GTV']

# Function to construct file paths for mask files using a patient ID
def construct_mask_file_paths(base_dir, patient_id):
    mask_files = []
    for name in mask_names:
        mask_file = os.path.join(base_dir, f"mask_{name}_Patient_{patient_id}_defaced_256.nii.gz")
        mask_files.append(mask_file)
    return mask_files

# Function to create combined mask for a patient
def create_combined_mask(patient_id, mask_files):
    combined_mask = None
    grayscale_labels = [50, 50, 60, 60, 70, 80, 100, 150]  # Chosen grayscale values for the masks

    for i, file_path in enumerate(mask_files):
        mask = nib.load(file_path)
        mask_data = mask.get_fdata()
        labeled_mask = mask_data * grayscale_labels[i]  # Assign specific grayscale value

        if combined_mask is None:
            combined_mask = labeled_mask
        else:
            # Combine masks, taking the maximum value at each position
            combined_mask = np.maximum(combined_mask, labeled_mask)

    # Save the combined mask as a new NIfTI file
    combined_nifti_file_path = os.path.join(base_dir, f"Patient_{patient_id}_combined_masks_gray.nii.gz")
    combined_nifti = nib.Nifti1Image(combined_mask, mask.affine)  # Assuming all masks have the same affine transformation
    nib.save(combined_nifti, combined_nifti_file_path)

    print(f"Combined grayscale mask for Patient {patient_id} saved as '{combined_nifti_file_path}'")

# Iterate over each patient ID
for patient_id in patient_ids:
    # Construct file paths for mask files using the patient ID
    mask_files = construct_mask_file_paths(base_dir, patient_id)
    # Create combined mask for the patient
    create_combined_mask(patient_id, mask_files)
