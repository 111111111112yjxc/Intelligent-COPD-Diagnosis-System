import os
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from tqdm import tqdm

def resample_and_save_nifti(input_dir, output_nifti_path):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(input_dir)
    if not dicom_files:
        return False
    reader.SetFileNames(dicom_files)
    original_image = reader.Execute()
    original_spacing = original_image.GetSpacing()
    target_spacing = [min(original_spacing)] * 3
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(original_image.GetDirection())
    resampler.SetOutputOrigin(original_image.GetOrigin())
    resampler.SetOutputSpacing(target_spacing)
    original_size = original_image.GetSize()
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in
                zip(original_size, original_spacing, target_spacing)]
    resampler.SetSize(new_size)
    resampled_image = resampler.Execute(original_image)
    array = sitk.GetArrayFromImage(resampled_image)
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(array, affine)
    nib.save(nifti_img, output_nifti_path)
    return True

def process_all_patients(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    patient_dirs = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    for patient in tqdm(patient_dirs):
        input_dir = os.path.join(input_root, patient)
        output_nifti_path = os.path.join(output_root, f"{patient}.nii.gz")
        resample_and_save_nifti(input_dir, output_nifti_path)

if __name__ == "__main__":
    input_root = r"E:\慢阻肺图像数据集\mzf\lung-000"
    output_root = r"E:\mzf-data\000\lung-001001001_nifti"
    process_all_patients(input_root, output_root)