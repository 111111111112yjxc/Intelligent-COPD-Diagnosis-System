import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

def extract_center_slices_from_nifti(input_nifti_path, output_nifti_path, num_slices=100):
    img = nib.load(input_nifti_path)
    z_size, y_size, x_size = img.shape
    if z_size < num_slices:
        return False
    start = (z_size - num_slices) // 2
    center_data = img.dataobj[start:start+num_slices, :, :].astype(np.float32)
    nib.save(nib.Nifti1Image(center_data, img.affine), output_nifti_path)
    return True

def process_all_patients(input_root, output_root, num_slices=100):
    os.makedirs(output_root, exist_ok=True)
    nifti_files = [f for f in os.listdir(input_root) if f.endswith('.nii.gz')]
    success = 0
    skipped = 0
    for fname in tqdm(nifti_files, desc="提取中心层"):
        input_path = os.path.join(input_root, fname)
        output_path = os.path.join(output_root, fname)
        if os.path.exists(output_path):
            skipped += 1
            success += 1
            continue
        if extract_center_slices_from_nifti(input_path, output_path, num_slices):
            success += 1
    print(f"成功处理 {success}/{len(nifti_files)} 个患者（其中 {skipped} 个已存在跳过）")

if __name__ == "__main__":
    input_root = r"E:\mzf-data\000\lung-001_nifti"
    output_root = r"E:\mzf-data\001\lung-002002002_nifti"
    process_all_patients(input_root, output_root, num_slices=100)