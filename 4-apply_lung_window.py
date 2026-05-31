import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from tqdm import tqdm

class COPDPreprocessor:
    def __init__(self, input_nifti_path, output_dir):
        self.input_nifti_path = input_nifti_path
        self.output_dir = output_dir
        self.image = None

    def load_volume(self):
        img = nib.load(self.input_nifti_path)
        data = img.get_fdata().astype(np.float32)
        self.image = sitk.GetImageFromArray(data)
        return self

    def apply_copd_window(self):
        min_val = -1000
        max_val = 500
        self.image = sitk.IntensityWindowing(
            self.image,
            windowMinimum=min_val,
            windowMaximum=max_val,
            outputMinimum=0.0,
            outputMaximum=1.0
        )
        return self

    def save_results(self):
        os.makedirs(self.output_dir, exist_ok=True)
        base_name = os.path.basename(self.input_nifti_path).replace('.nii.gz', '')
        output_path = os.path.join(self.output_dir, f"{base_name}.nii.gz")
        arr = sitk.GetArrayFromImage(self.image)
        nib.save(nib.Nifti1Image(arr, np.eye(4)), output_path)
        print(f"已保存: {output_path}")

    def process(self):
        self.load_volume()
        self.apply_copd_window()
        self.save_results()

def process_all(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    nifti_files = [f for f in os.listdir(input_root) if f.endswith('.nii.gz')]
    for fname in tqdm(nifti_files, desc="处理患者"):
        input_path = os.path.join(input_root, fname)
        processor = COPDPreprocessor(input_path, output_root)
        processor.process()

if __name__ == "__main__":
    input_root = r"E:\mzf-data\001\lung-002_nifti"
    output_root = r"E:\mzf-data\002\copd_processed_direct000001"
    process_all(input_root, output_root)