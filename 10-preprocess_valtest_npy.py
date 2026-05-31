import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm

data_root = r"E:\mzf-data\002\copd_processed_direct"
val_txt = r"E:\mzf-data\data004\val.txt"
test_txt = r"E:\mzf-data\data004\test.txt"
output_npy_dir = r"E:\mzf-data\valtest_preprocessed000011"
os.makedirs(output_npy_dir, exist_ok=True)

target_size = (64, 64, 64)

def process_volume(volume):
    if volume.shape != target_size:
        zoom_factors = [t / o for t, o in zip(target_size, volume.shape)]
        volume = zoom(volume, zoom_factors, order=1)
    volume = (volume - volume.mean()) / (volume.std() + 1e-8)
    return volume[np.newaxis]

def preprocess(txt_file, split_name):
    with open(txt_file, 'r') as f:
        lines = [line.strip() for line in f if '|' in line]
    for line in tqdm(lines, desc=f"预处理 {split_name}"):
        filename, label = line.split('|')
        filename = filename.strip()
        label = int(label.strip())
        nii_path = os.path.join(data_root, filename)
        img = nib.load(nii_path)
        data = img.get_fdata().astype(np.float32)
        processed = process_volume(data)
        npy_name = filename.replace('.nii.gz', '.npy')
        npy_path = os.path.join(output_npy_dir, npy_name)
        np.save(npy_path, processed)
        label_file = os.path.join(output_npy_dir, "valtest_labels.txt")
        with open(label_file, 'a') as lf:
            lf.write(f"{npy_name} | {label}\n")

if __name__ == "__main__":
    preprocess(val_txt, "val")
    preprocess(test_txt, "test")