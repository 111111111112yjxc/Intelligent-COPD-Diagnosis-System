import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm

data_root = r"E:\mzf-data\002\copd_processed_direct"
train_txt = r"E:\mzf-data\data004\train.txt"
output_dir = r"E:\mzf-data\train_preprocessed000011"
os.makedirs(output_dir, exist_ok=True)

target_size = (64, 64, 64)

def process_volume(volume):
    if volume.shape != target_size:
        zoom_factors = [t / o for t, o in zip(target_size, volume.shape)]
        volume = zoom(volume, zoom_factors, order=1)
    volume = (volume - volume.mean()) / (volume.std() + 1e-8)
    return volume[np.newaxis]

with open(train_txt, 'r') as f:
    lines = [line.strip() for line in f if '|' in line]

for line in tqdm(lines):
    filename, label = line.split('|')
    filename = filename.strip()
    label = int(label.strip())
    nii_path = os.path.join(data_root, filename)
    img = nib.load(nii_path)
    data = img.get_fdata().astype(np.float32)
    processed = process_volume(data)
    base = filename.replace('.nii.gz', '')
    np.save(os.path.join(output_dir, f"{base}.npy"), processed)