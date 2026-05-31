import os
from tqdm import tqdm

def generate_file_list(target_dir, list_name="copd000001.txt"):
    files = [f for f in os.listdir(target_dir) if f.endswith('.nii.gz')]
    files.sort()
    list_path = os.path.join(target_dir, list_name)
    with open(list_path, "w", encoding="utf-8") as f:
        for filename in files:
            f.write(filename + "\n")
    print(f"文件列表已生成：{list_path}")

if __name__ == "__main__":
    dataset_directory = r"E:\mzf-data\002\copd_processed_direct"
    generate_file_list(dataset_directory)
    print("文件列表生成完成，无需复制。")