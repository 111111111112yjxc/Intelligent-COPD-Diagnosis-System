import os
import pandas as pd


def match_labels(patch_dir, excel_path, output_txt):
    df = pd.read_excel(excel_path)
    label_map = df.set_index("img-ID")["Label"].to_dict()

    with open(os.path.join(patch_dir, "copd.txt"), "r") as f:
        patch_files = [line.strip() for line in f if line.strip()]

    with open(output_txt, "w") as f:
        for patch_file in patch_files:
            patient_id = patch_file.replace(".nii.gz", "")
            if patient_id in label_map:
                label = label_map[patient_id]
                f.write(f"{patch_file} | {label}\n")
            else:
                print(f"警告: {patient_id} 未在Excel中找到")

    print(f"标签匹配完成，输出到 {output_txt}")


if __name__ == "__main__":
    match_labels(r"E:\mzf-data\002\copd_processed_direct",
                 r"E:\mzf-data\慢阻肺标注样例.xlsx",
                 r"E:\mzf-data\002\copd_processed_direct\copd-002002.txt")