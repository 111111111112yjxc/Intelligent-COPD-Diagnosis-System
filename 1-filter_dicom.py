import os
import shutil
import pydicom

def is_eligible_dicom(file_path):
    try:
        dicom_file = pydicom.dcmread(file_path, force=False)
        if 'PixelData' not in dicom_file:
            return False
        series_desc = dicom_file.get('SeriesDescription', '').lower()
        if "lung" in series_desc and "1.0" in series_desc:
            return True
        return False
    except Exception as e:
        return False

def process_patient_folder(input_folder, output_base_folder):
    patient_id = os.path.basename(input_folder)
    output_folder = os.path.join(output_base_folder, patient_id)
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if not os.path.isfile(input_path):
            continue

        if is_eligible_dicom(input_path):
            output_path = os.path.join(output_folder, filename)
            shutil.copy2(input_path, output_path)
            print(f"✅ Copied eligible file: {filename} -> {output_folder}")

def process_main_folder(main_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for patient_folder in os.listdir(main_folder):
        input_path = os.path.join(main_folder, patient_folder)
        if not os.path.isdir(input_path):
            continue
        print(f"\nProcessing patient: {patient_folder}")
        process_patient_folder(input_path, output_folder)

def delete_empty_folders(output_folder):
    deleted_count = 0
    print("\n开始清理空文件夹...")
    for patient_folder in os.listdir(output_folder):
        folder_path = os.path.join(output_folder, patient_folder)
        if not os.path.isdir(folder_path):
            continue
        if not os.listdir(folder_path):
            try:
                os.rmdir(folder_path)
                print(f"🗑️ 已删除空文件夹: {folder_path}")
                deleted_count += 1
            except OSError as e:
                print(f"⚠️ 删除失败 [{folder_path}]: {str(e)}")
    print(f"清理完成，共删除 {deleted_count} 个空文件夹")

if __name__ == "__main__":
    main_folder = r"E:\慢阻肺图像数据集\lung"
    output_folder = r"E:\慢阻肺图像数据集\mzf\lung-000111222"

    process_main_folder(main_folder, output_folder)
    print("\n文件处理完成，符合条件的DICOM文件已复制到目标文件夹（未添加.dcm后缀）。")

    delete_empty_folders(output_folder)

    print("\n全部处理完成。")