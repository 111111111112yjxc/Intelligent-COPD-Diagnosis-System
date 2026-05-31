import random

def split_data(txt_file_path, train_file_path, test_file_path, val_file_path):
    try:
        with open(txt_file_path, 'r') as txt_file:
            lines = [line.strip() for line in txt_file if line.strip() and '|' in line]
        random.shuffle(lines)
        total_lines = len(lines)
        train_lines = int(total_lines * 0.8)
        test_lines = int(total_lines * 0.1)
        val_lines = total_lines - train_lines - test_lines

        train_data = lines[:train_lines]
        test_data = lines[train_lines:train_lines+test_lines]
        val_data = lines[train_lines+test_lines:]

        with open(train_file_path, 'w') as train_file:
            train_file.write('\n'.join(train_data))
        with open(test_file_path, 'w') as test_file:
            test_file.write('\n'.join(test_data))
        with open(val_file_path, 'w') as val_file:
            val_file.write('\n'.join(val_data))

        print(f"划分完成: 训练{len(train_data)}, 测试{len(test_data)}, 验证{len(val_data)}")
    except FileNotFoundError:
        print(f"文件不存在: {txt_file_path}")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    split_data(
        r"E:\mzf-data\002\copd_processed_direct\copd-3.txt",
        r"E:\mzf-data\data004\train00000.txt",
        r"E:\mzf-data\data004\test00000.txt",
        r"E:\mzf-data\data004\val00000.txt"
    )