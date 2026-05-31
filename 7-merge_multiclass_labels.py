def merge_labels(input_txt, output_txt):
    with open(input_txt, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.strip()
        if '|' not in line:
            continue
        parts = line.split('|')
        if len(parts) != 2:
            continue
        filename = parts[0].strip()
        num_str = parts[1].strip()
        try:
            num = int(num_str)
        except ValueError:
            continue
        if num == 4:
            new_num = 3
        else:
            new_num = num
        new_line = f"{filename} | {new_num}\n"
        new_lines.append(new_line)

    with open(output_txt, 'w') as f:
        f.writelines(new_lines)

    print(f"多分类标签合并完成，输出文件：{output_txt}")

if __name__ == "__main__":
    input_path = r"E:\mzf-data\002\copd_processed_direct\copd-2.txt"
    output_path = r"E:\mzf-data\002\copd_processed_direct\copd-003003.txt"
    merge_labels(input_path, output_path)