def convert_to_binary(input_txt, output_txt):
    with open(input_txt, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        line = line.strip()
        if '|' not in line:
            continue
        filename, num = line.split('|')
        num = int(num.strip())
        if num == 0:
            new_label = 0
        else:
            new_label = 1
        new_lines.append(f"{filename} | {new_label}\n")
    with open(output_txt, 'w') as f:
        f.writelines(new_lines)
    print(f"二分类转换完成，输出文件：{output_txt}")

if __name__ == "__main__":
    input_path = r"E:\mzf-data\002\copd_processed_direct\copd-2.txt"
    output_path = r"E:\mzf-data\002\copd_processed_direct\copd_binary-004004.txt"
    convert_to_binary(input_path, output_path)

