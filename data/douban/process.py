def replace_comma_with_space(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # 将逗号替换为空格
            modified_line = line.replace(',', ' ')
            # 将修改后的行写入输出文件
            outfile.write(modified_line)

# 示例用法
input_file = "edges.txt"  # 输入文件路径
output_file = "edges_p.txt"  # 输出文件路径
replace_comma_with_space(input_file, output_file)


