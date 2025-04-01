import re
import sys
import os
from chardet import detect  # 需要安装chardet库

def extract_times(input_filename):
    # 检测文件编码
    with open(input_filename, 'rb') as f:
        rawdata = f.read()
        encoding = detect(rawdata)['encoding'] or 'utf-8'

    # 初始化存储数据结构
    has_compile = False
    compile_times = []
    total_times = []
    other_times = []

    # 正则表达式匹配模式
    compile_pattern = re.compile(r'compile_opencl_time is (\d+\.\d+) seconds')
    time_pattern = re.compile(r'^\s*(\w+)\s*:\s*(\d+\.\d+)')

    # 读取输入文件（使用检测到的编码）
    # 忽略无法解码的字符（errors='ignore'）
    with open(input_filename, 'r', encoding=encoding, errors='ignore') as f:
        for line in f:
            # 检查是否为编译时间行
            compile_match = compile_pattern.search(line)
            if compile_match:
                has_compile = True
                compile_times.append(compile_match.group(1))
                continue

            # 检查是否为时间记录行
            time_match = time_pattern.search(line)
            if time_match:
                key = time_match.group(1)
                value = time_match.group(2)
                if key == 'total_time':
                    total_times.append(value)
                else:
                    other_times.append((key, value))

    # 生成输出内容
    output = []
    if has_compile:
        if compile_times:
            output.append('compile_opencl_time:')
            output.extend(compile_times)
        if total_times:
            output.append('total_time:')
            output.extend(total_times)
    else:
        if total_times:
            output.append('total_time:')
            output.extend(total_times)
            # for key, value in other_times:
            #     output.append(f'{key}:{value}')

    # 构造正确的输出文件路径
    output_dir = os.path.dirname(input_filename)
    output_base = 'e_' + os.path.basename(input_filename)
    output_filename = os.path.join(output_dir, output_base)

    # 确保输出目录存在（虽然通常输入文件存在则目录已存在）
    if output_dir != '' and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 写入输出文件
    with open(output_filename, 'w') as f:
        f.write('\n'.join(output))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_times.py <input_file.txt>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    extract_times(input_file)