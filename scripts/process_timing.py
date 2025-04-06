import os
import sys

def process_file(input_path):
    # 解析输出文件路径
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    output_name = 'last_' + base_name
    output_path = os.path.join(dir_name, output_name)

    # 初始化时间变量
    compile_time = None
    total_time = None

    # 读取并解析输入文件
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('compile_opencl_time_average: '):
                compile_time = float(line.split(': ')[1])
            elif line.startswith('total_time_average: '):
                total_time = float(line.split(': ')[1])

    # 确保至少存在total_time_average
    if total_time is None:
        raise ValueError("输入文件必须包含total_time_average")

    # 计算总和
    sum_time = total_time + compile_time if compile_time is not None else total_time

    # 写入输出文件
    with open(output_path, 'w') as f:
        f.write(f'total_time_average: {sum_time:.6f}\n')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法：python script.py <输入文件路径>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not os.path.isfile(input_file):
        print(f"错误：文件 '{input_file}' 不存在")
        sys.exit(1)
    
    try:
        process_file(input_file)
    except Exception as e:
        print(f"处理文件时发生错误：{e}")
        sys.exit(1)