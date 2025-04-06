import sys
import os

def parse_time_data(filepath):
    """解析时间数据文件，提取compile_opencl_time和total_time"""
    data = {
        'compile_opencl_time': [],
        'total_time': []
    }
    current_key = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # 检测关键字行
            if line == 'compile_opencl_time:':
                current_key = 'compile_opencl_time'
            elif line == 'total_time:':
                current_key = 'total_time'
            # 处理数据行
            elif current_key is not None and line:
                try:
                    value = float(line)
                    data[current_key].append(value)
                except ValueError:
                    continue  # 忽略无效数据行
    return data

def calculate_averages(data):
    """计算各时间数据的平均值"""
    averages = {}
    for key in ['compile_opencl_time', 'total_time']:
        if data[key]:
            averages[key] = sum(data[key]) / len(data[key])
    return averages

def write_averages(original_path, averages):
    """将平均值写入目标文件"""
    # 构建新文件路径
    directory = os.path.dirname(original_path)
    filename = 's_' + os.path.basename(original_path)
    output_path = os.path.join(directory, filename)
    
    # 写入格式化结果
    with open(output_path, 'w') as f:
        for key in ['compile_opencl_time', 'total_time']:
            if key in averages:
                f.write(f"{key}_average: {averages[key]:.6f}\n")
    return output_path

def main():
    if len(sys.argv) != 2:
        print("使用方法: python script.py <目标文件路径>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not os.path.isfile(input_file):
        print(f"错误: 文件 {input_file} 不存在")
        sys.exit(1)
    
    # 处理数据
    time_data = parse_time_data(input_file)
    avg_results = calculate_averages(time_data)
    output_file = write_averages(input_file, avg_results)
    
    print(f"统计结果已保存至: {output_file}")

if __name__ == "__main__":
    main()