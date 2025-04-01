import os
import numpy as np

def detect_outliers_iqr(data):
    """使用IQR方法检测并过滤异常值"""
    if len(data) < 3:  # 数据量太少时不处理
        return data
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    return [x for x in data if lower_bound <= x <= upper_bound]

def process_time_file(input_path):
    """处理时间文件主函数"""
    # 读取并解析原始文件
    time_data = {}
    current_key = None
    
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if ':' in line:  # 检测到新的时间类型
                key = line.split(':')[0].strip()
                time_data[key] = []
                current_key = key
            else:  # 时间数值行
                try:
                    value = float(line)
                    if current_key:
                        time_data[current_key].append(value)
                except ValueError:
                    continue
    
    # 处理异常值
    cleaned_data = {}
    for key, values in time_data.items():
        filtered = detect_outliers_iqr(values)
        if filtered:  # 只保留有数据的条目
            cleaned_data[key] = filtered
    
    # 构建输出路径
    dir_path = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    output_path = os.path.join(dir_path, f"rmo_{base_name}")
    
    # 写入清理后的文件
    with open(output_path, 'w') as f:
        for key in ['compile_opencl_time', 'total_time']:  # 保持固定顺序
            if key in cleaned_data:
                f.write(f"{key}:\n")
                for value in cleaned_data[key]:
                    f.write(f"{value}\n")
                
    print(f"处理完成！已保存清理后的文件至: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python clean_times.py <输入文件路径>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"错误: 文件 {input_file} 不存在")
        sys.exit(1)
    
    process_time_file(input_file)