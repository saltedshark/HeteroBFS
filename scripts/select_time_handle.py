# 该脚本读取文件，然后去除异常值，然后求取相应时间的均值，最后将均值相加并输出

import os
import argparse
import numpy as np

def process_times(input_path):
    output_dir = os.path.dirname(input_path)
    output_path = os.path.join(output_dir, "last_e_select.txt")
    
    # 读取原始数据
    select_times = []
    total_times = []
    current_section = None
    
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "select_time:":
                current_section = "select"
            elif line == "total_time:":
                current_section = "total"
            elif line and current_section:
                try:
                    value = float(line)
                    if current_section == "select":
                        select_times.append(value)
                    elif current_section == "total":
                        total_times.append(value)
                except ValueError:
                    pass

    # IQR 异常值过滤函数
    def filter_iqr(data):
        if len(data) < 4:
            return data  # 数据量过小时不处理
        
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return [x for x in data if lower_bound <= x <= upper_bound]

    # 处理异常值
    filtered_select = filter_iqr(select_times)
    filtered_total = filter_iqr(total_times)

    # 计算均值（处理空数据情况）
    select_avg = np.mean(filtered_select) if filtered_select else 0
    total_avg = np.mean(filtered_total) if filtered_total else 0
    combined_avg = select_avg + total_avg

    # 写入结果文件
    with open(output_path, 'w') as f:
        f.write(f"total_time_average: {combined_avg:.6f}\n")

    print(f"处理完成！结果已保存至 {output_path}")
    print(f"最终结果: {combined_avg:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="时间数据处理脚本")
    parser.add_argument("input_file", help="e_select.txt 文件路径")
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f"错误：文件 {args.input_file} 不存在")
        exit(1)

    process_times(args.input_file)