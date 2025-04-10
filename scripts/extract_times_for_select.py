import os
import argparse

def extract_times(input_path):
    output_dir = os.path.dirname(input_path)
    output_path = os.path.join(output_dir, "e_select.txt")
    
    # 用列表分别存储两类时间值
    select_times = []
    total_times = []
    
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                key_part, value_part = line.split(":", 1)
                key = key_part.strip()
                value = value_part.strip()
                
                # 分类收集数值
                if key == "select_time":
                    select_times.append(value)
                elif key == "total_time":
                    total_times.append(value)
    
    # 写入新格式文件
    with open(output_path, "w") as f_out:
        # 写入 select_time 部分
        if select_times:
            f_out.write("select_time:\n")
            f_out.write("\n".join(select_times) + "\n\n")  # 数值之间换行
        
        # 写入 total_time 部分
        if total_times:
            f_out.write("total_time:\n")
            f_out.write("\n".join(total_times) + "\n")

    print(f"结果已保存至 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取时间数据")
    parser.add_argument("input_file", help="输入文件路径")
    args = parser.parse_args()
    
    if not os.path.isfile(args.input_file):
        print(f"错误：文件 {args.input_file} 不存在")
        exit(1)
        
    extract_times(args.input_file)