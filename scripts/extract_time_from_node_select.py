import os
import re
import sys

def extract_d_value(folder_name):
    """从文件夹名提取d后的数值并转换为浮点数"""
    match = re.match(r'^d(\d+\.\d{2})$', folder_name)
    return float(match.group(1)) if match else None

def process_graph_folder(folder_path):
    """处理graph文件夹并输出排序后的时间值"""
    dirs = []
    filename = f"last_e_select.txt"
    
    # 遍历文件夹收集有效目录
    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)
        if os.path.isdir(entry_path):
            d_value = extract_d_value(entry)
            if d_value is not None:
                dirs.append((d_value, entry_path))

    # 按d值排序
    dirs.sort(key=lambda x: x[0])

    # 收集并验证结果
    results = []
    for d_value, path in dirs:
        file_path = os.path.join(path, filename)
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
                if content.startswith('total_time_average: '):
                    results.append(content.split(': ')[1])
        except FileNotFoundError:
            print(f"Warning: 文件 {filename} 未在目录 {path} 中找到", file=sys.stderr)
        except Exception as e:
            print(f"读取 {file_path} 错误: {str(e)}", file=sys.stderr)

    # 输出结果
    for value in results:
        print(value)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python script.py <graph文件夹路径>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    
    if not os.path.exists(input_folder):
        print(f"错误：文件夹 '{input_folder}' 不存在")
        sys.exit(1)
    
    process_graph_folder(input_folder)