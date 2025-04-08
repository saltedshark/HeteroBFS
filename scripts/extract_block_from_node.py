import os
import re
import sys
from typing import List, Tuple

def validate_directory_name(dir_name: str) -> float:
    """目录名验证与解析模块"""
    if not (match := re.match(r'^d(\d+\.\d{2})$', dir_name)):
        raise ValueError(f"无效目录名格式: {dir_name}")
    return float(match.group(1))

def extract_blocks(content: str) -> List[int]:
    """从文件内容提取graph_block值的模块"""
    blocks = list(map(int, re.findall(r'graph_block : (\d+)', content)))
    if not blocks:
        raise ValueError("未找到有效的graph_block数据")
    return blocks

def process_blocks(blocks: List[int]) -> float:
    """数据处理的管道操作：排序 → 去重 → 求平均"""
    sorted_blocks = sorted(blocks)
    unique_blocks = []
    prev = None
    for b in sorted_blocks:
        if b != prev:
            unique_blocks.append(b)
            prev = b
    return sum(unique_blocks) / len(unique_blocks)

def process_directory(d_path: str) -> float:
    """目录处理流水线"""
    d_value = validate_directory_name(os.path.basename(d_path))
    seq_path = os.path.join(d_path, "seq.txt")
    
    try:
        with open(seq_path) as f:
            return process_blocks(extract_blocks(f.read()))
    except FileNotFoundError:
        raise RuntimeError(f"缺少seq.txt文件: {seq_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyzer.py <graph_dir>")
        sys.exit(1)
    
    # 收集并排序结果
    results = []
    for entry in os.scandir(sys.argv[1]):
        if entry.is_dir() and entry.name.startswith('d'):
            try:
                avg = process_directory(entry.path)
                results.append( (validate_directory_name(entry.name), avg) )
            except Exception as e:
                print(f"跳过 {entry.name}: {e}", file=sys.stderr)
    
    # 按d值排序后输出平均值
    for avg in (v[1] for v in sorted(results, key=lambda x: x[0])):
        print(f"{avg:.2f}")

if __name__ == "__main__":
    main()