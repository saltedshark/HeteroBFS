# 脚本功能说明：用于验证csr生成程序的有效性
# 用法：python3 varify_csr.py --file | -f filename
import struct
import argparse
import array
import sys
import os

def inspect_csr_file(filename):
    """验证Gunrock兼容的CSR二进制文件"""
    try:
        with open(filename, "rb") as f:
            # 读取文件头（16字节）
            header_data = f.read(16)
            if len(header_data) < 16:
                raise ValueError("文件头不完整")

            # 解析头信息
            magic, nodes, edges, _ = struct.unpack("<IIII", header_data)
            
            # 获取文件总大小并校验
            f.seek(0, os.SEEK_END)  # 移动到文件末尾
            file_size = f.tell()
            expected_size = 16 + 4 * (nodes + 1) + 4 * edges
            if file_size != expected_size:
                raise ValueError(
                    f"文件大小不匹配，预期 {expected_size} 字节，实际 {file_size} 字节\n"
                    f"可能原因：头信息中的节点数({nodes})或边数({edges})异常"
                )
            f.seek(16)  # 回到数据起始位置

            print(f"验证文件: {filename}")
            print(f"  Magic标识: 0x{magic:08X} {'(有效)' if magic == 0x47535246 else '(无效)'}")
            print(f"  节点数量: {nodes}")
            print(f"  边数量:   {edges}")

            # 使用array高效读取数据
            print("\n数据结构验证:")
            
            # 读取offsets数组
            offsets = array.array('I')
            offsets.frombytes(f.read(4 * (nodes + 1)))
            if sys.byteorder != 'little':  # 统一转为小端存储
                offsets.byteswap()
            
            # 读取indices数组
            indices = array.array('I')
            indices.frombytes(f.read(4 * edges))
            if sys.byteorder != 'little':
                indices.byteswap()

            # 数据完整性检查
            print(f"  Offsets数组长度: {len(offsets)} (预期: {nodes + 1})")
            print(f"  Indices数组长度: {len(indices)} (预期: {edges})")
            print(f"  最后offset值: {offsets[-1]} (预期等于边数: {'✓' if offsets[-1] == edges else '✗'})")
            
            # 示例数据展示（仅当数据量较小时展示）
            print("\n示例数据（前5个节点）:")
            show_samples = min(5, nodes)
            for i in range(show_samples):
                start = offsets[i]
                end = offsets[i+1]
                if end - start > 20:  # 边数过多时只显示前5条
                    print(f"  节点{i}的边: {indices[start:start+5]}... (共{end-start}条边)")
                else:
                    print(f"  节点{i}的边: {indices[start:end]}")
                
    except FileNotFoundError:
        print(f"错误: 文件不存在 - {filename}")
    except Exception as e:
        print(f"验证失败: {str(e)}")

if __name__ == "__main__":
    # 配置命令行参数解析
    parser = argparse.ArgumentParser(description="验证Gunrock兼容的CSR二进制文件")
    parser.add_argument("-f", "--file", required=True, help="要验证的CSR文件路径")
    args = parser.parse_args()

    # 执行验证
    inspect_csr_file(args.file)