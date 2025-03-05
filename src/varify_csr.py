# 脚本功能说明：用于验证csr生成程序的有效性
# 用法：python3 varify_csr.py --file | -f filename
import struct
import argparse

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
            
            print(f"验证文件: {filename}")
            print(f"  Magic标识: 0x{magic:08X} {'(有效)' if magic == 0x47535246 else '(无效)'}")
            print(f"  节点数量: {nodes}")
            print(f"  边数量:   {edges}")

            # 读取CSR数据结构
            offsets = struct.unpack(f"<{nodes + 1}I", f.read(4 * (nodes + 1)))
            indices = struct.unpack(f"<{edges}I", f.read(4 * edges))
            
            # 数据完整性检查
            print("\n数据结构验证:")
            print(f"  Offsets数组长度: {len(offsets)} (预期: {nodes + 1})")
            print(f"  Indices数组长度: {len(indices)} (预期: {edges})")
            print(f"  最后offset值: {offsets[-1]} (预期等于边数: {'✓' if offsets[-1] == edges else '✗'})")
            
            # 示例数据展示
            print("\n示例数据（前5个节点）:")
            for i in range(min(5, nodes)):
                start = offsets[i]
                end = offsets[i+1]
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