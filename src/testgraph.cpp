#include <iostream>
#include <fstream>
#include <cstdint>

struct CSRHeader {
    uint32_t magic = 0x47535246;  // "GSRF"
    uint32_t num_nodes = 6;        // 6个节点
    uint32_t num_edges = 8;        // 8条边（无向图每条边存储两次）
    uint32_t _padding = 0;
};

int main() {
    // 无向图结构：
    // 0-1-2环状（每个节点连接两个邻居）
    // 3-4双向连接
    // 5孤立
    
    // CSR 结构
    uint32_t offsets[] = {0, 2, 4, 6, 7, 8, 8}; // 每个节点的边偏移量
    uint32_t edges[] = {
        /* 0 */ 1, 2,          // 节点0连接1和2
        /* 1 */ 0, 2,          // 节点1连接0和2
        /* 2 */ 0, 1,          // 节点2连接0和1
        /* 3 */ 4,             // 节点3连接4
        /* 4 */ 3,             // 节点4连接3
        // 节点5没有边
    };

    // 写入二进制文件
    std::ofstream ofs("test_undirected_graph.bin", std::ios::binary);
    
    // 写入文件头
    CSRHeader header;
    ofs.write(reinterpret_cast<char*>(&header), sizeof(CSRHeader));
    
    // 写入offsets数组
    ofs.write(reinterpret_cast<char*>(offsets), sizeof(offsets));
    
    // 写入edges数组
    ofs.write(reinterpret_cast<char*>(edges), sizeof(edges));
    
    std::cout << "无向测试图已生成到 test_undirected_graph.bin\n";
    std::cout << "节点数: 6\n边数: 8\n数据结构验证：\n";
    std::cout << "0的邻居: 1, 2\n1的邻居: 0, 2\n2的邻居: 0, 1\n3的邻居: 4\n4的邻居: 3\n5无邻居\n";

    return 0;
}