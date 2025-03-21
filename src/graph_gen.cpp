// src/graph_gen.cpp
// ================================
// 该代码使用了SNAP库 (https://snap.stanford.edu)的图生成相关代码
// Copyright (c) 2007-2019, Jure Leskovec
// 类BSD-3条款许可证见本项目根目录的 licenses/SNAP.txt
// ================================

/***
 * 程序功能：根据图节点数和节点平均度生成ER图，并将其以CSR二进制格式存储下来，方便后续使用,该csr文件适配gunrock的csr格式；
 * 用法 ./graph_gen  -n nodes -d average_degree
 * 存储图的命名规则graph1k_d10_csr.bin
 * bin文件内容:
 *  CSRBinHeader header;
 *  uint32_t offsets[header.num_nodes + 1];
 *  uint32_t indices[header.num_edges];
 */


#include "Snap.h"
#include <fstream>
#include <vector>
#include <cstdint>
#include <iostream>
#include <cstdlib>  // 用于参数解析
#include <cmath> //for round
#include <algorithm> //for sort
#include <sstream>
#include <iomanip>
// CSR二进制文件头（兼容Gunrock）
struct CSRHeader {
    uint32_t magic = 0x47535246;  // 'GSRF'的ASCII码十六进制表示
    uint32_t num_nodes;
    uint32_t num_edges;
    uint32_t _padding = 0;        // 对齐填充
};

// 节点数格式化函数
std::string FormatNodeCount(int64_t nodes) {
    constexpr int64_t K = 1000;
    constexpr int64_t M = K * 1000;

    if (nodes >= M) return std::to_string(nodes / M) + "M";
    if (nodes >= K) return std::to_string(nodes / K) + "K";
    return std::to_string(nodes);
}

// 生成并保存适配Gunrock的CSR格式
void GenerateERGraphForGunrock(int64_t num_nodes, double avg_degree, int seed) {
    //节点数量限制在32位范围
    // 参数校验（32位范围校验）
    if (num_nodes > UINT32_MAX) {
        throw std::runtime_error("节点数超过32位限制");
    }
    if (num_nodes < 1) {
        throw std::runtime_error("节点数必须大于0");
    }

    // 计算边数并校验合理性
    const int64_t num_edges = round((num_nodes * avg_degree) / 2);//取整
    const int64_t max_possible_edges = num_nodes * (num_nodes - 1) / 2;
    
    if (num_edges > max_possible_edges) {
        throw std::runtime_error("请求的边数超过最大可能值: " + 
                               std::to_string(max_possible_edges));
    }
    if (num_edges > UINT32_MAX) {
        throw std::runtime_error("边数超过32位限制");
    }

    // 生成无向图
    // TRnd& default_rnd = TInt::Rnd; // 默认随机数生成器
    // printf("当前种子: %u\n", default_rnd.GetSeed());
    TRnd rnd(seed);  //指定随机种子，便于复现
    printf("当前随机种子: %u\n", rnd.GetSeed());
    PUNGraph graph = TSnap::GenRndGnm<PUNGraph>(num_nodes, num_edges, false, rnd);

    // 构建CSR数据结构
    std::vector<uint32_t> offsets(num_nodes + 1, 0);
    std::vector<uint32_t> edges;

    // 第一遍：计算度数
    for (int64_t node = 0; node < num_nodes; ++node) {
        if (graph->IsNode(node)) {
            offsets[node + 1] = graph->GetNI(node).GetDeg();
        }
    }

    // 前缀和计算偏移量
    //offset[i]表示节点i在edges内的开始下标为offset[i]
    for (uint32_t i = 1; i <= num_nodes; ++i) {
        offsets[i] += offsets[i - 1];
    }

    // 第二遍：填充邻接表
    //节点i在edges邻居范围为edges[offset[i]] ~ edges[offset[i+1] - 1]
    edges.resize(offsets.back());//双边数量
    std::vector<uint32_t> temp_offsets(offsets.begin(), offsets.begin() + num_nodes);//只取到offset倒数第2个元素
    
    for (int64_t node = 0; node < num_nodes; ++node) {
        if (graph->IsNode(node)) {
            TUNGraph::TNodeI NI = graph->GetNI(node);
            const uint32_t start = temp_offsets[node];
            for (int64_t i = 0; i < NI.GetDeg(); ++i) {//遍历每个节点的所有边
                const uint32_t dest = NI.GetNbrNId(i);
                edges[start + i] = dest;
            }
        }
    }

    // +++++++++++ 新增排序代码 ++++++++++++
    // 对每个节点的邻接列表进行升序排序
    for (uint32_t node = 0; node < num_nodes; ++node) {
        const uint32_t start_idx = offsets[node];
        const uint32_t end_idx = offsets[node + 1];
        
        if (end_idx > start_idx) {  // 确保有邻居才排序
            auto begin_it = edges.begin() + start_idx;
            auto end_it = edges.begin() + end_idx;
            std::sort(begin_it, end_it);
            
            // 去重（如果需要）
            // end_it = std::unique(begin_it, end_it);
            // edges.resize(end_it - edges.begin());
        }
    }
    // +++++++++++ 新增代码结束 ++++++++++++

    // 生成文件名
    // 在生成文件名的代码段修改为：
    //确保转换只保留1位小数，输入时是保证1位小数的，正常to_string后面会有很多0
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << avg_degree;  // 强制显示1位小数
    std::string degree_str = ss.str();

    std::string filename = "graph" + FormatNodeCount(num_nodes) + 
                          "_d" + degree_str + 
                          "_gunrock.csr.bin";

    // 写入二进制文件
    std::ofstream ofs(filename, std::ios::binary);
    CSRHeader header;
    header.num_nodes = static_cast<uint32_t>(num_nodes);
    header.num_edges = static_cast<uint32_t>(edges.size());
    ofs.write(reinterpret_cast<const char*>(&header), sizeof(CSRHeader));
    ofs.write(reinterpret_cast<const char*>(offsets.data()), offsets.size() * sizeof(uint32_t));
    ofs.write(reinterpret_cast<const char*>(edges.data()), edges.size() * sizeof(uint32_t));
    
    std::cout << "成功生成图文件: " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    // 参数解析
    int64_t num_nodes = 0;
    double avg_degree = 0;
    int seed = -1;// 初始化为 -1 表示未设置

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-n" || arg == "--nodes") && i + 1 < argc) {
            num_nodes = std::atoll(argv[++i]);
        } else if ((arg == "-d" || arg == "--degree") && i + 1 < argc) {
            avg_degree = std::atof(argv[++i]);
        } else if ((arg == "-s" || arg == "--seed") && i + 1 < argc) {
            seed = std::atoi(argv[++i]);
        }else {
            std::cerr << "未知或无效参数: " << arg << std::endl;
            return 1;
        }
    }

    // 校验必要参数
    if (num_nodes <= 0 || avg_degree < 0 || seed < 0) {// seed 必须 ≥0
        std::cerr << "用法: " << argv[0] << " -n <节点数> -d <平均度> -s <随机种子>\n"
                  << "示例: " << argv[0] << " -n 1000 -d 10 -s 12345\n"
                  << "注意: 种子必须是非负整数"
                  << std::endl;
        return 1;
    }

    // 执行生成逻辑
    try {
        GenerateERGraphForGunrock(num_nodes, avg_degree, seed);
    } catch (const std::exception& e) {
        std::cerr << "生成失败: " << e.what() << std::endl;
        return 2;
    }
    return 0;
}