// src/bfs_cuda_edge.cu
// ================================
// 该代码使用了[Altis](https://github.com/utcs-scea/altis)代码并对其进行了部分修改
// Copyright (c) 2021, Bodun Hu, et al.
// [BSD-2条款]许可证见本项目根目录的licenses/altis.txt
// ================================

/***
 * 函数功能：使用cuda并行的bfs对图进行遍历，不管图有没有连通
 * 本程序使用边并行方案。
 * 特点：
 * 边比节点多，适合gpu
 */
#include <stdio.h>
#include <cstdlib>
#include <string>
using std::string;
#include <chrono>
using namespace std::chrono;
#include <iostream>
#include <fstream>

#include <iomanip>// 用于十六进制输出
#include <cstring> //for memset

#include "OptionParser.h"//for arguments parse

// Gunrock 核心
// #include <gunrock/gunrock.h>          // 核心功能及配置
#include <gunrock/graph/graph.hxx>      // 图结构定义
#include <gunrock/graph/graph_csr.h>  // CSR格式图结构（GraphCSR类）
#include <gunrock/algorithms/bfs.h>   // BFS算法实现
#include <gunrock/util/array.h>       // 设备内存数组操作（stats.distances）

// CUDA/Thrust
#include <thrust/device_ptr.h>        // Thrust设备指针
#include <thrust/execution_policy.h>  // Thrust执行策略（thrust::device）
#include <thrust/find.h>              // Thrust查找算法
#include <cuda_runtime.h>             // CUDA运行时API（cudaMalloc/cudaMemcpy）
#include <cstdint>                    // 确保uint32_t/int32_t类型定义


#include <algorithm>                  // std::fill


// CSR二进制文件头（兼容Gunrock）
struct CSRHeader {
    uint32_t magic;      // 魔数校验
    uint32_t num_nodes;  // 节点数
    uint32_t num_edges;  // 边数
    uint32_t _padding=0;   // 填充字段
};

// //纹理引用是全局静态的，只能放到main之外，不能在函数内定义
// texture<uint32_t> tex_offsets;
// texture<uint32_t> tex_edges;

// 打印十六进制值的辅助函数
void PrintHex(const char* label, uint32_t value) {
    std::cout << std::left << std::setw(15) << label 
              << "0x" << std::hex << std::uppercase 
              << std::setw(8) << std::setfill('0') << value << std::endl;
}

void checkCudaFeatureAvailability(OptionParser &op);


//initGraph负责从csr.bin的文件内读取信息，校验后从文件头获取节点数、边数信息(这里的边数已经是edges数组大小，无需再乘以2)，然后读取offset和edges数组
void initGraph(const string &filename, int &no_of_nodes, int &edge_list_size, uint32_t *&offsets, uint32_t *&edges);


void Opinit(OptionParser &op);

//main内根据参数解析判断是否使用uvm相关，调用不同的BFSGraph,BFSGraphUnifiedMemory
int main(int argc, char** argv){
    //参数预设置
    // Get args
    OptionParser op;
    Opinit(op);
    if (!op.parse(argc, argv))
    {
        op.usage();
        return (op.HelpRequested() ? 0 : 1);
    }
    // Check CUDA feature availability
    checkCudaFeatureAvailability(op);
    //设备设置
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    //解析本程序所用参数
    string filename = op.getOptionString("inputFile");
    int passes = op.getOptionInt("passes");
    bool quiet = op.getOptionBool("quiet");
    const bool uvm = op.getOptionBool("uvm");
    const bool uvm_advise = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");

	int no_of_nodes = 0;
	int edge_list_size = 0;
    uint32_t *offsets;
    uint32_t *edges;

    //读文件获取信息
	initGraph(filename, no_of_nodes, edge_list_size, offsets, edges);

    //截至目前，读图，读参数完毕

    //执行passes次，根据参数判断使用啥程序，普通还是uvm相关
    printf("Running bfs_cuda_gunrock\n");
	for (int i = 0; i < passes; i++) {
        if (!quiet) {
            printf("Pass %d:\n", i);
        }
        //调用bfsgunrock
        BFSGunrock(no_of_nodes, edge_list_size, offsets, edges);
    }

    //清理内存,这个内存分配发生在initGraph内
    free(offsets);
    free(edges);
	return 0;
}

void Opinit(OptionParser &op){
    // Add shared options to the parser
    op.addOption("passes", OPT_INT, "10", "specify number of passes", 'n');
    op.addOption("verbose", OPT_BOOL, "0", "enable verbose output", 'v');
    op.addOption("quiet", OPT_BOOL, "0", "enable concise output", 'q');
    op.addOption("inputFile", OPT_STRING, "", "path of input file", 'i');
    // op.addOption("outputFile", OPT_STRING, "", "path of output file", 'o');
    op.addOption("device", OPT_VECINT, "0", "specify device(s) to run on", 'd');

    // Add options for turn on/off CUDA features
    op.addOption("uvm", OPT_BOOL, "0", "enable CUDA Unified Virtual Memory, only demand paging");
    op.addOption("uvm-advise", OPT_BOOL, "0", "guide the driver about memory usage patterns");
    op.addOption("uvm-prefetch", OPT_BOOL, "0", "prefetch memory the specified destination device");
    op.addOption("uvm-prefetch-advise", OPT_BOOL, "0", "prefetch memory the specified destination device with memory guidance on");
    op.addOption("coop", OPT_BOOL, "0", "enable CUDA Cooperative Groups");
    op.addOption("dyn", OPT_BOOL, "0", "enable CUDA Dynamic Parallelism");
    op.addOption("graph", OPT_BOOL, "0", "enable CUDA Graphs");
}

void initGraph(const string &filename, int &no_of_nodes, int &edge_list_size, uint32_t *&offsets, uint32_t *&edges){
	
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(0);
    }

	printf("Reading graph file\n");
    // 读取文件头
    CSRHeader header;
    ifs.read(reinterpret_cast<char*>(&header), sizeof(CSRHeader));
    // 校验魔数
    if (header.magic != 0x47535246) {  // "GSRF" 的十六进制
        std::cerr << "文件格式错误，无效的魔数" << std::endl;
        PrintHex("预期魔数", 0x47535246);
        PrintHex("实际魔数", header.magic);
        exit(0);
    }
    // 校验数据合理性
    if (header.num_nodes == 0 || header.num_edges == 0) {
        std::cerr << "文件头数据异常: 节点数=" << header.num_nodes 
                  << " 边数=" << header.num_edges << std::endl;
        exit(0);
    }
    // 输出头信息
    std::cout << "====== 文件头信息 ======" << std::endl;
    PrintHex("魔数", header.magic);
    std::cout << std::dec;  // 切回十进制输出
    std::cout << "节点数: " << header.num_nodes << std::endl;
    std::cout << "边数: " << header.num_edges << std::endl;
    std::cout << "填充字段: " << header._padding << std::endl;

    no_of_nodes = header.num_nodes;
    edge_list_size = header.num_edges;

    // offsets.reserve(header.num_nodes + 1);
    // edges.reserve(header.num_edges);
    offsets = (uint32_t*) malloc(sizeof(uint32_t) * (no_of_nodes+1));//offset需要多分配1个
    edges = (uint32_t*) malloc(sizeof(uint32_t) * edge_list_size);

    // 读取偏移数组
    ifs.read(reinterpret_cast<char*>(offsets), 
            (no_of_nodes + 1) * sizeof(uint32_t));

    // 读取边索引
    ifs.read(reinterpret_cast<char*>(edges), 
            edge_list_size * sizeof(uint32_t));

    // 验证读取完整性
    if (!ifs) {
        std::cerr << "文件读取不完整或已损坏" << std::endl;
        exit(0);
    }
	
	std::cout << "\n文件验证通过，数据结构完整" << std::endl;
}


void BFSGunrock(int no_of_nodes, int edge_list_size, uint32_t *&offsets, uint32_t *&edges){
    // 假设已正确初始化offsets/edges数组
    // 初始化 cost 数组为 -1
    int32_t* cost = new int32_t[no_of_nodes];
    std::fill(cost, cost + no_of_nodes, -1);
    //运行gunrock
    run_gunrock_bfs(no_of_nodes, edge_list_size, cost);

    //资源清理
    delete[] cost;
}

// 生成cost数组的函数接口
void run_gunrock_bfs(int num_nodes, int num_edges, int32_t* cost) {
    // = 1. 初始化Gunrock图结构 = 
    // 分别对应 <VertexId, EdgeId, ValueT> 类型。
    gunrock::graph::GraphCSR<uint32_t, uint32_t, uint32_t> g;
    
    // 分配设备内存并拷贝数据
    g.Allocate(num_nodes, num_edges, gunrock::util::DEVICE);
    g.CopyToGpu(offsets, edges, nullptr);  // 无权重时第三个参数为nullptr
    
    // = 2. 初始化访问标记 = 
    bool* d_visited;
    cudaMalloc(&d_visited, num_nodes * sizeof(bool));
    cudaMemset(d_visited, 0, num_nodes * sizeof(bool));
    
    // = 3. 主循环处理所有连通分量 = 
    uint32_t current_source = 0;
    while (true) {
        // 在GPU上查找第一个未访问节点
        // GPU 并行查找：Thrust 的 find 算法在 GPU 上并行搜索，时间复杂度为 O(1)（利用 GPU 的 SIMT 架构）。
        thrust::device_ptr<bool> dev_ptr(d_visited);
        auto pos = thrust::find(thrust::device, dev_ptr, dev_ptr + num_nodes, false);
        if (pos - dev_ptr >= num_nodes) break;
        // 计算找到的位置与起始位置的偏移量（即节点索引）
        current_source = pos - dev_ptr;
        
        // = 4. 运行BFS = 
        gunrock::bfs::Stats stats;
        gunrock::bfs::Run(
            g,                                  // 图结构
            current_source,                     // 源节点
            -1,                                 // 最大迭代次数（无限制）
            d_visited,                          // 访问标记数组
            nullptr,                            // 前驱节点数组（不需要）
            nullptr,                            // 边计数数组（不需要）
            stats,                              // 统计信息
            false,                              // 不标记前驱
            0                                   // 遍历模式（默认）
        );
        
        // = 5. 合并距离到cost数组 = 
        // 将GPU端的距离数组拷贝到临时主机数组
        int32_t* d_distances = stats.distances.GetPointer(gunrock::util::DEVICE);
        int32_t* h_distances = new int32_t[num_nodes];
        cudaMemcpy(h_distances, d_distances, num_nodes * sizeof(int32_t), cudaMemcpyDeviceToHost);
        
        // 更新全局cost数组（仅修改未被访问过的节点）
        for (uint32_t i = 0; i < num_nodes; ++i) {
            if (cost[i] == -1 && h_distances[i] != -1) {
                cost[i] = h_distances[i];
            }
        }
        delete[] h_distances;
    }
    
    // = 6. 清理资源 = 
    cudaFree(d_visited);
    g.Free();
}

void checkCudaFeatureAvailability(OptionParser &op) {
    int device = 0;
    checkCudaErrors(cudaGetDevice(&device));
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device));
    
    // Check UVM availability
    if (op.getOptionBool("uvm") || op.getOptionBool("uvm-advise") ||
            op.getOptionBool("uvm-prefetch") || op.getOptionBool("uvm-prefetch-advise")) {
        if (!deviceProp.unifiedAddressing) {
            std::cerr << "device doesn't support unified addressing, exiting..." << std::endl;
            safe_exit(-1);
        }
    }

    // Check Cooperative Group availability
    if (op.getOptionBool("coop")) {
        if (!deviceProp.cooperativeLaunch) {
            std::cerr << "device doesn't support cooperative kernels, exiting..." << std::endl;
            safe_exit(-1);
        }
    }

    // Check Dynamic Parallelism availability
    if (op.getOptionBool("dyn")) {
        int runtimeVersion = 0;
        checkCudaErrors(cudaRuntimeGetVersion(&runtimeVersion));
        if (runtimeVersion < 5000) {
            std::cerr << "CUDA runtime version less than 5.0, doesn't support \
                dynamic parallelism, exiting..." << std::endl;
            safe_exit(-1);
        }
    }

    // Check CUDA Graphs availability
    if (op.getOptionBool("graph")) {
        int runtimeVersion = 0;
        checkCudaErrors(cudaRuntimeGetVersion(&runtimeVersion));
        if (runtimeVersion < 10000) {
            std::cerr << "CUDA runtime version less than 10.0, doesn't support \
                CUDA Graph, exiting..." << std::endl;
            safe_exit(-1);
        }
    }
}