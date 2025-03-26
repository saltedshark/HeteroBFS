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
#include <cstdint> // 确保uint32_t/int32_t类型定义
#include "OptionParser.h"//for arguments parse

// #include <cugraph_c.h>
#include <cugraph_c/traversal_algorithms.h>     //该文件包含了error.h、graph.h以及resource_handle.h
#include <cugraph_c/types.h>                    
#include <thrust/device_ptr.h>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <cstdint>

// 示例错误检查宏
#define CUDA_CHECK(call) {                              \
    cudaError_t status = call;                          \
    if (status != cudaSuccess) {                        \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1);                                         \
    }}

#define CUGRAPH_CHECK(call) {                           \
    cugraph_error_code_t status = call;                 \
    if (status != CUGRAPH_SUCCESS) {                    \
        std::cerr << "cuGraph Error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1);                                         \
    }}

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
void initGraph(const string &filename, size_t &no_of_nodes, size_t &edge_list_size, uint32_t *&offsets, uint32_t *&edges);

void inner_bfs(cugraph_resource_handle_t* handle,cugraph_graph_t* graph,
    uint32_t* d_distances,size_t num_vertices,double &transfer_time, size_t &cnt);
void run_bfs(const uint32_t* h_offsets,const uint32_t* h_edges,size_t num_vertices,
    size_t num_edges);

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

	size_t no_of_nodes = 0;
	size_t edge_list_size = 0;
    uint32_t *offsets;
    uint32_t *edges;

    //读文件获取信息
	initGraph(filename, no_of_nodes, edge_list_size, offsets, edges);

    //截至目前，读图，读参数完毕

    
    //执行passes次，根据参数判断使用啥程序，普通还是uvm相关
    printf("Running bfs_cugraph\n");
	for (int i = 0; i < passes; i++) {
        if (!quiet) {
            printf("Pass %d:\n", i);
        }
        run_bfs(offsets, edges, no_of_nodes, edge_list_size);
        //结果处理
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

void initGraph(const string &filename, size_t &no_of_nodes, size_t &edge_list_size, uint32_t *&offsets, uint32_t *&edges){
	
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

void checkCudaFeatureAvailability(OptionParser &op) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));
    
    // Check UVM availability
    if (op.getOptionBool("uvm") || op.getOptionBool("uvm-advise") ||
            op.getOptionBool("uvm-prefetch") || op.getOptionBool("uvm-prefetch-advise")) {
        if (!deviceProp.unifiedAddressing) {
            std::cerr << "device doesn't support unified addressing, exiting..." << std::endl;
            exit(-1);
        }
    }

    // Check Cooperative Group availability
    if (op.getOptionBool("coop")) {
        if (!deviceProp.cooperativeLaunch) {
            std::cerr << "device doesn't support cooperative kernels, exiting..." << std::endl;
            exit(-1);
        }
    }

    // Check Dynamic Parallelism availability
    if (op.getOptionBool("dyn")) {
        int runtimeVersion = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
        if (runtimeVersion < 5000) {
            std::cerr << "CUDA runtime version less than 5.0, doesn't support \
                dynamic parallelism, exiting..." << std::endl;
            exit(-1);
        }
    }

    // Check CUDA Graphs availability
    if (op.getOptionBool("graph")) {
        int runtimeVersion = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
        if (runtimeVersion < 10000) {
            std::cerr << "CUDA runtime version less than 10.0, doesn't support \
                CUDA Graph, exiting..." << std::endl;
            exit(-1);
        }
    }
}


// 自定义仿函数用于查找首个未访问节点
//distances初始化值为全-1
struct find_unvisited {
    const int32_t* distances;
    __host__ __device__
    bool operator()(const int32_t& idx) const {
        return distances[idx] == -1;//UINT32_MAX就是-1
    }
};

// 设备端查找首个未访问节点
int32_t find_first_unvisited(int32_t* d_distances,int32_t num_vertices)
{
    auto counting = thrust::make_counting_iterator<int32_t>(0);
    auto end = counting + num_vertices;
    
    auto it = thrust::find_if(
        thrust::device,
        counting,
        end,
        find_unvisited{d_distances});
    
    return (it != end) ? *it : -1;//使用-1表示未找到
}

void inner_bfs(
    cugraph_resource_handle_t* handle,
    cugraph_graph_t* graph,
    int32_t* d_distances,
    size_t num_vertices,
    double &transfer_time, double &kernel_time, size_t &cnt)
{
    int32_t* d_start_vertex;
    CUDA_CHECK(cudaMalloc(&d_start_vertex, sizeof(int32_t)));
    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);
    float elapsedTime = 0;

    // printf("while开始执行\n");
    while(true) {
        // 查找首个未访问节点（设备端操作）
        int32_t start_vertex = find_first_unvisited(d_distances, num_vertices);
        printf("start node is %d\n", start_vertex);
        if(start_vertex == -1) break;//全访问过

        cudaEventRecord(tstart, 0);
        // 设置起始顶点
        CUDA_CHECK(cudaMemcpy(d_start_vertex, &start_vertex, sizeof(int32_t), cudaMemcpyHostToDevice));
        cudaEventRecord(tstop, 0);
        cudaEventSynchronize(tstop);
        cudaEventElapsedTime(&elapsedTime, tstart, tstop);
        transfer_time += elapsedTime * 1.e-3;

        //创建起点视图
        cugraph_type_erased_device_array_view_t* source_view =  cugraph_type_erased_device_array_view_create(
            d_start_vertex, 1, INT32);//这里也是同样问题不支持UINT32？
        // 执行BFS
        cugraph_paths_result_t* bfs_result = nullptr;
        cugraph_error_t* error = nullptr;
        // printf("执行cugraph_bfs\n");
        
        auto start_t = high_resolution_clock::now();
        cugraph_error_code_t error_code = cugraph_bfs(
            handle,
            graph,
            source_view,
            TRUE,               //true自动优化
            num_vertices,       //size_t depth_limit, 不可达后值会被设置为int32max
            FALSE,              //compute_predecessors,
            FALSE,              //do_expensive_check,
            &bfs_result,
            &error
        );
        if (error_code != CUGRAPH_SUCCESS) {
            const char* error_msg = cugraph_error_message(error);
            std::cerr << "cugraph_bfs执行失败: " << error_msg << std::endl;
            cugraph_error_free(error);
            exit(1);
        }

        auto end_t = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end_t - start_t);
        double duration_t = double(duration.count()) * microseconds::period::num / microseconds::period::den;
        kernel_time += duration_t;
        cnt++;//记录graph_blcok

        
        
        // printf("执行cugraph_bfs结束\n");
        // 合并结果
        cugraph_type_erased_device_array_view_t* current_dist_view = 
            cugraph_paths_result_get_distances(bfs_result);

        // 使用Thrust合并结果（保留最短距离）
        thrust::transform(
            thrust::device,
            d_distances,
            d_distances + num_vertices,
            static_cast<const int32_t*>(cugraph_type_erased_device_array_view_pointer(current_dist_view)),
            d_distances,
            [] __device__ (int32_t a, int32_t b) {
                // int32_t adjusted_b = (b == INT32_MAX) ? -1 : b;
                // if (a == -1) return adjusted_b;
                // if (adjusted_b == -1) return a;
                // return min(a, adjusted_b);
                // 转换 cuGraph 的不可达标记
                // int32_t adjusted_b = (b == INT32_MAX) ? -1 : b;

                // if (a == -1) {
                //     // 原距离无效，采用 BFS 结果
                //     return adjusted_b;
                // } else if (adjusted_b == -1) {
                //     // BFS 结果无效，保留原距离
                //     return a;
                // } else {
                //     // 两者均有效，取最小值
                //     return min(a, adjusted_b);
                // }


                // 区分 BFS 结果：-1表示全局未访问，INT32_MAX表示本次未访问
                // constexpr int32_t UNVISITED_GLOBAL = -1;
                // constexpr int32_t UNVISITED_LOCAL = INT32_MAX;

                // if (b == UNVISITED_LOCAL) {
                //     // 本次 BFS 未访问该节点，保留原值
                //     return a;
                // } else if (a == UNVISITED_GLOBAL) {
                //     // 全局未访问，采用本次 BFS 结果
                //     return b;
                // } else {
                //     // 两者均有效，取最小值
                //     return min(a, b);
                // }

                // 全局未访问标记: -1
                // 本次 BFS 未访问标记: INT32_MAX
                if (b == INT32_MAX) return a;  // 保留原值
                if (a == -1) return b;         // 全局未访问，采用本次结果
                return min(a, b);              // 两者均有效，取最小值

                //  // 调整逻辑：优先保留有效距离
                // int32_t adjusted_b = (b == INT32_MAX) ? -1 : b;
                // if (a != -1 && adjusted_b != -1) {
                //     return min(a, adjusted_b);  // 两者均有效，取最小
                // } else if (a != -1) {
                //     return a;                   // 仅原值有效
                // } else if (adjusted_b != -1) {
                //     return adjusted_b;          // 仅 BFS 结果有效
                // } else {
                //     return -1;                 // 两者均无效
                // }

                // if (a == -1) return b;
                // if (b == -1) return a;
                // return min(a, b);
            }
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        // 释放资源
        cugraph_type_erased_device_array_view_free(source_view);
        cugraph_paths_result_free(bfs_result);
    }
    // printf("while执行完毕\n");
    //事件销毁
    cudaEventDestroy(tstart);
    cudaEventDestroy(tstop);
    CUDA_CHECK(cudaFree(d_start_vertex));
}

void run_bfs(const uint32_t* h_offsets, const uint32_t* h_edges,
     size_t num_vertices, size_t num_edges){

    auto start_t = high_resolution_clock::now();
    double total_time = 0;
    double transfer_time = 0;
    double kernel_time = 0;
    size_t cnt = 0;//记录graph_block

    // allocate mem for the result on host side
    int *h_distances = (int*) malloc( sizeof(int)*num_vertices);
    memset(h_distances, 0xFF, num_vertices * sizeof(int));//这里使用距离数组当作未访问，初始化为INT32_MAX
    // std::fill(h_distances, h_distances + num_vertices, INT32_MAX);

    printf("h_distance拷贝前检查\n");
    for(int i = 0; i < num_vertices; i++){
        std::cout << h_distances[i] << " ";
    }
    std::cout << std::endl;

    // printf("创建资源句柄\n");
    // 创建资源句柄
    cugraph_resource_handle_t* handle = cugraph_create_resource_handle(NULL);
    
    // printf("设备内存分配及资源拷贝\n");
    // 设备内存分配
    uint32_t *d_offsets, *d_edges;
    int32_t *d_distances;
    CUDA_CHECK(cudaMalloc(&d_offsets, (num_vertices + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_edges, num_edges * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_distances, num_vertices * sizeof(int32_t)));


    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);
    float elapsedTime = 0;
    cudaEventRecord(tstart, 0);
    //内容拷贝
    cudaMemcpy( d_offsets, h_offsets, sizeof(int32_t)*(num_vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy( d_edges, h_edges, sizeof(int32_t)*num_edges, cudaMemcpyHostToDevice);
    cudaEventRecord(tstop, 0);
    cudaEventSynchronize(tstop);
    cudaEventElapsedTime(&elapsedTime, tstart, tstop);
    transfer_time += elapsedTime * 1.e-3;
    
    // 初始化距离数组
    CUDA_CHECK(cudaMemset(d_distances, 0xFF, num_vertices * sizeof(int32_t)));//这里使用距离数组当作未访问，初始化为INT32_MAX
    // CUDA_CHECK(cudaMemcpy(d_distances, h_distances, sizeof(int32_t)*num_vertices, cudaMemcpyHostToDevice));
    
    // printf("距离数组初始化完毕\n");s

    // 创建图对象
    cugraph_graph_t* graph = nullptr;
    cugraph_error_t* error = nullptr;
    //图属性设置
    cugraph_graph_properties_t properties;
    properties.is_symmetric = TRUE;   // 无向图需设为 TRUE
    properties.is_multigraph = FALSE;   //不允许多重边

    //由设备内存直接创建视图,简洁
    cugraph_type_erased_device_array_view_t* offsets_view = cugraph_type_erased_device_array_view_create(
        d_offsets, num_vertices + 1, INT32);//这里应该是bug，居然不支持UINT32，INT32在下面的graph_create老报错

    cugraph_type_erased_device_array_view_t* edges_view = cugraph_type_erased_device_array_view_create(
        d_edges, num_edges, INT32);//这里应该是bug，居然不支持UINT32
    
    // printf("数组视图创建完毕\n");
    

    cugraph_error_code_t error_code = cugraph_graph_create_sg_from_csr(
        handle,
        &properties,
        offsets_view,
        edges_view,
        NULL,       // weights
        NULL,       // NULL if edge types are not used.
        NULL,       // NULL if edge ids are not used
        FALSE,      //TRUE表示存储CSC格式（列优先），默认FALSE
        TRUE,       //开启后自动将顶点ID映射为连续整数，提升性能
        FALSE,      //If true, symmetrize the edgelist
        FALSE,      //bool_t do_expensive_check
        &graph,
        &error);
    if (error_code != CUGRAPH_SUCCESS) {
        const char* error_msg = cugraph_error_message(error);
        std::cerr << "图创建失败: " << error_msg << std::endl;
        cugraph_error_free(error);
        exit(1);
    }


    // printf("图创建创建完毕\n");
    

    //图创建后视图即可清理，节省空间
    cugraph_type_erased_device_array_view_free(edges_view);
    cugraph_type_erased_device_array_view_free(offsets_view);

    // printf("开始执行inner_bfs\n");
    // 执行分层BFS
    inner_bfs(handle, graph, d_distances, num_vertices, 
        transfer_time, kernel_time, cnt);

    // printf("拷贝cost回主机\n");
    // 拷贝结果回主机
    cudaEventRecord(tstart, 0);
    CUDA_CHECK(cudaMemcpy(h_distances, d_distances, num_vertices * sizeof(int32_t), cudaMemcpyDeviceToHost));
    cudaEventRecord(tstop, 0);
    cudaEventSynchronize(tstop);
    cudaEventElapsedTime(&elapsedTime, tstart, tstop);
    transfer_time += elapsedTime * 1.e-3;

    printf("h_distance拷贝回检查\n");
    for(int i = 0; i < num_vertices; i++){
        std::cout << h_distances[i] << " ";
    }
    std::cout << std::endl;

    auto end_t = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end_t - start_t);
    double duration_t = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    total_time += duration_t;

    // 清理资源
    //事件销毁
    cudaEventDestroy(tstart);
    cudaEventDestroy(tstop);
    cugraph_graph_free(graph);
    cugraph_free_resource_handle(handle);
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_edges));
    CUDA_CHECK(cudaFree(d_distances));
    
    free(h_distances);

    //时间统输出一记录
    printf("Time record(seconds)\n");
    printf("total_time : %f\n", total_time);
    printf("transfer_time : %f\n", transfer_time);
    printf("kernel_time : %f\n", kernel_time);
    printf("graph_block : %zu\n", cnt);//size_t
}
