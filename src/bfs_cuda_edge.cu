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
#include <queue>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <iomanip>// 用于十六进制输出
#include <cstring> //for memset

#include <cuda.h>
#include <cuda_runtime.h>
#include "OptionParser.h"//for arguments parse
#include "cudacommon.h" //for checkCudaErrors，assert
#include <cfloat>//for FLT_MAX
#include <algorithm>//for sort

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


float BFSGraph(int no_of_nodes, int edge_list_size, uint32_t *&offsets, uint32_t *&edges);

float BFSGraphUnifiedMemory(OptionParser &op, int no_of_nodes, int edge_list_size, uint32_t *&offsets, uint32_t *&edges);


__global__ void bfs_edge_kernel(uint32_t *offsets, uint32_t *edges,int *current_queue, int *current_size,
    int *next_queue, int *next_size,uint32_t *visited, int* d_cost, int total_edges) 
{
    // ​确定边所属的源节点​（属于当前层的 current_queue）。
    // ​检查邻居节点是否被访问过，若未访问则更新其距离。
    // ​将未访问的邻居加入下一层队列 next_queue。

    // 每个线程处理一个全局边索引
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_edges) return; // 超出边数范围则退出,线程id是0~total_edges-1

    // 二分查找确定当前边对应的源节点（current_queue中）
    //在current_queue中找到对应的src_node，使得该边（tid）位于该节点的边范围内（offsets[node] <= tid < offsets[node + 1]）。
    //关键假设：current_queue 中的节点按 offsets 升序排列
    //这个是保证的吗？？
    //没保证，生成时并未排序
    //同时按边并行后，就算原来有序，加入队列后也不一定有序
    //在kernel启动之前对当前队列进行排序，既方便找边又方便算total_edges
    int left = 0, right = *current_size - 1;
    int src_node = -1;
    while (left <= right) {
        int mid = (left + right) / 2;
        int node = current_queue[mid];
        if (offsets[node] <= tid && tid < offsets[node + 1]) {
            src_node = node;
            break;
        } else if (tid < offsets[node]) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    if (src_node == -1) return; // 边不属于当前层

    // 处理当前边
    int neighbor = edges[tid];

    // 检查是否已访问（使用位掩码）
    int word_idx = neighbor / 32;
    int bit_idx = neighbor % 32;
    uint32_t mask = 1U << bit_idx;
    //先将visited[word_idx]对应位置1,表示已访问,然后再返回修改前的值
    uint32_t old = atomicOr(&visited[word_idx], mask); // 原子操作标记

    if ((old & mask) == 0) { // 未被访问过
        // 获取源节点的距离
        int src_cost = d_cost[src_node]; // 从显存读取源节点距离
        int expected = -1;
        if (atomicCAS(&d_cost[neighbor], expected, src_cost + 1) == expected) {
            int pos = atomicAdd(next_size, 1);
            next_queue[pos] = neighbor;
        }
    }
}


void cuda_bfs_edge(int no_of_nodes, int source, uint32_t *&h_offsets,int *current_queue,
    uint32_t *&d_offsets, uint32_t *&d_edges, uint32_t *&h_graph_visited,int *&d_cost, uint32_t *&d_visited,
    int *d_current_queue, int *d_current_size, int *d_next_queue, int *d_next_size,
    double &kernel_time, double &transfer_time, int &k);

void cuda_bfs_edge_uvm(OptionParser &op, int no_of_nodes, int source, 
    uint32_t *&h_offsets, uint32_t *&graph_offsets, 
    uint32_t *&graph_edges, int *&current_queue, int *&current_size, 
    int *&next_queue, int *&next_size, uint32_t *&visited, int *&cost, 
    double &kernel_time, int &k);

//initGraph负责从csr.bin的文件内读取信息，校验后从文件头获取节点数、边数信息(这里的边数已经是edges数组大小，无需再乘以2)，然后读取offset和edges数组
void initGraph(const string &filename, int &no_of_nodes, int &edge_list_size, uint32_t *&offsets, uint32_t *&edges);

// void cuda_bfs_frontier(cudaDeviceProp &deviceProp, int no_of_nodes, int source, 
//     int *&h_graph_visited,int *&d_cost, int *&d_visited,
//     int **frontier, int *&d_frontier_size, 
//     double &kernel_time, double &transfer_time, int &k);

void cuda_bfs_uvm(int no_of_nodes, int source, uint32_t *&graph_offsets, uint32_t *&graph_edges,
        bool* &graph_mask, bool* &updating_graph_mask, bool* &graph_visited, int* &cost, bool *&over, 
        dim3 &grid, dim3 &block, double &kernel_time, int &k);

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
    printf("Running bfs_cuda_edge\n");
	for (int i = 0; i < passes; i++) {
        if (!quiet) {
            printf("Pass %d:\n", i);
        }
        if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
            //调用uvm相关函数
            float timeUM = BFSGraphUnifiedMemory(op, no_of_nodes, edge_list_size, offsets, edges);
            if (!quiet) {
                if (timeUM == FLT_MAX) {
                    printf("Executing BFS using unified memory...Error.\n");
                } else {
                    printf("Executing BFS using unified memory...Done.\n");
                }
            }
        } else {
            //调用普通cuda执行
            float time = BFSGraph(no_of_nodes, edge_list_size,  offsets, edges);
            if (!quiet) {
                if (time == FLT_MAX) {
                    printf("Executing BFS...Error.\n");
                } else {
                    printf("Executing BFS...Done.\n");
                }
            }
        }
    }

    //清理内存,这个内存分配发生在initGraph内
    free(offsets);
    free(edges);
	return 0;
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


float BFSGraph(int no_of_nodes, int edge_list_size, uint32_t *&h_offsets, uint32_t *&h_edges) 
{
    //该函数功能：
        //分配状态相关以及存结果内存并初始化
        //分配设备内存，避免每次都在内层函数初始化，外层分配一次，内层只涉及多次拷贝
            //部分数据拷贝一次全局即可，如h_graph_nodes、h_graph_edges以及h_updating_graph_mask，每次调用内层循环都不会更改
            //部分数据需要全局统一拷贝一次，之后用的时候再拷贝具体数据即可，避免大批量数据拷贝，如h_graph_mask,h_graph_visited,h_cost
        //设置执行环境+参数，避免内层循环多次设置
        //调用内层循环执行
        //结果拷贝回来，只需要拷贝d_cost回来即可，其余用不到
        //释放内存,设备+主机
        //时间统一记录
        //返回函数值
    auto start_t = high_resolution_clock::now();
    double total_time = 0;
    double kernel_time = 0;
    double transfer_time = 0;
    
	//主机端内存分配
    //状态及存结果的数组，主机端都不需要初始化，直接在设备内存初始化即可，后面还要拷贝回来
    //h_graph_visited使用位掩码，省空间,但增加操作复杂性
    int visited_ints = (no_of_nodes + 31) / 32;
    uint32_t *h_graph_visited = (uint32_t*) malloc(sizeof(uint32_t) * visited_ints);
    assert(h_graph_visited);
    int *h_cost = (int*) malloc( sizeof(int) * no_of_nodes);
    assert(h_cost);
    //初始化
    memset(h_graph_visited, 0, visited_ints * sizeof(uint32_t));
    memset(h_cost, -1, no_of_nodes * sizeof(int));
    //设备端当前层队列，用于预处理
    int *current_queue =  (int*) malloc( sizeof(int) * no_of_nodes);
    memset(current_queue, 0, no_of_nodes * sizeof(int));

    
    //设备端内存分配
    //先确定指针
    uint32_t *d_offsets = nullptr;
    uint32_t *d_edges = nullptr;
    int *d_current_queue= nullptr;
    int *d_next_queue= nullptr;
    int *d_current_size= nullptr;
    int *d_next_size= nullptr; // 队列大小（设备端）
    uint32_t *d_visited= nullptr;    // 位掩码标记已访问节点
    int *d_cost = nullptr;
    
    //计算大小，防止后面重复计算，且不易错
    const size_t offsets_size = sizeof(uint32_t) * (no_of_nodes + 1);
    //const size_t edges_size = sizeof(uint32_t) * h_offsets[no_of_nodes];
    const size_t edges_size = sizeof(uint32_t) * edge_list_size;

    //分配内存
    CUDA_SAFE_CALL(cudaMalloc(&d_offsets, offsets_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_edges, edges_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_current_queue, no_of_nodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_next_queue, no_of_nodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_current_size, sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_next_size, sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_visited, visited_ints * sizeof(uint32_t)));// 位掩码：每个 uint32_t 存储 32 个节点的访问状态
    CUDA_SAFE_CALL(cudaMalloc(&d_cost, sizeof(int) * no_of_nodes));
    
    cudaEvent_t tstart, tstop;
    float elapsedTime = 0;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);
    // 拷贝图数据，offsets和edges
    cudaEventRecord(tstart, 0);
    CUDA_SAFE_CALL(cudaMemcpy(d_offsets, h_offsets, offsets_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_edges, h_edges, edges_size, cudaMemcpyHostToDevice));
    cudaEventRecord(tstop, 0);
    cudaEventSynchronize(tstop);
    cudaEventElapsedTime(&elapsedTime, tstart, tstop);
    transfer_time += elapsedTime * 1.e-3;

    // 初始化设备数据，确实没必要设置了主机再拷贝，直接在设备上初始化
    CUDA_SAFE_CALL(cudaMemset(d_cost, -1, sizeof(int) * no_of_nodes));
    CUDA_SAFE_CALL(cudaMemset(d_visited, 0, sizeof(uint32_t) * visited_ints));// 清零位掩码
    // 初始化队列尺寸
    cudaMemset(d_current_size, 0, sizeof(int));
    cudaMemset(d_next_size, 0, sizeof(int));
    
    // //绑定纹理内存
    // //CUDA纹理内存（Texture Memory）​本质是全局内存的缓存视图，
    // //数据仍然存储在全局内存中，但通过纹理缓存（Texture Cache）的硬件机制进行访问。
    // CUDA_SAFE_CALL(cudaBindTexture(0, tex_offsets, d_offsets, offsets_size));
    // CUDA_SAFE_CALL(cudaBindTexture(0, tex_edges, d_edges, edges_size));

	//遍历所有节点，未访问就进入遍历
    int cnt = 0;//用于记录连通块数量
    int k = 0; //记录kernel执行次数
	for(int i = 0; i < no_of_nodes; i++){
		//未访问才进入遍历
        //在这里从设备拷贝visited不现实，反向思考，内层循环每次执行完都拷贝一次visited，只需拷贝i之后的即可，前面无需
        //return (d_visited[word_idx] & mask) != 0;
        int word_idx = i / 32;
        int bit_idx = i % 32;
        uint32_t mask = 1U << bit_idx;
        int status = h_graph_visited[word_idx];
        bool is_visited = (status & mask);
		if(!is_visited){
            ++cnt;
            // printf("visite node : %d status %d\n", i, is_visited);//used for debug
            cuda_bfs_edge(no_of_nodes, i, h_offsets, current_queue,
                    d_offsets, d_edges, h_graph_visited,d_cost, d_visited,
                    d_current_queue, d_current_size,d_next_queue, d_next_size,
                    kernel_time, transfer_time, k);
		}
	}
	
    //统一拷贝回值
    //只有d_cost拷贝回传
    cudaEventRecord(tstart, 0);
    cudaMemcpy(h_cost, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost);
    cudaEventRecord(tstop, 0);
    cudaEventSynchronize(tstop);
    cudaEventElapsedTime(&elapsedTime, tstart, tstop);
    transfer_time += elapsedTime * 1.e-3; // convert to seconds

    auto end_t = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end_t - start_t);
    double duration_t = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    total_time += duration_t;

    // 清理资源
    // CUDA_SAFE_CALL(cudaUnbindTexture(tex_offsets));
    // CUDA_SAFE_CALL(cudaUnbindTexture(tex_edges));
    cudaFree(d_offsets);
    cudaFree(d_edges);
    cudaFree(d_current_queue);
    cudaFree(d_next_queue);
    cudaFree(d_current_size);
    cudaFree(d_next_size);
    cudaFree(d_visited);
    cudaFree(d_cost);

	//cleanup memory host
	free( h_graph_visited);
	free( h_cost);
    free( current_queue);
	
    //时间统输出一记录
    printf("Time record(seconds)\n");
    printf("total_time : %f\n", total_time);
    printf("transfer_time : %f\n", transfer_time);
    printf("kernel_time : %f\n", kernel_time);
    printf("graph_block : %d\n", cnt);
    printf("kernel_exe_times : %d\n", k);

    return total_time;
}


float BFSGraphUnifiedMemory(OptionParser &op, int no_of_nodes, int edge_list_size, uint32_t *&offsets, uint32_t *&edges) {
    //该函数功能:
        //获取输入参数
        //设置执行环境+参数，避免内层循环多次设置
        //统一内存相关操作
            //分配nodes、edges的统一内存，并从主存拷贝到统一内存，然后根据输入参数进行相应操作
            //分配状态相关打3个统一内存数组以及存储结果打cost统一内存数据，初始化后，根据输入参数做相应操作
            //为over分配统一内存
        //调用内层循环进行执行
        //拷贝回结果数组cost到cpu
        //释放内存
        //统一时间记录
        //返回函数值


    auto start_t = high_resolution_clock::now();
    double total_time = 0;
    double kernel_time = 0;
    // double transfer_time = 0;

    //获取输入参数
    // bool verbose = op.getOptionBool("verbose");
    // bool quiet = op.getOptionBool("quiet");
    int device = op.getOptionInt("device");
    const bool uvm = op.getOptionBool("uvm");
    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    const bool uvm_advise = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");

    
    //分配状态相关数组及cost并初始化
    //原来设备端指针全改uvm
    uint32_t *graph_offsets = nullptr;
    uint32_t *graph_edges = nullptr;
    int *current_queue = nullptr;
    int *next_queue = nullptr;
    int *current_size = nullptr;
    int *next_size = nullptr; // 队列大小（设备端）
    uint32_t *visited = nullptr; // 位掩码标记已访问节点
    int *cost = nullptr;
    

    //计算大小，防止后面重复计算，且不易错
    const size_t offsets_size = sizeof(uint32_t) * (no_of_nodes + 1);
    const size_t edges_size = sizeof(uint32_t) * edge_list_size;
    const size_t visited_size = sizeof(uint32_t) * ((no_of_nodes + 31) /32);//使用位掩码操作

    //内存分配与操作分开
    //全局内存分配
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        //graph_offsets
        checkCudaErrors(cudaMallocManaged(&graph_offsets, offsets_size));
        //graph_edges
        checkCudaErrors(cudaMallocManaged(&graph_edges, edges_size));
        //current_queue
        checkCudaErrors(cudaMallocManaged(&current_queue, sizeof(int)*no_of_nodes));
        //next_queue
        checkCudaErrors(cudaMallocManaged(&next_queue, sizeof(int)*no_of_nodes));
        //current_size
        checkCudaErrors(cudaMallocManaged(&current_size, sizeof(int)));
        //next_size
        checkCudaErrors(cudaMallocManaged(&next_size, sizeof(int)));
        //visited
        checkCudaErrors(cudaMallocManaged(&visited, visited_size));
        //cost
        checkCudaErrors(cudaMallocManaged(&cost, sizeof(int)*no_of_nodes));
    }
    //内存操作，graph_offsets和graph_edges直接从原数组复制
    memcpy(graph_offsets, offsets, offsets_size);
    memcpy(graph_edges, edges, edges_size);
    //其余需要初始化
    memset(current_queue, -1, sizeof(int)*no_of_nodes);
    memset(next_queue, -1, sizeof(int)*no_of_nodes);
    memset(current_size, 0, sizeof(int));
    memset(next_size, 0, sizeof(int));
    memset(visited, 0, sizeof(uint32_t)*visited_size);
    memset(cost, -1, sizeof(int)*no_of_nodes);

    //graph_offsets和graph_edges都是设备只读
    if (uvm) {
        // do nothing, graph_offsets remains on CPU
    } else if (uvm_prefetch) { 
        //数据预取技术
        checkCudaErrors(cudaMemPrefetchAsync(graph_offsets, offsets_size, device));
        checkCudaErrors(cudaMemPrefetchAsync(graph_edges, edges_size, device));
    } else if (uvm_advise) {
        checkCudaErrors(cudaMemAdvise(graph_offsets, offsets_size, cudaMemAdviseSetReadMostly, device));
        checkCudaErrors(cudaMemAdvise(graph_offsets, offsets_size, cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemAdvise(graph_edges, edges_size, cudaMemAdviseSetReadMostly, device));
        checkCudaErrors(cudaMemAdvise(graph_edges, edges_size, cudaMemAdviseSetPreferredLocation, device));
    } else if (uvm_prefetch_advise) {
        checkCudaErrors(cudaMemAdvise(graph_offsets, offsets_size, cudaMemAdviseSetReadMostly, device));
        checkCudaErrors(cudaMemAdvise(graph_offsets, offsets_size, cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemPrefetchAsync(graph_offsets, offsets_size, device));
        checkCudaErrors(cudaMemAdvise(graph_edges, edges_size, cudaMemAdviseSetReadMostly, device));
        checkCudaErrors(cudaMemAdvise(graph_edges, edges_size, cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemPrefetchAsync(graph_edges, edges_size, device));
    } else {
        std::cerr << "unrecognized uvm flag, exiting..." << std::endl;
        exit(-1);
    }

    //其余数组用到之前再选择

    //遍历所有节点，未访问就进入遍历
    int cnt = 0;//用于记录连通块数量
    int k = 0; //用于记录kernel执行次数
	for(int i = 0; i < no_of_nodes; i++){
		//未访问才进入遍历
        //在这里从设备拷贝visited不现实，反向思考，内层循环每次执行完都拷贝一次visited，只需拷贝i之后的即可，前面无需
		int word_idx = i / 32;
        int bit_idx = i % 32;
        uint32_t mask = 1U << bit_idx;
        int status = visited[word_idx];
        bool is_visited = (status & mask);
        if(!is_visited){
            ++cnt;
            cuda_bfs_edge_uvm(op, no_of_nodes, i, offsets, 
                graph_offsets, graph_edges, current_queue, current_size, 
                next_queue, next_size, visited, cost, kernel_time, k);
		}
	}
    //统一拷贝回值
    // copy result from device to host
    if (uvm) {
        // Do nothing, cost stays on CPU
    } else if (uvm_advise) {
        checkCudaErrors(cudaMemAdvise(cost, sizeof(int)*no_of_nodes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        checkCudaErrors(cudaMemAdvise(cost, sizeof(int)*no_of_nodes, cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
    } else if (uvm_prefetch) {
        checkCudaErrors(cudaMemPrefetchAsync(cost, sizeof(int)*no_of_nodes, cudaCpuDeviceId));
    } else if (uvm_prefetch_advise) {
        checkCudaErrors(cudaMemAdvise(cost, sizeof(int)*no_of_nodes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        checkCudaErrors(cudaMemAdvise(cost, sizeof(int)*no_of_nodes, cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
        checkCudaErrors(cudaMemPrefetchAsync(cost, sizeof(int)*no_of_nodes, cudaCpuDeviceId));
    } else {
        std::cerr << "Unrecognized uvm option, exiting..." << std::endl;
        exit(-1);
    }
    
    auto end_t = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end_t - start_t);
    double duration_t = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    total_time += duration_t;
    
    // cleanup memory
	cudaFree(graph_offsets);
	cudaFree(graph_edges);
	cudaFree(current_queue);
	cudaFree(next_queue);
    cudaFree(current_size);
    cudaFree(next_size);
	cudaFree(visited);
	cudaFree(cost);


    //时间统输出一记录
    printf("Time record(seconds)\n");
    printf("total_time : %f\n", total_time);
    //printf("transfer_time : %f\n", transfer_time);
    printf("kernel_time : %f\n", kernel_time);
    printf("graph_block : %d\n", cnt);
    printf("kernel_exe_times : %d\n", k);

    return total_time;
}



//BFSGraphUVM内再封装一层，进行从source出发具体的遍历
//BFSGraphUVM内就进行创建相关状态数组和存结果数组，然后将这些数组传给cuda_bfs_uvm即可
//设备参数也应该传进来，毕竟是在外面进行内存创建,但拷贝还是发生在内层循环内，执行前要拷贝进去，执行完要拷贝出来
void cuda_bfs_edge_uvm(OptionParser &op, int no_of_nodes, int source, 
    uint32_t *&h_offsets, uint32_t *&graph_offsets, 
    uint32_t *&graph_edges, int *&current_queue, int *&current_size, 
    int *&next_queue, int *&next_size, uint32_t *&visited, int *&cost, 
    double &kernel_time, int &k){
        //内层循环功能:
        //设置source相关状态，无需拷贝，这里将会自动管理内存
        //while控制按层遍历

        //获取参数
        int device = op.getOptionInt("device");
        const bool uvm = op.getOptionBool("uvm");
        const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
        const bool uvm_advise = op.getOptionBool("uvm-advise");
        const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
        
        int visited_size = sizeof(uint32_t) * (no_of_nodes + 31) / 32;

        //set the source node as true in the mask and cost
        int word_idx = source / 32;
        int bit_idx = source % 32;
        visited[word_idx] |= (1U << bit_idx); // 设置对应位为1
        cost[source] = 0;
        //初始化队列
        current_queue[*current_size] = source;
        *current_size = 1;

        const int block_size = 256;
        while (*current_size > 0) {
            // 重置下一层队列大小
            // cudaMemset(next_size, 0, sizeof(int));
            *next_size = 0;
            if(uvm){
                //do nothing
            }else if (uvm_advise){
                //do nothing
            }else if (uvm_prefetch){
                checkCudaErrors(cudaMemPrefetchAsync(current_queue, sizeof(int)*no_of_nodes, cudaCpuDeviceId));
                checkCudaErrors(cudaMemPrefetchAsync(current_size, sizeof(int), cudaCpuDeviceId));
            }else if(uvm_prefetch_advise){
                checkCudaErrors(cudaMemPrefetchAsync(current_queue, sizeof(int)*no_of_nodes, cudaCpuDeviceId));
                checkCudaErrors(cudaMemPrefetchAsync(current_size, sizeof(int), cudaCpuDeviceId));
            }
            std::sort(current_queue, current_queue + *current_size, 
            [&h_offsets](int a, int b) { return h_offsets[a] < h_offsets[b]; });
            
            // 配置内核参数
            //当前层总边数，需要全局的边id，即offsets对应
            int last_node = current_queue[*current_size-1];
            int total_edges = h_offsets[last_node + 1];//last_node对应的所有边
            int grid_size = (total_edges + block_size - 1) / block_size;
            
            //启动内核前选择
            if (uvm) {
                // Do nothing.
            } else if (uvm_advise) {
                // Do nothing.
            } else if (uvm_prefetch) {
                //current_queue
                checkCudaErrors(cudaMemPrefetchAsync(current_queue, sizeof(int)*no_of_nodes, device));
                //next_queue,和current差不多
                checkCudaErrors(cudaMemPrefetchAsync(next_queue, sizeof(int)*no_of_nodes, device));
                //current_size
                checkCudaErrors(cudaMemPrefetchAsync(current_size, sizeof(int), device));
                //next_size
                checkCudaErrors(cudaMemPrefetchAsync(next_size, sizeof(int), device));
                //visited; 
                checkCudaErrors(cudaMemPrefetchAsync(visited, visited_size, device));
                //cost
                checkCudaErrors(cudaMemPrefetchAsync(cost, sizeof(int)*no_of_nodes, device));
            } else if (uvm_prefetch_advise) {
                //current_queue
                checkCudaErrors(cudaMemPrefetchAsync(current_queue, sizeof(int)*no_of_nodes, device));
                //next_queue
                checkCudaErrors(cudaMemPrefetchAsync(next_queue, sizeof(int)*no_of_nodes, device));
                //current_size
                checkCudaErrors(cudaMemPrefetchAsync(current_size, sizeof(int), device));
                //next_size
                checkCudaErrors(cudaMemPrefetchAsync(next_size, sizeof(int), device));
                //visited; 
                checkCudaErrors(cudaMemPrefetchAsync(visited, visited_size, device));
                //cost
                checkCudaErrors(cudaMemPrefetchAsync(cost, sizeof(int)*no_of_nodes, device));
            }
            // 启动边并行内核
            //这里有个易出错的地方，total_edges命名不对容易导致误解
            //bfs_edge_kernel内本质是用全局线程id去处理全局全局的边id
            //若只是单纯总边数的话，将会启动线程的全局id对不上边的全局id
            cudaEvent_t tstart, tstop;
            cudaEventCreate(&tstart);
            cudaEventCreate(&tstop);
            float elapsedTime = 0;
            cudaEventRecord(tstart, 0);
            bfs_edge_kernel<<<grid_size, block_size>>>(graph_offsets, 
                graph_edges,current_queue, current_size, next_queue, 
                next_size, visited, cost, total_edges);
            cudaEventRecord(tstop, 0);
            cudaEventSynchronize(tstop);
            cudaEventElapsedTime(&elapsedTime, tstart, tstop);
            kernel_time += elapsedTime * 1.e-3;

            if(uvm){
                //do nothing
            }else if (uvm_advise){
                //do nothing
            }else if (uvm_prefetch){
                checkCudaErrors(cudaMemPrefetchAsync(current_queue, sizeof(int)*no_of_nodes, cudaCpuDeviceId));
                checkCudaErrors(cudaMemPrefetchAsync(current_size, sizeof(int), cudaCpuDeviceId));
                checkCudaErrors(cudaMemPrefetchAsync(next_queue, sizeof(int)*no_of_nodes, cudaCpuDeviceId));
                checkCudaErrors(cudaMemPrefetchAsync(next_size, sizeof(int), cudaCpuDeviceId));
            }else if(uvm_prefetch_advise){
                checkCudaErrors(cudaMemPrefetchAsync(current_queue, sizeof(int)*no_of_nodes, cudaCpuDeviceId));
                checkCudaErrors(cudaMemPrefetchAsync(current_size, sizeof(int), cudaCpuDeviceId));
                checkCudaErrors(cudaMemPrefetchAsync(next_queue, sizeof(int)*no_of_nodes, cudaCpuDeviceId));
                checkCudaErrors(cudaMemPrefetchAsync(next_size, sizeof(int), cudaCpuDeviceId));
            }
            k++;//统计kernel运行次数
            // 交换当前队列和下一队列，就是两个指针而已
            std::swap(current_queue, next_queue);
            //从设备端拷贝回来当前队列，便于下次预处理
            
            //当前层拷贝到主机进行排序
            //将d_next_size复制到current_size_host,便于while控制
            *current_size = *next_size;
            // cudaMemcpy(&current_size, next_size, sizeof(int), cudaMemcpyDeviceToHost);
        }

        if(uvm){
            //do nothing
        }else if (uvm_advise){
            //do nothing
        }else if (uvm_prefetch){
            checkCudaErrors(cudaMemPrefetchAsync(visited, visited_size, cudaCpuDeviceId));
        }else if(uvm_prefetch_advise){
            checkCudaErrors(cudaMemPrefetchAsync(visited, visited_size, cudaCpuDeviceId));
        }
    }



void cuda_bfs_edge( int no_of_nodes, int source, uint32_t *&h_offsets,int *current_queue,
    uint32_t *&d_offsets, uint32_t *&d_edges, uint32_t *&h_graph_visited,int *&d_cost, uint32_t *&d_visited,
    int *d_current_queue, int *d_current_size, int *d_next_queue, int *d_next_size,
    double &kernel_time, double &transfer_time, int &k)
    {
        //相关内存初始化
        //source相关的d_cost、d_visited
        //队列状态设置
        //循环控制分层处理
        //处理完visited数组同步回主机

        //设置d_visited的source对应位为1
        //直接拷贝h_graph_visited
        int word_idx = source / 32;
        int bit_idx = source % 32;
        h_graph_visited[word_idx] |= (1U << bit_idx); // 设置对应位为1
        cudaEvent_t tstart, tstop;
        float elapsedTime = 0;
        cudaEventCreate(&tstart);
        cudaEventCreate(&tstop);

        cudaEventRecord(tstart, 0);
        CUDA_SAFE_CALL(cudaMemcpy(d_visited + word_idx, h_graph_visited + word_idx, sizeof(uint32_t), cudaMemcpyHostToDevice));
        cudaEventRecord(tstop, 0);
        cudaEventSynchronize(tstop);
        cudaEventElapsedTime(&elapsedTime, tstart, tstop);
        transfer_time += elapsedTime * 1.e-3;

        // int is_visited = (h_graph_visited[word_idx]) & (1U << bit_idx);
        // printf("set node %d visited : %d\n", source, is_visited);

        //设置d_cost[source] = 0
        int init_cost = 0;
        //直接初始化source设备相关内存，相当于设置mask，visited和cost,顺带初始化设备端值
        //拷贝一个int就行
        cudaEventRecord(tstart, 0);
        CUDA_SAFE_CALL(cudaMemcpy(&d_cost[source], &init_cost, sizeof(int), cudaMemcpyHostToDevice));
        cudaEventRecord(tstop, 0);
        cudaEventSynchronize(tstop);
        cudaEventElapsedTime(&elapsedTime, tstart, tstop);
        transfer_time += elapsedTime * 1.e-3;

        // 将起始节点加入当前队列
        int initial_current_size = 1;
        cudaEventRecord(tstart, 0);
        cudaMemcpy(d_current_queue, &source, sizeof(int), cudaMemcpyHostToDevice);//source入队
        cudaMemcpy(d_current_size, &initial_current_size, sizeof(int), cudaMemcpyHostToDevice);
        cudaEventRecord(tstop, 0);
        cudaEventSynchronize(tstop);
        cudaEventElapsedTime(&elapsedTime, tstart, tstop);
        transfer_time += elapsedTime * 1.e-3;

        //线程相关参数设置，动态部分在while内
        const int block_size = 256;
        // const int sm_count = deviceProp.multiProcessorCount;
        // const int max_blocks_per_sm = deviceProp.maxBlocksPerMultiProcessor; // 每个SM最大block数
        // const int max_blocks = sm_count * max_blocks_per_sm;//根据硬件设置的理论值

        
        //初始化当前层
        int current_size_host = initial_current_size;//用于控制while循环
        while (current_size_host > 0) {

            // 重置下一层队列大小
            cudaMemset(d_next_size, 0, sizeof(int));
            // 主机端对current_queue排序处理，便于内核使用二分法找src_node,也方便找total_edges
            cudaEventRecord(tstart, 0);
            CUDA_SAFE_CALL(cudaMemcpy(current_queue, d_current_queue, current_size_host * sizeof(int), cudaMemcpyDeviceToHost));
            cudaEventRecord(tstop, 0);
            cudaEventSynchronize(tstop);
            cudaEventElapsedTime(&elapsedTime, tstart, tstop);
            transfer_time += elapsedTime * 1.e-3;
            std::sort(current_queue, current_queue + current_size_host, 
            [&h_offsets](int a, int b) { return h_offsets[a] < h_offsets[b]; });
            // 再拷贝到设备端
            cudaEventRecord(tstart, 0);
            CUDA_SAFE_CALL(cudaMemcpy(d_current_queue, current_queue, 
            current_size_host * sizeof(int), cudaMemcpyHostToDevice));
            cudaEventRecord(tstop, 0);
            cudaEventSynchronize(tstop);
            cudaEventElapsedTime(&elapsedTime, tstart, tstop);
            transfer_time += elapsedTime * 1.e-3;
            // 配置内核参数
            //当前层总边数，需要全局的边id，即offsets对应
            int last_node = current_queue[current_size_host-1];
            int total_edges = h_offsets[last_node + 1];//last_node对应的所有边
            int grid_size = (total_edges + block_size - 1) / block_size;
            //需要判断下吧？会不会超限？
            //理论上不会的，grid的x维度可以是2^31-1约20多亿块，我800万的图，150的度，总边数也就6亿，双倍也就12亿
            // printf("max_blocks is %d\n", max_blocks);
            // printf("grid_size is %d\n", grid_size);
            // if(grid_size > max_blocks){
            //     printf("grid_size over max_blocks!\n");
            //     exit(1);
            // }
            // 启动边并行内核
            //这里有个易出错的地方，total_edges命名不对容易导致误解
            //bfs_edge_kernel内本质是用全局线程id去处理全局全局的边id
            //若只是单纯总边数的话，将会启动线程的全局id对不上边的全局id
            cudaEventRecord(tstart, 0);
            bfs_edge_kernel<<<grid_size, block_size>>>(d_offsets, d_edges,d_current_queue, d_current_size,
            d_next_queue, d_next_size,d_visited,d_cost, total_edges);
            cudaEventRecord(tstop, 0);
            cudaEventSynchronize(tstop);
            cudaEventElapsedTime(&elapsedTime, tstart, tstop);
            kernel_time += elapsedTime * 1.e-3;
            k++;//统计kernel运行次数
            // 交换当前队列和下一队列，就是两个指针而已
            std::swap(d_current_queue, d_next_queue);
            //从设备端拷贝回来当前队列，便于下次预处理
            cudaEventRecord(tstart, 0);
            //当前层拷贝到主机进行排序
            //将d_next_size复制到current_size_host,便于while控制
            cudaMemcpy(&current_size_host, d_next_size, sizeof(int), cudaMemcpyDeviceToHost);
            //再将current_size_host复制到d_current_size,是方便下一次启动bfs_edge_kernel执行
            cudaMemcpy(d_current_size, &current_size_host, sizeof(int), cudaMemcpyHostToDevice);
            cudaEventRecord(tstop, 0);
            cudaEventSynchronize(tstop);
            cudaEventElapsedTime(&elapsedTime, tstart, tstop);
            transfer_time += elapsedTime * 1.e-3;
        }
        
        //每次执行完需要将visited状态信息拷贝到主机侧，便于判断进入下一次遍历
        int visited_ints = (no_of_nodes + 31) / 32;

        // 同步 visited 到主机
        cudaEventRecord(tstart, 0);
        CUDA_SAFE_CALL(cudaMemcpy(h_graph_visited, d_visited, visited_ints * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        cudaEventRecord(tstop, 0);
        cudaEventSynchronize(tstop);
        cudaEventElapsedTime(&elapsedTime, tstart, tstop);
        transfer_time += elapsedTime * 1.e-3;
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