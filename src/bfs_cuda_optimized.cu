/***
 * 函数功能：使用cuda并行的bfs对图进行遍历，不管图有没有连通
 * 本程序使用Frontier-based，即数据驱动模式，动态维护一个“活跃节点队列”（Frontier），每个线程处理队列中的节点并生成下一层的队列
 * 特点：
 *  ​适合 GPU：通过原子操作或前缀和（Prefix Sum）高效管理队列，减少同步次数。
 * ​ 优点：负载均衡、减少全局同步。
 *  结合动态队列管理和细粒度并行，最大化 GPU 利用率：
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

// CSR二进制文件头（兼容Gunrock）
struct CSRHeader {
    uint32_t magic;      // 魔数校验
    uint32_t num_nodes;  // 节点数
    uint32_t num_edges;  // 边数
    uint32_t _padding=0;   // 填充字段
};

//纹理引用是全局静态的，只能放到main之外，不能在函数内定义
texture<uint32_t> tex_offsets;
texture<uint32_t> tex_edges;

// 打印十六进制值的辅助函数
void PrintHex(const char* label, uint32_t value) {
    std::cout << std::left << std::setw(15) << label 
              << "0x" << std::hex << std::uppercase 
              << std::setw(8) << std::setfill('0') << value << std::endl;
}

void checkCudaFeatureAvailability(OptionParser &op);


float BFSGraph(cudaDeviceProp &deviceProp, int no_of_nodes, int edge_list_size, uint32_t *&offsets, uint32_t *&edges);

float BFSGraphUnifiedMemory(OptionParser &op, cudaDeviceProp &deviceProp, int no_of_nodes, int edge_list_size, uint32_t *&offsets, uint32_t *&edges);

//Frontier-base处理内核
__global__ void ProcessFrontier(int* frontier_in, int frontier_size, int* frontier_out, 
                                int* frontier_out_size, int* d_cost, int* d_visited)
    {
        extern __shared__ int s_buf[]; // 每个block独立使用共享内存
        
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;
    
        if (warp_id >= frontier_size) return;
    
        const int node = frontier_in[warp_id];
        const uint32_t start = tex1Dfetch(tex_offsets, node);
        const uint32_t end = tex1Dfetch(tex_offsets, node + 1);
        const int num_neighbors = end - start;
    
        // ===== 第一次遍历：精确统计有效邻居 =====
        int valid_neighbors = 0;
        for (int i = lane_id; i < num_neighbors; i += 32) {
            if (i >= num_neighbors) break; // 边界保护
            const uint32_t neighbor = tex1Dfetch(tex_edges, start + i);
            if (atomicCAS(&d_visited[neighbor], 0, 1) == 0) {
                d_cost[neighbor] = d_cost[node] + 1;
                valid_neighbors++;
            }
        }
    
        // ===== Warp级归约求和 =====
        unsigned active_mask = __ballot_sync(0xFFFFFFFF, valid_neighbors > 0);
        int warp_total = __reduce_add_sync(0xFFFFFFFF, valid_neighbors);
    
        // ===== 原子操作分配写入基址 =====
        if (lane_id == 0 && warp_total > 0) {
            const int base = atomicAdd(frontier_out_size, warp_total);
            s_buf[threadIdx.x / 32] = base; // 使用block内局部索引
        }
        __syncthreads();
    
        // ===== 精确写入所有有效邻居 =====
        if (warp_total > 0) {
            // Step 1: 计算warp内前缀和
            int offset = 0;
            for (int i = 0; i < lane_id; ++i) {
                offset += __shfl_sync(active_mask, valid_neighbors, i);
            }
    
            // Step 2: 获取本warp的写入基址
            const int base = s_buf[threadIdx.x / 32];
    
            // Step 3: 安全写入
            int write_counter = 0;
            for (int i = lane_id; i < num_neighbors; i += 32) {
                if (i >= num_neighbors) break; // 二次边界检查
                const uint32_t neighbor = tex1Dfetch(tex_edges, start + i);
                if (d_cost[neighbor] == d_cost[node] + 1) {
                    if (base + offset + write_counter < *frontier_out_size){
                        //全局边界检查
                        frontier_out[base + offset + write_counter] = neighbor;
                        write_counter++;
                    } 
                }
            }
        }
    }

//initGraph负责从csr.bin的文件内读取信息，校验后从文件头获取节点数、边数信息(这里的边数已经是edges数组大小，无需再乘以2)，然后读取offset和edges数组
void initGraph(const string &filename, int &no_of_nodes, int &edge_list_size, uint32_t *&offsets, uint32_t *&edges);

void cuda_bfs_frontier(cudaDeviceProp &deviceProp, int no_of_nodes, int source, 
    int *&h_graph_visited,int *&d_cost, int *&d_visited,
    int **frontier, int *&d_frontier_size, 
    double &kernel_time, double &transfer_time, int &k);

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
    printf("Running bfs_cuda\n");
	for (int i = 0; i < passes; i++) {
        if (!quiet) {
            printf("Pass %d:\n", i);
        }
        if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
            //调用uvm相关函数
            float timeUM = BFSGraphUnifiedMemory(op, deviceProp, no_of_nodes, edge_list_size, offsets, edges);
            if (!quiet) {
                if (timeUM == FLT_MAX) {
                    printf("Executing BFS using unified memory...Error.\n");
                } else {
                    printf("Executing BFS using unified memory...Done.\n");
                }
            }
        } else {
            //调用普通cuda执行
            float time = BFSGraph(deviceProp, no_of_nodes, edge_list_size,  offsets, edges);
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


float BFSGraph(cudaDeviceProp &deviceProp, int no_of_nodes, int edge_list_size, uint32_t *&h_offsets, uint32_t *&h_edges) 
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
    int *h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
    int *h_graph_visited = (int*) malloc(sizeof(int)*no_of_nodes);
    

    //设备端内层分配
    uint32_t *d_offsets = nullptr;
    uint32_t *d_edges = nullptr;
    int *d_cost = nullptr;
    int *frontier[2] = {nullptr, nullptr};//对应原来mask和updating
    int *d_visited = nullptr;//对应visited
    int *d_frontier_size = nullptr;//新增存大小

    //先计算大小，防止后面重复计算，且不易错
    const size_t offsets_size = sizeof(uint32_t) * (no_of_nodes + 1);
    //const size_t edges_size = sizeof(uint32_t) * h_offsets[no_of_nodes];
    const size_t edges_size = sizeof(uint32_t) * edge_list_size;

    CUDA_SAFE_CALL(cudaMalloc(&d_offsets, offsets_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_edges, edges_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_cost, sizeof(int) * no_of_nodes));
    CUDA_SAFE_CALL(cudaMalloc(&d_visited, sizeof(int) * no_of_nodes));
    CUDA_SAFE_CALL(cudaMalloc(&frontier[0], sizeof(int) * no_of_nodes));
    CUDA_SAFE_CALL(cudaMalloc(&frontier[1], sizeof(int) * no_of_nodes));
    CUDA_SAFE_CALL(cudaMalloc(&d_frontier_size, sizeof(int)));
	
    cudaEvent_t tstart, tstop;
    float elapsedTime = 0;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);
    cudaEventRecord(tstart, 0);
    // 拷贝图数据，offsets和edges
    CUDA_SAFE_CALL(cudaMemcpy(d_offsets, h_offsets, offsets_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_edges, h_edges, edges_size, cudaMemcpyHostToDevice));
    cudaEventRecord(tstop, 0);
    cudaEventSynchronize(tstop);
    cudaEventElapsedTime(&elapsedTime, tstart, tstop);
    transfer_time += elapsedTime * 1.e-3;

    // 初始化设备数据，确实没必要设置了主机再拷贝，直接在设备上初始化
    CUDA_SAFE_CALL(cudaMemset(d_cost, 0xFF, sizeof(int) * no_of_nodes));
    CUDA_SAFE_CALL(cudaMemset(d_visited, 0, sizeof(int) * no_of_nodes));
    

    //绑定纹理内存
    //CUDA纹理内存（Texture Memory）​本质是全局内存的缓存视图，
    //数据仍然存储在全局内存中，但通过纹理缓存（Texture Cache）的硬件机制进行访问。
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_offsets, d_offsets, offsets_size));
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_edges, d_edges, edges_size));

	//遍历所有节点，未访问就进入遍历
    int cnt = 0;//用于记录连通块数量
    int k = 0; //记录kernel执行次数
	for(int i = 0; i < no_of_nodes; i++){
		//未访问才进入遍历
        //在这里从设备拷贝visited不现实，反向思考，内层循环每次执行完都拷贝一次visited，只需拷贝i之后的即可，前面无需
		if(!h_graph_visited[i]){
            ++cnt;
            cuda_bfs_frontier(deviceProp, no_of_nodes, i, h_graph_visited, d_cost, d_visited, 
                frontier, d_frontier_size, kernel_time, transfer_time, k);
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
    CUDA_SAFE_CALL(cudaUnbindTexture(tex_offsets));
    CUDA_SAFE_CALL(cudaUnbindTexture(tex_edges));
    cudaFree(d_offsets);
    cudaFree(d_edges);
    cudaFree(d_cost);
    cudaFree(d_visited);
    cudaFree(frontier[0]);
    cudaFree(frontier[1]);
    cudaFree(d_frontier_size);

	//cleanup memory host
	free( h_graph_visited);
	free( h_cost);
	
    //时间统输出一记录
    printf("total_time is %f seconds\n", total_time);
    printf("transfer_time is %f seconds\n", transfer_time);
    printf("kernel_time is %f seconds\n", kernel_time);
    printf("graph_block is %d\n", cnt);
    printf("kernel_exe_times are %d\n", k);

    return total_time;
}


float BFSGraphUnifiedMemory(OptionParser &op, cudaDeviceProp &deviceProp, int no_of_nodes, int edge_list_size, uint32_t *&offsets, uint32_t *&edges) {
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

    //获取输入参数
    // bool verbose = op.getOptionBool("verbose");
    // bool quiet = op.getOptionBool("quiet");
    int device = op.getOptionInt("device");
    const bool uvm = op.getOptionBool("uvm");
    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    const bool uvm_advise = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");

    double total_time = 0;
    double kernel_time = 0;
    // double transfer_time = 0;

    //内层循环执行参数设置
    int num_of_blocks = 1;
    int num_of_threads_per_block = no_of_nodes;//正常限制为1024，这里先设置为节点数，然后再计算
    int max_threads_per_block = deviceProp.maxThreadsPerBlock;//硬件限制的
    //Make execution Parameters according to the number of nodes
    //Distribute threads across multiple Blocks if necessary
    if (no_of_nodes > max_threads_per_block)
    {
        num_of_blocks = (int)ceil(no_of_nodes/(double)max_threads_per_block);
        num_of_threads_per_block = max_threads_per_block; 
    }
    //setup execution parameters
    dim3  grid( num_of_blocks, 1, 1);//1维grid，有num_of_blocks个block
    dim3  block( num_of_threads_per_block, 1, 1);//1维threads


    //根据输入参数创建相应的全局内存数组，并做出相应操作
    //nodes，edges，3个状态相关数组及cost数组

    //首先是offsets数组，得注意这个大小为no_of_nodes+1
    // copy offsets to unified memory
    //checkCudaErrors定义与cudacommon
    uint32_t* graph_offsets = nullptr;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        //统一内存分配，之后数据第一次访问时将会按需自动迁移到访问设备
        checkCudaErrors(cudaMallocManaged(&graph_offsets, sizeof(uint32_t)*(no_of_nodes+1)));
    }
    memcpy(graph_offsets, offsets, sizeof(uint32_t)*(no_of_nodes+1));

    if (uvm) {
        // do nothing, graph_offsets remains on CPU
    } else if (uvm_prefetch) { 
        //数据预取技术
        checkCudaErrors(cudaMemPrefetchAsync(graph_offsets, sizeof(uint32_t)*(no_of_nodes+1), device));
    } else if (uvm_advise) {
        checkCudaErrors(cudaMemAdvise(graph_offsets, sizeof(uint32_t)*(no_of_nodes+1), cudaMemAdviseSetReadMostly, device));
        checkCudaErrors(cudaMemAdvise(graph_offsets, sizeof(uint32_t)*(no_of_nodes+1), cudaMemAdviseSetPreferredLocation, device));
    } else if (uvm_prefetch_advise) {
        checkCudaErrors(cudaMemAdvise(graph_offsets, sizeof(uint32_t)*(no_of_nodes+1), cudaMemAdviseSetReadMostly, device));
        checkCudaErrors(cudaMemAdvise(graph_offsets, sizeof(uint32_t)*(no_of_nodes+1), cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemPrefetchAsync(graph_offsets, sizeof(uint32_t)*(no_of_nodes+1), device));
    } else {
        std::cerr << "unrecognized uvm flag, exiting..." << std::endl;
        exit(-1);
    }

    //然后是edges数组
    // copy edges to unified memory
    uint32_t* graph_edges = nullptr;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        checkCudaErrors(cudaMallocManaged(&graph_edges, sizeof(uint32_t)*edge_list_size));
    }
    memcpy(graph_edges, edges, sizeof(uint32_t)*edge_list_size);
    if (uvm) {
        // Do nothing, graph_edges remains on CPU
    } else if (uvm_prefetch) {
        checkCudaErrors(cudaMemPrefetchAsync(graph_edges, sizeof(uint32_t)*edge_list_size, device));
    } else if (uvm_advise) {
        checkCudaErrors(cudaMemAdvise(graph_edges, sizeof(uint32_t)*edge_list_size, cudaMemAdviseSetReadMostly, device));
        checkCudaErrors(cudaMemAdvise(graph_edges, sizeof(uint32_t)*edge_list_size, cudaMemAdviseSetPreferredLocation, device));
    } else if (uvm_prefetch_advise) {
        checkCudaErrors(cudaMemAdvise(graph_edges, sizeof(uint32_t)*edge_list_size, cudaMemAdviseSetReadMostly, device));
        checkCudaErrors(cudaMemAdvise(graph_edges, sizeof(uint32_t)*edge_list_size, cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemPrefetchAsync(graph_edges, sizeof(uint32_t)*edge_list_size, device));
    } else {
        std::cerr << "unrecognized uvm flag, exiting..." << std::endl;
        exit(-1);
    }

    //分配状态相关数组及cost并初始化
	// allocate and initalize the memory
    bool* graph_mask;
    bool* updating_graph_mask;
    bool* graph_visited;
    // allocate and initialize memory for result
    int *cost = NULL;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        checkCudaErrors(cudaMallocManaged(&graph_mask, sizeof(bool)*no_of_nodes));
        checkCudaErrors(cudaMallocManaged(&updating_graph_mask, sizeof(bool)*no_of_nodes));
        checkCudaErrors(cudaMallocManaged(&graph_visited, sizeof(bool)*no_of_nodes));
        cudaError_t err = cudaMallocManaged(&cost, sizeof(int)*no_of_nodes);
        if (err != cudaSuccess) {
            checkCudaErrors(cudaFree(graph_offsets));
            checkCudaErrors(cudaFree(graph_edges));
            checkCudaErrors(cudaFree(graph_mask));
            checkCudaErrors(cudaFree(updating_graph_mask));
            checkCudaErrors(cudaFree(graph_visited));
            checkCudaErrors(cudaFree(cost));
            return FLT_MAX;
        }
    }

    //初始化状态相关数组及cost
    for( int i = 0; i < no_of_nodes; i++) {
        graph_mask[i]=false;
        updating_graph_mask[i]=false;
        graph_visited[i]=false;
        cost[i]=-1;
    }
 
    //根据输入参数对3个状态相关数组做相应操作，cost放在之后进行错误处理
    if (uvm) {
        // Do nothing. graph_mask, updating_graph_mask, and graph_visited unallocated
    } else if (uvm_advise) {
        checkCudaErrors(cudaMemAdvise(graph_mask, sizeof(bool)*no_of_nodes, cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemAdvise(updating_graph_mask, sizeof(bool)*no_of_nodes, cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemAdvise(graph_visited, sizeof(bool)*no_of_nodes, cudaMemAdviseSetPreferredLocation, device));
    } else if (uvm_prefetch) {
        checkCudaErrors(cudaMemPrefetchAsync(graph_mask, sizeof(bool)*no_of_nodes, device));
        cudaStream_t s1, s2;
        checkCudaErrors(cudaStreamCreate(&s1));
        checkCudaErrors(cudaStreamCreate(&s2));
        checkCudaErrors(cudaMemPrefetchAsync(updating_graph_mask, sizeof(bool)*no_of_nodes, device, s1));
        checkCudaErrors(cudaMemPrefetchAsync(graph_visited, sizeof(bool)*no_of_nodes, device, s2));
        checkCudaErrors(cudaStreamDestroy(s1));
        checkCudaErrors(cudaStreamDestroy(s2));
    } else if (uvm_prefetch_advise) {
        checkCudaErrors(cudaMemAdvise(graph_mask, sizeof(bool)*no_of_nodes, cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemAdvise(updating_graph_mask, sizeof(bool)*no_of_nodes, cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemAdvise(graph_visited, sizeof(bool)*no_of_nodes, cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemPrefetchAsync(graph_mask, sizeof(bool)*no_of_nodes, device));
        cudaStream_t s1, s2;
        checkCudaErrors(cudaStreamCreate(&s1));
        checkCudaErrors(cudaStreamCreate(&s2));
        checkCudaErrors(cudaMemPrefetchAsync(updating_graph_mask, sizeof(bool)*no_of_nodes, device, s1));
        checkCudaErrors(cudaMemPrefetchAsync(graph_visited, sizeof(bool)*no_of_nodes, device, s2));
        checkCudaErrors(cudaStreamDestroy(s1));
        checkCudaErrors(cudaStreamDestroy(s2));
    }

    // cudaError_t err = cudaGetLastError();

    if (uvm) {
        // Do nothing, cost stays on CPU
    } else if (uvm_advise) {
        checkCudaErrors(cudaMemAdvise(cost, sizeof(int)*no_of_nodes, cudaMemAdviseSetPreferredLocation, device));
    } else if (uvm_prefetch) {
        checkCudaErrors(cudaMemPrefetchAsync(cost, sizeof(int)*no_of_nodes, device));
    } else if (uvm_prefetch_advise) {
        checkCudaErrors(cudaMemAdvise(cost, sizeof(int)*no_of_nodes, cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemPrefetchAsync(cost, sizeof(int)*no_of_nodes, device));
    } else {
        std::cerr << "Unrecognized uvm option, exiting...";
        exit(-1);
    }


	// bool if execution is over
    bool *over = nullptr;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        checkCudaErrors(cudaMallocManaged(&over, sizeof(bool)));
    }

    
    //遍历所有节点，未访问就进入遍历
    int cnt = 0;//用于记录连通块数量
    int k = 0; //用于记录kernel执行次数
	for(int i = 0; i < no_of_nodes; i++){
		//未访问才进入遍历
        //在这里从设备拷贝visited不现实，反向思考，内层循环每次执行完都拷贝一次visited，只需拷贝i之后的即可，前面无需
		if(!graph_visited[i]){
            ++cnt;
			cuda_bfs_uvm(no_of_nodes, i, graph_offsets, graph_edges, 
                graph_mask, updating_graph_mask, graph_visited, cost, over,
                grid, block, kernel_time, k);
		}
	}
    // if(verbose){
    //     printf("kernel executed %d times\n", k);
    // }

    //统一拷贝回值
    
    // copy result from device to host
    // checkCudaErrors(cudaEventRecord(tstart, 0));   
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
	checkCudaErrors(cudaFree(graph_offsets));
	checkCudaErrors(cudaFree(graph_edges));
	checkCudaErrors(cudaFree(graph_mask));
	checkCudaErrors(cudaFree(updating_graph_mask));
	checkCudaErrors(cudaFree(graph_visited));
	checkCudaErrors(cudaFree(cost));
    checkCudaErrors(cudaFree(over));

    //时间统输出一记录
    printf("total_time is %f seconds\n", total_time);
    printf("kernel_time is %f seconds\n", kernel_time);
    printf("graph_block is %d\n", cnt);
    printf("kernel_exe_times are %d\n", k);

    return total_time;
}


//BFSGraphUVM内再封装一层，进行从source出发具体的遍历
//BFSGraphUVM内就进行创建相关状态数组和存结果数组，然后将这些数组传给cuda_bfs_uvm即可
//设备参数也应该传进来，毕竟是在外面进行内存创建,但拷贝还是发生在内层循环内，执行前要拷贝进去，执行完要拷贝出来
void cuda_bfs_uvm(int no_of_nodes, int source, uint32_t *&graph_offsets, uint32_t *&graph_edges,
     bool* &graph_mask, bool* &updating_graph_mask, bool* &graph_visited, int* &cost, bool *&over, 
     dim3 &grid, dim3 &block, double &kernel_time, int &k){
        //内层循环功能:
        //设置source相关状态，无需拷贝，这里将会自动管理内存
        //执行do-while执行两个kernel
        //因为使用uvm，所以少了数据管理

        //set the source node as true in the mask and cost
	    graph_mask[source]=true;
	    graph_visited[source]=true;
        cost[source]=0;
        
        
        // events for timing
        cudaEvent_t tstart, tstop;
        checkCudaErrors(cudaEventCreate(&tstart));
        checkCudaErrors(cudaEventCreate(&tstop));
        float elapsedTime = 0; 

	
        
        bool stop;
        //Call the Kernel until all the elements of Frontier are not false
        do
        {
            stop = false;
            *over = stop;

            checkCudaErrors(cudaEventRecord(tstart, 0));
            Kernel<<< grid, block, 0 >>>(graph_offsets, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes);
            checkCudaErrors(cudaEventRecord(tstop, 0));
            checkCudaErrors(cudaEventSynchronize(tstop));
            checkCudaErrors(cudaEventElapsedTime(&elapsedTime, tstart, tstop));
            kernel_time += elapsedTime * 1.e-3;
            CHECK_CUDA_ERROR();

            // check if kernel execution generated an error
            checkCudaErrors(cudaEventRecord(tstart, 0));
            Kernel2<<< grid, block, 0 >>>(graph_mask, updating_graph_mask, graph_visited, over, no_of_nodes);
            checkCudaErrors(cudaEventRecord(tstop, 0));
            checkCudaErrors(cudaEventSynchronize(tstop));
            checkCudaErrors(cudaEventElapsedTime(&elapsedTime, tstart, tstop));
            kernel_time += elapsedTime * 1.e-3;
            CHECK_CUDA_ERROR()

            stop = *over;
            k++;
        }
        while (stop);

        // copy result from device to host
        //uvm，无需手动拷贝，用时会自动
        
        
        /*
        //为了统一时间记录规则，这部分不需要了
        string outfile = op.getOptionString("outputFile");
        if(outfile != "") {
            FILE *fpo = fopen(outfile.c_str(),"w");
            for(int i=0;i<no_of_nodes;i++) {
                fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
            }
            fclose(fpo);
        }
        */
        //Store the result into a file
    }


//该函数实现了从source节点开始的遍历，使用Frontier-based方法
void cuda_bfs_frontier(cudaDeviceProp &deviceProp, int no_of_nodes, int source, 
    int *&h_graph_visited,int *&d_cost, int *&d_visited,
    int **frontier, int *&d_frontier_size, 
    double &kernel_time, double &transfer_time, int &k)
    {
    cudaEvent_t tstart, tstop;
    float elapsedTime = 0;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);
    //直接初始化source设备相关内存，相当于设置mask，visited和cost,顺带初始化设备端值
    int init_cost = 0;
    cudaEventRecord(tstart, 0);
    CUDA_SAFE_CALL(cudaMemcpy(&d_cost[source], &init_cost, sizeof(int), cudaMemcpyHostToDevice));
    cudaEventRecord(tstop, 0);
    cudaEventSynchronize(tstop);
    cudaEventElapsedTime(&elapsedTime, tstart, tstop);
    transfer_time += elapsedTime * 1.e-3;
    // 初始化队列
    int current = 0;//初始当前层编号为0
    int frontier_size[2] = {1, 0};//最开始当前层只有source，下一层没有节点
    cudaEventRecord(tstart, 0);
    CUDA_SAFE_CALL(cudaMemcpy(&frontier[0], &source, sizeof(int), cudaMemcpyHostToDevice));
    cudaEventRecord(tstop, 0);
    cudaEventSynchronize(tstop);
    cudaEventElapsedTime(&elapsedTime, tstart, tstop);
    transfer_time += elapsedTime * 1.e-3;
    //设置d_visited
    int init_visited = 1;
    cudaEventRecord(tstart, 0);
    CUDA_SAFE_CALL(cudaMemcpy(&d_visited[source], &init_visited, sizeof(int), cudaMemcpyHostToDevice));
    cudaEventRecord(tstop, 0);
    cudaEventSynchronize(tstop);
    cudaEventElapsedTime(&elapsedTime, tstart, tstop);
    transfer_time += elapsedTime * 1.e-3;
    
    //已修复，动态在do-while内设置
    const int block_size = 256;
    dim3  threads(block_size);
    const int sm_count = deviceProp.multiProcessorCount;
    const int max_blocks_per_sm = deviceProp.maxBlocksPerMultiProcessor; // 每个SM最大block数
    const int max_blocks = sm_count * max_blocks_per_sm;//根据硬件设置的理论值
    // BFS主循环
    do {
        // 重置下一层队列大小
        frontier_size[1 - current] = 0;
        cudaEventRecord(tstart, 0);
        CUDA_SAFE_CALL(cudaMemcpy(d_frontier_size, &frontier_size[1 - current], sizeof(int), cudaMemcpyHostToDevice));
        cudaEventRecord(tstop, 0);
        cudaEventSynchronize(tstop);
        cudaEventElapsedTime(&elapsedTime, tstart, tstop);
        transfer_time += elapsedTime * 1.e-3;

        // 动态计算grid配置 --------------------------
        const int warps_per_block = block_size / 32;  // 每个block包含的warp数
        const int needed_blocks = (frontier_size[current] + warps_per_block - 1) / warps_per_block;
        dim3 dynamic_grid(min(needed_blocks, max_blocks)); 
        // 启动处理内核
        cudaEventRecord(tstart, 0);
        ProcessFrontier<<<dynamic_grid, threads, block_size*sizeof(int)>>>(frontier[current], frontier_size[current], 
            frontier[1 - current], d_frontier_size, d_cost, d_visited);
        cudaEventRecord(tstop, 0);
        cudaEventSynchronize(tstop);
        cudaEventElapsedTime(&elapsedTime, tstart, tstop);
        kernel_time += elapsedTime * 1.e-3;
        
        // 更新队列状态
        cudaEventRecord(tstart, 0);
        CUDA_SAFE_CALL(cudaMemcpy(&frontier_size[1 - current], d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost));
        cudaEventRecord(tstop, 0);
        cudaEventSynchronize(tstop);
        cudaEventElapsedTime(&elapsedTime, tstart, tstop);
        transfer_time += elapsedTime * 1.e-3;
        current = 1 - current;
        k++;
    } while (frontier_size[current] > 0);
    //每次执行完需要将visited状态信息拷贝到主机侧，便于判断进入下一次遍历
    //先计算拷贝的节点数量
    //这里和原版不同，这里需要从source开始拷贝，外层循环初始化时是直接在设备上初始化的
    //拷贝[source, no_of_nodes)
    int count = no_of_nodes - source;//左闭右开区间
    cudaEventRecord(tstart, 0);
    cudaMemcpy(h_graph_visited + source, d_visited + source, sizeof(int)*count, cudaMemcpyDeviceToHost);
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