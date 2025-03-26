/***
 * 函数功能：使用基于omp的版本对图进行遍历，不管图有没有连通
 * 分为普通版本与优化版本，优化版本使用本地掩码缓存去除原子操作，通过设置USE_OPTIMIZE进行条件编译
 * 使用方法:./bfs_omp passes num_omp_threads input_file
 */
#include <stdio.h>
#include <stdlib.h>
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
#include <omp.h>
#include <vector>


#define OPTIMIZE_LEVEL 1
//优化等级
//分0,1,2，3
//0 for 原始版本
//1 for 将当前层与下一层状态修改分开
//2 for 在原始版本基础上使用局部缓存去除原子竞争

// CSR二进制文件头（兼容Gunrock）
struct CSRHeader {
    uint32_t magic;      // 魔数校验
    uint32_t num_nodes;  // 节点数
    uint32_t num_edges;  // 边数
    uint32_t _padding=0;   // 填充字段
};

// 打印十六进制值的辅助函数
void PrintHex(const char* label, uint32_t value) {
    std::cout << std::left << std::setw(15) << label 
              << "0x" << std::hex << std::uppercase 
              << std::setw(8) << std::setfill('0') << value << std::endl;
}

void Usage(int argc, char**argv){
	fprintf(stderr,"Usage: %s <passes> <num_omp_threads> <input_file>\n", argv[0]);
}


//initGraph负责从csr.bin的文件内读取信息，校验后从文件头获取节点数、边数信息(这里的边数已经是edges数组大小，无需再乘以2)，然后读取offset和edges数组
void initGraph(const string &filename, int &no_of_nodes, int &edge_list_size, uint32_t *&offsets, uint32_t *&edges);

//再增加一层中间层，用于从外层遍历不同节点，判断访问与否，未访问则进入bfs_queue调用
//状态数组创建应该放到这里
//传入4个参数:节点数，边表大小，offsets和edges，节点编号从0起
void omp(int no_of_nodes, int edge_list_size, uint32_t *&offsets, uint32_t *&edges, int num_omp_threads);

//该函数按source进入遍历,需要将seq内创建的状态相关数组及存储结果的数组传进去
//版本1,有原子操作
void bfs_omp( int no_of_nodes, int source,
        uint32_t *&offsets, uint32_t *&edges, 
        bool *&h_graph_mask, bool *&h_updating_graph_mask,
        bool *&h_graph_visited, int *&h_cost);
//版本2, h_graph_mask与h_updating_mask分开
void bfs_omp_apart( int no_of_nodes, int source,
            uint32_t *&offsets, uint32_t *&edges, 
            bool *&h_graph_mask, bool *&h_updating_graph_mask,
            bool *&h_graph_visited, int *&h_cost);     

//优化版本，在版本1基础上使用本地缓存去除原子操作
void bfs_omp_optimized( int no_of_nodes, int source,
            uint32_t *&offsets, uint32_t *&edges, 
            bool *&h_graph_mask, bool *&h_updating_graph_mask,
            bool *&h_graph_visited, int *&h_cost, int num_omp_threads);

int main(int argc, char** argv){
	//int num_omp_threads;
	int passes = 10;//程序执行次数,默认为10
    int num_omp_threads = 20;

	//修改输入3个参数，即程序名、执行次数以及文件名
	if(argc != 4){
		Usage(argc, argv);
		exit(0);
	}
	passes = atoi(argv[1]);//第2个参数是程序执行次数
    num_omp_threads = atoi(argv[2]);//第3个参数是线程数量
    string filename = argv[3];//文件名称

	printf("Running bfs_omp\n");

	int no_of_nodes = 0;
	int edge_list_size = 0;
    
    uint32_t *offsets;
    uint32_t *edges;

    //读文件获取信息
	initGraph(filename, no_of_nodes, edge_list_size, offsets, edges);
	//执行passes次
	for(int i = 0; i < passes; i++){
		printf("Pass %d:\n",i);
		omp(no_of_nodes, edge_list_size, offsets, edges, num_omp_threads);
	}

    //清理内存
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
        std::cerr << std::dec << "文件头数据异常: 节点数=" << header.num_nodes 
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
    edge_list_size = header.num_edges;//头文件内的num_edges已经算是实际边数的双倍了

    
    offsets = (uint32_t*) malloc(sizeof(uint32_t) * (no_of_nodes+1));//offset需要多分配1个
    edges = (uint32_t*) malloc(sizeof(uint32_t) * edge_list_size);

    // 读取偏移数组
    ifs.read(reinterpret_cast<char*>(offsets), 
            (no_of_nodes + 1) * sizeof(uint32_t));

    // 读取边索引
    ifs.read(reinterpret_cast<char*>(edges), 
        edge_list_size * sizeof(uint32_t));//这里的数量不能写错了

    // 验证读取完整性
    if (!ifs) {
        std::cerr << "文件读取不完整或已损坏" << std::endl;
        exit(0);
    }
	
	std::cout << "\n文件验证通过，数据结构完整" << std::endl;
}

//剩下的mask,visited,cost等与输出相关的数组在s_bfs内再创建，不然会影响多次执行
void omp(int no_of_nodes, int edge_list_size, uint32_t *&offsets, uint32_t *&edges, int num_omp_threads){
	//该函数功能:
	//分配1个状态相关数组和1个存结果数组，并进行初始化
	//判断节点未访问后调用bfs_queu进行具体节点开始打遍历
	//时间输出
    //内层销毁

	//时间记录开始
    //包含环境参数设置，必要数组创建，主存到设备内存的拷贝、执行，拷贝回，数组销毁
    auto start_t = high_resolution_clock::now();

    //allocate host memory
	//状态相关的数组，mask，visited
	bool *h_graph_mask  = (bool*) malloc(sizeof(bool)*no_of_nodes);//用于判定节点是否在当前层
	bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);//用于更新数据使用
	bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);//用于判断节点访问与否
	// allocate mem for the result on host side
	int *h_cost = (int*) malloc( sizeof(int)*no_of_nodes);//用于存储结果,这里相当于存储所在层次

    // 设置OpenMP线程数
    omp_set_num_threads(num_omp_threads);

	
    //初始化相关数组
    memset(h_graph_mask, 0, no_of_nodes * sizeof(bool));
    memset(h_updating_graph_mask, 0, no_of_nodes * sizeof(bool));
    memset(h_graph_visited, 0, no_of_nodes * sizeof(bool));
    memset(h_cost, -1, no_of_nodes * sizeof(int));

	//遍历所有节点，未访问就进入遍历
    int cnt = 0;
	for(int i = 0; i < no_of_nodes; i++){
		//未访问才进入遍历
		if(!h_graph_visited[i]){
            cnt++;
            #if OPTIMIZE_LEVEL == 0
            bfs_omp(no_of_nodes, i, offsets, edges,
                h_graph_mask, h_updating_graph_mask,
				h_graph_visited, h_cost);
            #elif OPTIMIZE_LEVEL == 1
            bfs_omp_apart(no_of_nodes, i, offsets, edges,
                h_graph_mask, h_updating_graph_mask,
				h_graph_visited, h_cost);
            #elif OPTIMIZE_LEVEL == 2
            bfs_omp_optimized(no_of_nodes, i, offsets, edges,
                h_graph_mask, h_updating_graph_mask,
				h_graph_visited, h_cost, num_omp_threads);
            #endif 
		}
	}
	//时间记录结束
    //记录的时间包括执行参数设置，主存到设备内存拷贝，执行，拷贝回主存
    auto end_t = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end_t - start_t);
    double total_time = double(duration.count()) * microseconds::period::num / microseconds::period::den;


    //时间统输出一记录
    printf("Time record(seconds)\n");
    printf("total_time : %f\n", total_time);
    //printf("transfer_time : %f\n", transfer_time);
    //printf("kernel_time : %f\n", kernel_time);
    printf("graph_block : %d\n", cnt);
    //printf("kernel_exe_times : %d\n", k);

    //清理内存
    free(h_graph_mask);
    free(h_updating_graph_mask);
    free(h_graph_visited);
    free(h_cost);
}

//原始版本
void bfs_omp( int no_of_nodes, int source,
	uint32_t *&offsets, uint32_t *&edges, 
    bool *&h_graph_mask, bool *&h_updating_graph_mask,
	bool *&h_graph_visited, int *&h_cost){
	//函数功能
	//设置源点相关状态
	//使用队列遍历source开始的连通块

    //h_graph_mask: 当前活跃节点层
    //​h_updating_graph_mask: 下一层节点收集器
    //每层处理完成后交换掩码数组指针（代码中通过数组值交换实现)

	//set the source node as true in the mask
    h_graph_mask[source] = true;
	h_graph_visited[source]=true;
	h_cost[source]=0;

    
    bool has_active = true;

    // BFS层级迭代
    while (has_active) {
        has_active = false;
        // 并行处理当前层节点
        #pragma omp parallel for reduction(||:has_active)
        for (int node = 0; node < no_of_nodes; ++node) {
            if (h_graph_mask[node]) {
                // 遍历邻接表
                const uint32_t start = offsets[node];
                const uint32_t end = offsets[node + 1];
                for (uint32_t i = start; i < end; ++i) {
                    const uint32_t neighbor = edges[i];
                    // 原子操作避免竞争
                    if (!h_graph_visited[neighbor]) {
                        #pragma omp atomic write
                        h_cost[neighbor] = h_cost[node] + 1;

                        #pragma omp atomic write
                        h_graph_visited[neighbor] = true;

                        #pragma omp atomic write
                        h_updating_graph_mask[neighbor] = true;

                        has_active = true;
                    }
                }
            }
        }

        // 交换掩码并重置下一层
        //值交换
        #pragma omp parallel for
        for (int i = 0; i < no_of_nodes; ++i) {
            h_graph_mask[i] = h_updating_graph_mask[i];
            h_updating_graph_mask[i] = false;
        }
    }
}

//优化版本1,当前层与下一次分开
void bfs_omp_apart( int no_of_nodes, int source,
	uint32_t *&offsets, uint32_t *&edges, 
    bool *&h_graph_mask, bool *&h_updating_graph_mask,
	bool *&h_graph_visited, int *&h_cost){
	//函数功能
	//设置源点相关状态
	//使用队列遍历source开始的连通块

    //h_graph_mask: 当前活跃节点层
    //​h_updating_graph_mask: 下一层节点收集器
    
	//set the source node as true in the mask
    h_graph_mask[source] = true;
	h_graph_visited[source]=true;
	h_cost[source]=0;

    bool has_active = true;

    // BFS层级迭代
    while (has_active) {
        has_active = false;
        // 并行处理当前层节点
        #pragma omp parallel for
        for (int node = 0; node < no_of_nodes; ++node) {
            if (h_graph_mask[node]) {
                h_graph_mask[node] = false;//移出当前层
                // 遍历邻接表
                const uint32_t start = offsets[node];
                const uint32_t end = offsets[node + 1];
                for (uint32_t i = start; i < end; ++i) {
                    const uint32_t neighbor = edges[i];
                    // 无竞争
                    if (!h_graph_visited[neighbor]) {
                        h_cost[neighbor] = h_cost[node] + 1;
                        h_updating_graph_mask[neighbor] = true;
                    }
                }
            }
        }

        // 交换掩码并重置下一层
        #pragma omp parallel for reduction(||:has_active)
        for (int i = 0; i < no_of_nodes; ++i) {
            if(h_updating_graph_mask[i]){
                h_graph_mask[i] = true;//加入当前层
                h_graph_visited[i] = true;
                has_active = true;
                h_updating_graph_mask[i] = false;//重置   
            }           
        }
    }
}

//优化版本optimized，使用swap交换h_graph_mask和​h_updating_graph_mask，并重置​h_updating_graph_mask
void bfs_omp_optimized( int no_of_nodes, int source,
	uint32_t *&offsets, uint32_t *&edges, 
    bool *&h_graph_mask, bool *&h_updating_graph_mask,
	bool *&h_graph_visited, int *&h_cost, int num_omp_threads){
	//函数功能
	//设置源点相关状态
	//使用队列遍历source开始的连通块

    //h_graph_mask: 当前活跃节点层
    //​h_updating_graph_mask: 下一层节点收集器
    //每层处理完成后交换掩码数组指针（代码中通过数组值交换实现)

	//set the source node as true in the mask
    h_graph_mask[source] = true;
	h_graph_visited[source]=true;
	h_cost[source]=0;

    
    bool has_active = true;

    // BFS层级迭代
    while (has_active) {
        has_active = false;
        // 并行处理当前层节点
        #pragma omp parallel for schedule(dynamic, 64) reduction(||:has_active)
        for (int node = 0; node < no_of_nodes; ++node) {
            if (h_graph_mask[node]) {
                // 遍历邻接表
                const uint32_t start = offsets[node];
                const uint32_t end = offsets[node + 1];

                for (uint32_t i = start; i < end; ++i) {
                    const uint32_t neighbor = edges[i];

                    // 原子操作避免竞争
                    if (!h_graph_visited[neighbor]) {
                        #pragma omp atomic write
                        h_graph_visited[neighbor] = true;
                        #pragma omp atomic write
                        h_cost[neighbor] = h_cost[node] + 1;
                        #pragma omp atomic write
                        h_updating_graph_mask[neighbor] = true;

                        has_active = true;
                    }
                }
            }
        }

        // 交换掩码缓冲区
        std::swap(h_graph_mask, h_updating_graph_mask);
        memset(h_updating_graph_mask, 0, no_of_nodes * sizeof(bool)); // 清空 next_mask
    }
}

/*
//再优化版本，复杂度增加了
void bfs_omp_nobranch(
    int no_of_nodes,
    int source,
    uint32_t* offsets,
    uint32_t* edges,
    uint64_t* h_graph_mask,       // 改为位掩码数组 (uint64_t)
    uint64_t* h_updating_graph_mask,
    uint8_t* h_graph_visited,     // 改为字节数组 (0/1)
    int* h_cost,
    int num_omp_threads
) {
    omp_set_num_threads(num_omp_threads);
    constexpr int BIT_WIDTH = 64;
    const int num_words = (no_of_nodes + BIT_WIDTH - 1) / BIT_WIDTH;

    // 初始化：使用 SIMD 指令批量写入
    #pragma omp parallel for simd
    for (int i = 0; i < no_of_nodes; ++i) {
        h_cost[i] = (i == source) ? 0 : -1;
        h_graph_visited[i] = (i == source) ? 1 : 0;
    }

    // 初始化掩码
    #pragma omp parallel for simd
    for (int i = 0; i < num_words; ++i) {
        h_graph_mask[i] = (i == source / BIT_WIDTH) ? 
                          (1ULL << (source % BIT_WIDTH)) : 0;
        h_updating_graph_mask[i] = 0;
    }

    // 主循环（无分支设计）
    while (true) {
        uint64_t global_active = 0;

        #pragma omp parallel reduction(|:global_active)
        {
            __m256i* mask_vec = (__m256i*)h_graph_mask;
            __m256i* update_vec = (__m256i*)h_updating_graph_mask;
            const int vec_chunks = num_words / 4;

            // 向量化处理掩码
            #pragma omp for schedule(static)
            for (int i = 0; i < vec_chunks; ++i) {
                __m256i mask = _mm256_load_si256(&mask_vec[i]);
                __m256i active = _mm256_cmpeq_epi64(mask, _mm256_setzero_si256());
                active = _mm256_xor_si256(active, _mm256_set1_epi64x(-1));
                _mm256_store_si256(&update_vec[i], _mm256_setzero_si256());
                
                if (!_mm256_testz_si256(active, active)) {
                    global_active = 1;
                }
            }

            // 处理剩余字（不足256位部分）
            #pragma omp for schedule(static) nowait
            for (int i = vec_chunks * 4; i < num_words; ++i) {
                if (h_graph_mask[i]) {
                    global_active = 1;
                }
            }
        }

        if (!global_active) break;

        // 无分支邻接遍历
        #pragma omp parallel for schedule(dynamic, 1024) reduction(|:global_active)
        for (int node = 0; node < no_of_nodes; ++node) {
            const uint64_t mask_word = h_graph_mask[node / BIT_WIDTH];
            const uint64_t node_bit = 1ULL << (node % BIT_WIDTH);

            // 使用位运算代替条件判断
            uint64_t is_active = (mask_word & node_bit) >> (node % BIT_WIDTH);
            is_active = (is_active | (is_active >> 1)) & 0x1;  // 转换为 0/1

            const uint32_t start = offsets[node] * is_active;
            const uint32_t end = offsets[node + 1] * is_active;

            // 预取邻接表数据（减少缓存未命中）
            _mm_prefetch((const char*)&edges[start], _MM_HINT_T0);
            _mm_prefetch((const char*)&edges[start + 16], _MM_HINT_T0);

            // 向量化处理邻接边
            for (uint32_t i = start; i < end; i += 8) {
                __m256i neighbors = _mm256_loadu_si256(
                    (__m256i*)&edges[i]);
                __m256i visited = _mm256_loadu_si256(
                    (__m256i*)&h_graph_visited[_mm256_extract_epi32(neighbors, 0)]);

                // 无分支条件更新
                __m256i cost_val = _mm256_set1_epi32(h_cost[node] + 1);
                __m256i mask = _mm256_cmpeq_epi32(visited, _mm256_setzero_si256());
                _mm256_maskstore_epi32((int*)&h_cost[0], mask, cost_val);
                _mm256_maskstore_epi32((int*)&h_graph_visited[0], mask, 
                    _mm256_set1_epi32(1));

                // 更新掩码
                __m256i update_mask = _mm256_slli_epi32(mask, 31);
                _mm256_maskstore_epi64((long long*)h_updating_graph_mask, 
                    update_mask, _mm256_set1_epi64x(1LL << (node % BIT_WIDTH)));
            }
        }

        // 交换掩码（向量化操作）
        #pragma omp parallel for simd
        for (int i = 0; i < num_words; ++i) {
            h_graph_mask[i] = h_updating_graph_mask[i];
            h_updating_graph_mask[i] = 0;
        }
    }
}

*/

