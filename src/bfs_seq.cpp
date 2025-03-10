/***
 * 函数功能：使用基于队列的版本对图进行遍历，不管图有没有连通
 * 更新：将涉及vector的数组全换为普通数组
 * 
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
	fprintf(stderr,"Usage: %s <passes> <input_file>\n", argv[0]);
}


//initGraph负责从csr.bin的文件内读取信息，校验后从文件头获取节点数、边数信息(这里的边数已经是edges数组大小，无需再乘以2)，然后读取offset和edges数组
void initGraph(const string &filename, int &no_of_nodes, int &edge_list_size, uint32_t *&offsets, uint32_t *&edges);

//再增加一层中间层，用于从外层遍历不同节点，判断访问与否，未访问则进入bfs_queue调用
//状态数组创建应该放到这里
//传入4个参数:节点数，边表大小，offsets和edges，节点编号从0起
void seq(int no_of_nodes, int edge_list_size, uint32_t *&offsets, uint32_t *&edges);

//该函数按source进入遍历,需要将seq内创建的状态相关数组及存储结果的数组传进去
void bfs_queue( int no_of_nodes, int source,
	uint32_t *&offsets, uint32_t *&edges, 
	bool *&h_graph_visited, int *&h_cost);


int main(int argc, char** argv){
	//int num_omp_threads;
	int passes = 10;//程序执行次数,默认为10

	//修改输入3个参数，即程序名、执行次数以及文件名
	if(argc != 3){
		Usage(argc, argv);
		exit(0);
	}
	passes = atoi(argv[1]);//第2个参数是程序执行次数
	//num_omp_threads = atoi(argv[1]);//在这个程序中该参数无用
    string filename = argv[2];//文件名称

	printf("Running bfs_seq\n");
	int no_of_nodes = 0;
	int edge_list_size = 0;
    
    uint32_t *offsets;
    uint32_t *edges;

    //读文件获取信息
	initGraph(filename, no_of_nodes, edge_list_size, offsets, edges);
	//执行passes次
	for(int i = 0; i < passes; i++){
		printf("Pass %d:\n",i);
		seq(no_of_nodes, edge_list_size, offsets, edges);
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

//剩下的mask,visited,cost等与输出相关的数组在s_bfs内再创建，不然会影响多次执行
void seq(int no_of_nodes, int edge_list_size, uint32_t *&offsets, uint32_t *&edges){
	//该函数功能:
	//分配1个状态相关数组和1个存结果数组，并进行初始化
	//判断节点未访问后调用bfs_queu进行具体节点开始打遍历
	//时间+连通块个数输出

	//时间记录开始
    //包含环境参数设置，必要数组创建，主存到设备内存的拷贝、执行，拷贝回，数组销毁
    auto start_t = high_resolution_clock::now();

	//allocate host memory
	//状态相关的数组，visited
    bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);
	// allocate mem for the result on host side
    int *h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
    //初始化相关数组
    memset(h_graph_visited, 0, no_of_nodes * sizeof(bool));
    memset(h_cost, -1, no_of_nodes * sizeof(int));

	//遍历所有节点，未访问就进入遍历
    int cnt = 0;//统计连通块
	for(int i = 0; i < no_of_nodes; i++){
		//未访问才进入遍历
		if(!h_graph_visited[i]){
            cnt++;
			bfs_queue(no_of_nodes, i, offsets, edges,
				h_graph_visited, h_cost);
		}
	}
	//时间记录结束
    //记录的时间包括执行参数设置，主存到设备内存拷贝，执行，拷贝回主存
    auto end_t = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end_t - start_t);
    double total_time = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    printf("total_time is %f seconds\n", total_time);
    printf("graph_block is %d\n", cnt);
    //清理内存
    free(h_graph_visited);
    free(h_cost);
}

void bfs_queue( int no_of_nodes, int source,
	uint32_t *&offsets, uint32_t *&edges, 
	bool *&h_graph_visited, int *&h_cost){
	//函数功能
	//设置源点相关状态
	//使用队列遍历source开始的连通块

	//set the source node as true in the mask
	h_graph_visited[source]=true;
	h_cost[source]=0;


	//使用队列进行访问
	std::queue<uint32_t> q;
    q.push(source);
	while (!q.empty()) {
        uint32_t current = q.front();
        q.pop();
        // 遍历当前节点的所有邻居
        uint32_t start = offsets[current];
        uint32_t end = offsets[current + 1];
        
        for (uint32_t i = start; i < end; ++i) {
            uint32_t neighbor = edges[i];
            
            if (!h_graph_visited[neighbor]) {
                h_cost[neighbor] = h_cost[current] + 1;
                h_graph_visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}





