/***
 * 仿照Altis中的模块化bfs代码，修改本处的bfs代码；
 * 用法：./program -p platform -d device -n 1 -i inputfile -k kernelfile -l localworksize -k kernelfile
 */


#ifdef __cplusplus
extern "C" {  // 确保以C风格链接OpenCL函数
#endif

// 根据系统选择头文件路径
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#ifdef __cplusplus
}

#endif

//其他头文件
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>//for recording time
using namespace std::chrono;
#include <fstream> //读文件
#include <cstdint> //for uint32_t
#include <iomanip>// 用于十六进制输出
#include <cstring> //for memset
#include <cfloat>//for FLT_MAX
#include <algorithm>

#include "OptionParser.h"//for arguments parse

//宏定义
#define CHKERR(err, str) \
    if (err != CL_SUCCESS) \
    { \
        fprintf(stdout, "CL Error %d: %s\n", err, str); \
        exit(1); \
    }

// CSR二进制文件头
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

//initGraph负责从csr.bin的文件内读取信息，校验后从文件头获取节点数、边数信息(这里的边数已经是edges数组大小，无需再乘以2)，然后读取offset和edges数组
void initGraph(const std::string &filename, int &no_of_nodes, int &edge_list_size, uint32_t *&offsets, uint32_t *&edges);

//程序参数初始化
void opinit(OptionParser &op);

//根据硬件信息配置工作组
void GetOptimalWorkSize(const cl_context &context, int device, int no_of_nodes, int &localWork, 
	size_t *WorkSize, size_t *localWorkSize);


//再增加一层中间层，用于从外层遍历不同节点，判断访问与否，未访问则进入bfs_opencl调用
//相关状态数组创建、释放应该放到这里
void ocl(int no_of_nodes, int edge_list_size, uint32_t *&h_offsets, uint32_t *&h_edges,
	int device, cl_context &context, cl_command_queue &commands, int &localWork, cl_kernel kernel1, cl_kernel kernel2);

//bfs_opencl负责进入具体点进行执行，相当于遍历连通分量
void bfs_opencl(int no_of_nodes, int source,
		bool *&h_graph_mask, bool *&h_graph_visited, int *&h_cost, 
		cl_mem &d_graph_mask, cl_mem &d_graph_visited, cl_mem &d_cost, cl_mem &d_over,
		cl_command_queue &commands, size_t *WorkSize, size_t *localWorkSize, 
		cl_kernel kernel1, cl_kernel kernel2, cl_event &ocdEvent, int &k, double &kernel_time, double &transfer_time);

//辅助函数用于初始化
cl_context create_context(cl_platform_id platform, cl_device_id device);
cl_command_queue create_command_queue(cl_context context, cl_device_id device);

//进行opencl初始化，根据输入的硬件相关参数进行平台、设备选择和验证，计算环境设置
void ocd_initCL(int platform, int device, cl_context &context, cl_command_queue &commond);

// 辅助函数：读取内核文件内容
std::string read_kernel_file(const std::string &filename);

// 辅助函数：获取上下文关联的设备列表
std::vector<cl_device_id> get_context_devices(cl_context context);

//辅助函数，用于根据设备选择优化等级
std::string get_device_specific_options(cl_device_id device);

//动态编译模块
void compile_kernel(const std::string &kernelfilename, cl_context &context, int platform_index,
	int device_index,cl_kernel &kernel1, cl_kernel &kernel2);

void getElapsedTime(double &elapsedTime, cl_event event);

int main(int argc, char** argv){
    //参数预设置
    OptionParser op;
    opinit(op);
    if (!op.parse(argc, argv)){
        op.usage();
        return (op.HelpRequested() ? 0 : 1);
    }

    //解析本程序所用参数
    std::string inputfile = op.getOptionString("inputFile");
    int passes = op.getOptionInt("passes");
    int localWork = op.getOptionInt("localworksize");
	//平台和设备选择
	int platform = op.getOptionInt("platform");
	int device = op.getOptionInt("device");
	//kernel文件名
	std::string kernelfile = op.getOptionString("kernelFile");

	printf("kernelfilename is %s\n", kernelfile);

	//初始化环境
	//ocd_initCL该函数初始化opencl环境，包括选择设备、创建计算环境等
	cl_context context;
	cl_command_queue commands;
	ocd_initCL(platform, device, context, commands);
	
	int no_of_nodes = 0;
	int edge_list_size = 0;
    uint32_t *offsets;
    uint32_t *edges;

    //读文件获取信息
	initGraph(inputfile, no_of_nodes, edge_list_size, offsets, edges);

	//进行动态编译并记录时间
	auto start_t = high_resolution_clock::now();

	// 编译内核（显示完整设备信息）
	cl_kernel kernel1;
	cl_kernel kernel2;
	compile_kernel(kernelfile, context, platform, device, kernel1, kernel2);

	auto end_t = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(end_t - start_t);
	double duration_t = double(duration.count()) * microseconds::period::num / microseconds::period::den;
	printf("compile_opencl_time is %f seconds\n",duration_t);

    //执行passes次，根据参数判断使用啥程序，普通还是uvm相关
    printf("Running bfs_cuda\n");
	for (int i = 0; i < passes; i++) {
        printf("Pass %d:\n", i);
		ocl(no_of_nodes, edge_list_size, offsets, edges,
			device, context, commands, localWork, kernel1, kernel2);
    }

	//释放kernel,program已在编译函数内释放
	clReleaseKernel(kernel1);
	clReleaseKernel(kernel2);
	//commandqueue和context应该放到这里释放
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

    //清理内存,这个内存分配发生在initGraph内
    free(offsets);
    free(edges);
	return 0;
}


//程序参数初始化
void opinit(OptionParser &op){
    // Add shared options to the parser
    op.addOption("passes", OPT_INT, "10", "specify number of passes", 'n');
    op.addOption("verbose", OPT_BOOL, "0", "enable verbose output", 'v');
    op.addOption("quiet", OPT_BOOL, "0", "enable concise output", 'q');
    op.addOption("inputFile", OPT_STRING, "", "path of input file", 'i');
    // op.addOption("outputFile", OPT_STRING, "", "path of output file", 'o');
    op.addOption("platform", OPT_VECINT, "1", "specify which platform to run on", 'p');
	//clinfo -l , 1 for cpu, 2 for igpu, 3 for ngpu
	op.addOption("device", OPT_VECINT, "0", "specify device to run on", 'd');
	//0 for default device
	op.addOption("localworksize", OPT_VECINT, "0", "specify localWorkSize to run on", 'l');
	op.addOption("kernelFile", OPT_STRING, "../src/bfs_kernel.cl", "kernel file", 'k');
	//可指定编译kernel文件的名字,默认为bfs_kernel.cl
}

cl_context create_context(cl_platform_id platform, cl_device_id device) {
	cl_int err;
	cl_context_properties props[] = {
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platform,
		0
	};
	
	cl_context context = clCreateContext(props, 1, &device, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Error creating context: %d\n", err);
		exit(EXIT_FAILURE);
	}
	return context;
}

cl_command_queue create_command_queue(cl_context context, cl_device_id device) {
    cl_int err;
	cl_queue_properties queue_props[] = {
		CL_QUEUE_PROPERTIES,       // 属性名：队列类型
		CL_QUEUE_PROFILING_ENABLE, // 属性值：启用性能分析
		0                          // 终止符（必须为0）
	};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, queue_props, &err);
	//创建队列必须启用CL_QUEUE_PROFILING_ENABLE才能使用事件记录时间
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating command queue: %d\n", err);
        exit(EXIT_FAILURE);
    }
    return queue;
}

// 辅助函数：读取内核文件内容
std::string read_kernel_file(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Failed to open kernel file %s\n", filename.c_str());
        exit(EXIT_FAILURE);
    }
    return std::string(std::istreambuf_iterator<char>(file), 
                      std::istreambuf_iterator<char>());
}

// 辅助函数：获取上下文关联的设备列表
std::vector<cl_device_id> get_context_devices(cl_context context) {
    cl_uint num_devices = 0;
    clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL);
    
    std::vector<cl_device_id> devices(num_devices);
    clGetContextInfo(context, CL_CONTEXT_DEVICES, num_devices * sizeof(cl_device_id), devices.data(), NULL);
    return devices;
}

std::string get_device_specific_options(cl_device_id device) {
    cl_device_type dtype;
    char vendor[128], name[128];
    
    // 获取设备信息
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(dtype), &dtype, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);

    // Intel CPU优化
    if (dtype == CL_DEVICE_TYPE_CPU && strstr(vendor, "Intel")) {
        return "-cl-mad-enable -cl-no-signed-zeros";
		//-cl-mad-enable：允许将乘加操作合并为单条指令（如FMA指令集）
		//-cl-no-signed-zeros：忽略符号零处理，提升浮点运算速度
    }
    
    // Intel核显优化
    if (dtype == CL_DEVICE_TYPE_GPU && strstr(vendor, "Intel")) {
		// return "-cl-fast-relaxed-math -cl-single-precision-constant";
		//-cl-single-precision-constant：强制单精度常量优化
        return "-cl-single-precision-constant";
    }
	//-cl-fast-relaxed-math允许编译器重新排列浮点运算顺序
	//该选项可能导致浮点精度误差累积，关闭
    
    // NVIDIA GPU优化
    if (strstr(vendor, "NVIDIA")) {
		//return "-cl-nv-verbose -cl-fast-relaxed-math -cl-nv-opt-level=3";
		//-cl-nv-verbose：显示详细的PTX汇编输出（调试用）
		//-cl-nv-opt-level=3：启用最高级别优化（寄存器重用等）
        return "-cl-nv-verbose -cl-nv-opt-level=3";
    }

    // 默认选项
    return "-cl-opt-disable";  // 当设备不匹配时禁用优化
}

//kernel编译模块
void compile_kernel(const std::string &kernelfilename, cl_context &context, int platform_index,
	int device_index,cl_kernel &kernel1, cl_kernel &kernel2){
	// 获取上下文关联设备列表
	auto devices = get_context_devices(context);
	if (device_index >= devices.size()) {
		fprintf(stderr, "Device index %d invalid (max %zu)\n", 
		device_index, devices.size()-1);
		exit(EXIT_FAILURE);
	}
	cl_device_id device = devices[device_index];

	// 获取平台和设备详细信息
	cl_platform_id platform;
	clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL);

	char platform_name[128] = {0};
	char device_name[128] = {0};
	clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
	clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);

	// 设备详细信息查询
	cl_uint compute_units;
	clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);

	// 智能选择编译选项
    std::string options = get_device_specific_options(device);

	printf("\n[OpenCL编译环境]\n");
	printf("▌ 平台索引: %d\n", platform_index);
	printf("▌ 平台名称: %s\n", platform_name);
	printf("▌ 设备索引: %d\n", device_index);
	printf("▌ 设备名称: %s\n", device_name);
	printf("▌ 计算单元: %u\n", compute_units);
	printf("▌ 编译选项: %s\n", options.c_str());
	printf("▌ 内核文件: %s\n\n", kernelfilename.c_str());

	// 读取内核源码
	const std::string kernel_source = read_kernel_file(kernelfilename);
	const char* source_str = kernel_source.c_str();
	size_t source_size = kernel_source.size();

	// 创建程序对象
	cl_int err;
	cl_program program = clCreateProgramWithSource(context, 1, &source_str, &source_size, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "创建程序对象失败: %d\n", err);
		exit(EXIT_FAILURE);
	}

	// 编译程序（带详细日志）
	//err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	err = clBuildProgram(program, 1, &device, options.c_str(), NULL, NULL);//启用自动编译选项,会根据设备类型选定优化选项
	if (err != CL_SUCCESS) {
		size_t log_size;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		std::vector<char> log(log_size);
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);

		fprintf(stderr, "编译失败 (错误码 %d):\n%s\n", err, log.data());
		exit(EXIT_FAILURE);
	}

	// 创建内核对象
	kernel1 = clCreateKernel(program, "kernel1", &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "创建kernel1失败: %d (检查函数名是否匹配)\n", err);
		exit(EXIT_FAILURE);
	}

	kernel2 = clCreateKernel(program, "kernel2", &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "创建kernel2失败: %d (检查函数名是否匹配)\n", err);
		exit(EXIT_FAILURE);
	}

	// 释放程序对象
	clReleaseProgram(program);

	printf("✔ 成功编译内核: kernel1, kernel2\n");
	printf("-----------------------------------\n");
}

//opencl相关初始化
void ocd_initCL(int platform_id, int device_id, cl_context &context, cl_command_queue &commands)
{
	//根据输入的平台和设备的id进行对contex和command的初始化;

	//外部平台和设备转换
    cl_uint platform_index = platform_id;
    cl_uint device_index = device_id;

    // 获取所有平台
    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms, NULL);

    // 验证平台索引
    if (platform_index >= num_platforms) {
        fprintf(stderr, "Invalid platform index (max %u)\n", num_platforms - 1);
        exit(1);
    }

    // 获取平台下的设备
    cl_uint num_devices;
    clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (num_devices == 0) {
        fprintf(stderr, "No devices found on platform %u\n", platform_index);
        exit(1);
    }

    cl_device_id* devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
    clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

    // 验证设备索引
    if (device_index >= num_devices) {
        fprintf(stderr, "Invalid device index (max %u)\n", num_devices - 1);
        exit(1);
    }

    // 创建上下文和命令队列
    context = create_context(platforms[platform_index], devices[device_index]);
	//已在创建函数内开启事件时间记录
    commands = create_command_queue(context, devices[device_index]);
	

    printf("Successfully initialized OpenCL context on:\n");
    printf("  Platform %u: ", platform_index);
    char platform_name[128];
    clGetPlatformInfo(platforms[platform_index], CL_PLATFORM_NAME, 128, platform_name, NULL);
    printf("%s\n", platform_name);
    
    printf("  Device %u: ", device_index);
    char device_name[128];
    clGetDeviceInfo(devices[device_index], CL_DEVICE_NAME, 128, device_name, NULL);
    printf("%s\n", device_name);

    // 清理资源
    free(platforms);
    free(devices);
}

//从文件读图
void initGraph(const std::string &filename, int &no_of_nodes, int &edge_list_size, uint32_t *&offsets, uint32_t *&edges){
	
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


void ocl(int no_of_nodes, int edge_list_size, uint32_t *&h_offsets, uint32_t *&h_edges,
	int device, cl_context &context, cl_command_queue &commands, int &localWork, cl_kernel kernel1, cl_kernel kernel2){
	//函数功能:
	//分配状态相关数组、存结果数组并初始化
	//分配设备内存，避免每次在内层函数分配
		//部分全局拷贝一次即可,如nodes和edges数组
		//部分需要全局统一拷贝一次，变更时还需要单独拷贝具体数据,如visited，dcost，mask
	//设置执行环境、参数等
	//调用内存函数
	//结果d_cost拷贝回主存
	//输入统一时间和数据
	//释放内存，设备+主机
	
	//仅记录有效的时间，即环境设置,主存到设备传送，执行，拷贝回
	//时间记录开始
	//包含环境参数设置，主存到设备内存的拷贝、执行，拷贝回
	auto start_t = high_resolution_clock::now();
	
	double total_time = 0;
    double kernel_time = 0;
    double transfer_time = 0;

	// allocate host memory
    bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
    bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);
    int *h_cost = (int*) malloc( sizeof(int)*no_of_nodes);

	//初始化相关数组
	memset(h_graph_mask, 0, no_of_nodes * sizeof(bool));
    memset(h_updating_graph_mask, 0, no_of_nodes * sizeof(bool));
    memset(h_graph_visited, 0, no_of_nodes * sizeof(bool));
    memset(h_cost, -1, no_of_nodes * sizeof(int));
	

	//设备内存创建
	//计算大小
	int offsets_size = sizeof(uint32_t) * (no_of_nodes + 1);
	int edges_size = sizeof(uint32_t) * edge_list_size;


	cl_int err;
	cl_event ocdTempEvent;
	//创建设备内存并拷贝到设备端
	//offsets
	cl_mem  d_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, offsets_size, NULL, &err);
	//edges
	cl_mem  d_edges = clCreateBuffer(context, CL_MEM_READ_ONLY, edges_size, NULL, &err);
	//Mask
	cl_mem  d_graph_mask = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(bool) * no_of_nodes, NULL, &err);
	//updating graph mask
	cl_mem  d_updating_graph_mask = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(bool) * no_of_nodes, NULL, &err);
	//Visited nodes
	cl_mem  d_graph_visited = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(bool) * no_of_nodes, NULL,  &err);
	//Allocate device memory for result
	cl_mem d_cost = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * no_of_nodes, NULL, &err);
	//Make a bool to check if the execution is over
	cl_mem d_over = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(bool), NULL, &err);

	//拷贝相关主存信息到设备
	//clEnqueWriteBuffer：host-->device
	//offsets
	clEnqueueWriteBuffer(commands, d_offsets, CL_TRUE, 0, offsets_size, h_offsets, 0, NULL, &ocdTempEvent);
	clFinish(commands);
	double elapsedTime = 0;
	getElapsedTime(elapsedTime, ocdTempEvent);
	transfer_time += elapsedTime;
	//edges
	clEnqueueWriteBuffer(commands, d_edges, CL_TRUE, 0, edges_size, h_edges, 0, NULL, &ocdTempEvent);
	clFinish(commands);
	getElapsedTime(elapsedTime, ocdTempEvent);
	transfer_time += elapsedTime;

	//d_updating_graph_mask也是拷贝一次，只有kernel期间会更改，其余地方未变化
	clEnqueueWriteBuffer(commands, d_updating_graph_mask, CL_TRUE, 0, sizeof(bool) * no_of_nodes, h_updating_graph_mask, 0, NULL, &ocdTempEvent);
	clFinish(commands);
	getElapsedTime(elapsedTime, ocdTempEvent);
	transfer_time += elapsedTime;

	//以下三种数组，内层每次需要设置source相关状态，因此使用时每次只拷贝一个值就行，但要统一在第一次执行时全部拷贝一次
	clEnqueueWriteBuffer(commands, d_graph_mask, CL_TRUE, 0, sizeof(bool) * no_of_nodes, h_graph_mask, 0, NULL, &ocdTempEvent);
	clFinish(commands);
	getElapsedTime(elapsedTime, ocdTempEvent);
	transfer_time += elapsedTime;

	clEnqueueWriteBuffer(commands, d_graph_visited, CL_TRUE, 0, sizeof(bool) * no_of_nodes, h_graph_visited, 0, NULL, &ocdTempEvent);
	clFinish(commands);
	getElapsedTime(elapsedTime, ocdTempEvent);
	transfer_time += elapsedTime;

	clEnqueueWriteBuffer(commands, d_cost, CL_TRUE, 0, sizeof(int) * no_of_nodes, h_cost, 0, NULL, &ocdTempEvent);
	clFinish(commands);
	getElapsedTime(elapsedTime, ocdTempEvent);
	transfer_time += elapsedTime;

	//Set Arguments for Kernel1 and 2
	clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void*)&d_offsets);
	clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void*)&d_edges);
	clSetKernelArg(kernel1, 2, sizeof(cl_mem), (void*)&d_graph_mask);
	clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void*)&d_updating_graph_mask);
	clSetKernelArg(kernel1, 4, sizeof(cl_mem), (void*)&d_graph_visited);
	clSetKernelArg(kernel1, 5, sizeof(cl_mem), (void*)&d_cost);
	clSetKernelArg(kernel1, 6, sizeof(int), (void*)&no_of_nodes);
	//clSetKernelArg(kernel1, 7, sizeof(unsigned int), (void*)&useThreads);
	
	clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void*)&d_graph_mask);
	clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void*)&d_updating_graph_mask);
	clSetKernelArg(kernel2, 2, sizeof(cl_mem), (void*)&d_graph_visited);
	clSetKernelArg(kernel2, 3, sizeof(cl_mem), (void*)&d_over);
	clSetKernelArg(kernel2, 4, sizeof(int), (void*)&no_of_nodes);
	//clSetKernelArg(kernel2, 5, sizeof(unsigned int), (void*)&useThreads);

	//设置kernel启动参数
	size_t WorkSize[1];
	size_t localWorkSize[1];
	GetOptimalWorkSize(context, device, no_of_nodes, localWork, WorkSize, localWorkSize);
	

	//遍历所有节点，未访问就进入遍历
	//这里是顺序的
	int cnt = 0;//记录连通块数量
	int k = 0;//记录kernel执行次数
	for(int i = 0; i < no_of_nodes; i++){
		if(!h_graph_visited[i]){
			++cnt;
			bfs_opencl(no_of_nodes, i,
				h_graph_mask, h_graph_visited, h_cost, 
				d_graph_mask, d_graph_visited, d_cost, d_over,
				commands, WorkSize, localWorkSize, kernel1, kernel2,
				ocdTempEvent, k, kernel_time, transfer_time);
		}
	}

	//统一拷贝回值
    //只有d_cost拷贝回传
	clEnqueueReadBuffer(commands, d_cost, CL_TRUE, 0, sizeof(int)*no_of_nodes, (void*)h_cost, 0, NULL, &ocdTempEvent);
	clFinish(commands);
	getElapsedTime(elapsedTime, ocdTempEvent);
	transfer_time += elapsedTime;

	//c++记录时间结束
	//记录的时间包含主存到设备内存拷贝，执行，结果传回
	auto end_t = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(end_t - start_t);
	total_time = double(duration.count()) * microseconds::period::num / microseconds::period::den;
	
    printf("total_time is %f seconds\n", total_time);
    printf("transfer_time is %f seconds\n", transfer_time);
    printf("kernel_time is %f seconds\n", kernel_time);
    printf("graph_block is %d\n", cnt);
    printf("kernel_exe_times are %d\n", k);

	// 释放事件
    clReleaseEvent(ocdTempEvent);

	//device mem release
	clReleaseMemObject(d_offsets);
	clReleaseMemObject(d_edges);
	clReleaseMemObject(d_graph_mask);
	clReleaseMemObject(d_updating_graph_mask);
	clReleaseMemObject(d_graph_visited);
	clReleaseMemObject(d_cost);
	clReleaseMemObject(d_over);

	//释放状态相关数组，nodes和edges相关数组是在外面释放的
	free(h_graph_mask);
	free(h_updating_graph_mask);
	free(h_graph_visited);
	free(h_cost);
}

//bfs_opencl负责计算环境设置、设备内存分配、拷贝，计算，结果返回主存
void bfs_opencl(int no_of_nodes, int source,
	 bool *&h_graph_mask, bool *&h_graph_visited, int *&h_cost, 
	 cl_mem &d_graph_mask, cl_mem &d_graph_visited, cl_mem &d_cost, cl_mem &d_over,
	 cl_command_queue &commands, size_t *WorkSize, size_t *localWorkSize, cl_kernel kernel1, 
	 cl_kernel kernel2, cl_event &ocdEvent, int &k, double &kernel_time, double &transfer_time){
	
	//set the source node
	h_graph_mask[source]=true;
	h_graph_visited[source]=true;
	h_cost[source]=0;

	//需要将更改的source相关值传递到设备端
	//只拷贝source节点，该值取值范围为[0,no_of_nodes)
	//这里与cuda不同，cuda偏移量就是元素，而opencl内是字节偏移量
	int byte_offset_b = sizeof(bool) * source;//b for bool，设备端偏移量
	int byte_offset_i = sizeof(int) * source;//i for int，设备端偏移量
    //拷贝source相关值
	clEnqueueWriteBuffer(commands, d_graph_mask, CL_TRUE, byte_offset_b, sizeof(bool), &h_graph_mask[source], 0, NULL, &ocdEvent);
	clFinish(commands);
	double elapsedTime = 0;
	getElapsedTime(elapsedTime, ocdEvent);
	transfer_time += elapsedTime;

	clEnqueueWriteBuffer(commands, d_graph_visited, CL_TRUE, byte_offset_b, sizeof(bool), &h_graph_mask[source], 0, NULL, &ocdEvent);
	clFinish(commands);
	getElapsedTime(elapsedTime, ocdEvent);
	transfer_time += elapsedTime;

	clEnqueueWriteBuffer(commands, d_cost, CL_TRUE, byte_offset_i, sizeof(int), &h_graph_mask[source], 0, NULL, &ocdEvent);
	clFinish(commands);
	getElapsedTime(elapsedTime, ocdEvent);
	transfer_time += elapsedTime;

	bool stop;
	do
	{
		stop = false;
		//Copy stop to device
		clEnqueueWriteBuffer(commands, d_over, CL_TRUE, 0, sizeof(bool), (void*)&stop, 0, NULL, &ocdEvent);
		clFinish(commands);
		getElapsedTime(elapsedTime, ocdEvent);
		transfer_time += elapsedTime;

		//Run Kernel1 
		cl_int err = clEnqueueNDRangeKernel(commands, kernel1, 1, NULL,
				WorkSize, localWorkSize, 0, NULL, &ocdEvent);
		clFinish(commands);
		getElapsedTime(elapsedTime, ocdEvent);
		kernel_time += elapsedTime;

		if(err != CL_SUCCESS)
			printf("Error occurred running kernel1.(%d)\n", err);
		
		//Run Kernel 2
		err = clEnqueueNDRangeKernel(commands, kernel2, 1, NULL,
				WorkSize, localWorkSize, 0, NULL, &ocdEvent);
		clFinish(commands);
		getElapsedTime(elapsedTime, ocdEvent);
		kernel_time += elapsedTime;

		if(err != CL_SUCCESS)
			printf("Error occurred running kernel2.\n");

		//Copy stop from device
		clEnqueueReadBuffer(commands, d_over, CL_TRUE, 0, sizeof(bool), (void*)&stop, 0, NULL, &ocdEvent);
		clFinish(commands);
		getElapsedTime(elapsedTime, ocdEvent);
		transfer_time += elapsedTime;

		k++;
	}while(stop);

	//三个状态数组
	//d_graph_mask无需拷贝回去，外面没用到，留在设备上共享就行
	//d_updating_graph_mask也无需拷贝回去，只有kernel内用到
	//d_graph_visited只需要拷贝source开始的，即[source, no_of_nodes)
	//首先计算偏移及拷贝量
	int byte_offset = (source) * sizeof(bool);//设备端的字节偏移量
	int count = no_of_nodes-source;//左闭右开区间
	bool *host_ptr = h_graph_visited + source;
	clEnqueueReadBuffer(commands, d_graph_visited, CL_TRUE, byte_offset, sizeof(bool)*count, (void*)host_ptr, 0, NULL, &ocdEvent);
	clFinish(commands);
	getElapsedTime(elapsedTime, ocdEvent);
	transfer_time += elapsedTime;
}

// 自动选择最佳本地工作组大小
//传参数进去后，自动设置localWork， WorkSize， localWorkSize
void GetOptimalWorkSize(const cl_context &context, int device, int no_of_nodes, int &localWork, 
	size_t *WorkSize, size_t *localWorkSize){
    //cuda与opencl术语对应关系
	//Thread		Work Item   			完全对应，最小执行单元
	//Block	    	Work Group  			同步方式相同 (barrier)
	//​Grid	         NDRange     			 OpenCL支持1D/2D/3D，CUDA固定3D
	//blockDim.x	local_work_size[0]      工作组内X维度工作项数
	//gridDim.x	    global_work_size[0]/local_work_size[0]  OpenCL需手动计算
	size_t maxThreads[3];
	//这里获取3个维度的信息，但本程序实际只用了1维
	std::vector<cl_device_id> devices = get_context_devices(context);
	cl_device_id device_id = devices[device];
	//查询设备各维度的线程数限制
	cl_int err = clGetDeviceInfo(device_id,CL_DEVICE_MAX_WORK_ITEM_SIZES,sizeof(size_t)*3, &maxThreads, NULL);
	CHKERR(err, "Error checking for work item sizes\n");
	size_t maxWorkGroupSize;
	//查询单个工作组中所有维度的总线程数上限
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
	//检查localWork合法性
	//参数内有localWork，用户未指定时默认值为0
	//maxThreads对应cuda的blockDim.x的最大值
	int good_size = std::min(maxThreads[0], maxWorkGroupSize);
	if(localWork > 0){
		localWork = std::min(good_size, localWork);
	}else{
		//未指定localWork时default 为0
		localWork = good_size;
	}
	//设置全局工作项总数，根据总节点数变化
	//WorkSize对应CUDA中的gridDim.x * blockDim.x（1d）
	WorkSize[0] = {((no_of_nodes + localWork - 1) / localWork) * localWork};
	//localWorkSize对应CUDA的Block Size
	//最终确定本地工作组大小
	localWorkSize[0] = {localWork};//这里用1d相当于blockDim.x
}


void getElapsedTime(double &elapsedTime, cl_event event) {
    cl_int err = CL_SUCCESS;
    
    // 2. 获取时间戳
    cl_ulong start_time = 0, end_time = 0;
    
    // 获取开始时间
    err = clGetEventProfilingInfo(
        event, 
        CL_PROFILING_COMMAND_START, 
        sizeof(cl_ulong), 
        &start_time, 
        nullptr
    );
    
    // 获取结束时间
    err = clGetEventProfilingInfo(
        event, 
        CL_PROFILING_COMMAND_END, 
        sizeof(cl_ulong), 
        &end_time, 
        nullptr
    );
    
    // 3. 计算耗时（纳秒 → 秒）
    elapsedTime = static_cast<double>(end_time - start_time) / 1e9;
}