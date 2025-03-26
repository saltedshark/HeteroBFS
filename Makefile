# 编译器配置
CXX := g++
CXXFLAGS := -std=c++11 -Wall
LDFLAGS := -lpthread

# OpenMP编译选项
OMP_CXXFLAGS := -fopenmp
OMP_LDFLAGS := -lgomp


# CUDA 编译器配置
NVCC := nvcc
CUDA_ARCH := sm_86  # 根据实际GPU架构修改

# 公共CUDA选项
COMMON_NVCCFLAGS := -O3 -Isrc/include

# 普通CUDA程序选项
CUDA_NVCCFLAGS := $(COMMON_NVCCFLAGS) \
                  -arch=$(CUDA_ARCH) \
                  -std=c++11 \
				  --expt-extended-lambda

CUDA_LDFLAGS := -lcudart -lcudadevrt

# cuGraph专用选项（包含特殊实验性选项）（假设通过 conda 安装）
CUGRAPH_NVCCFLAGS := $(COMMON_NVCCFLAGS) \
                     --expt-extended-lambda \
                     --expt-relaxed-constexpr \
                     -std=c++17 \
                     -gencode arch=compute_86,code=sm_86 \
                     -I$(CONDA_PREFIX)/include
				
CUGRAPH_LDFLAGS := -L$(CONDA_PREFIX)/lib \
                   -lcugraph_c \
                   -lcugraph \
                   -lcudart \
                   -Xlinker "-rpath,$(CONDA_PREFIX)/lib"

# OpenCL配置
OPENCL_CXXFLAGS := -I/usr/include/CL       # 根据实际头文件路径调整
OPENCL_LDFLAGS := -lOpenCL                 # 链接 OpenCL 库

# SNAP库路径
SNAP_CORE := extern/snap/snap-core
SNAP_LIB := $(SNAP_CORE)/libsnap.a
SNAP_INCLUDES := -Iextern/snap/glib-core -I$(SNAP_CORE)
SNAP_LDFLAGS := -L$(SNAP_CORE) -lsnap $(LDFLAGS)
SNAP_CXXFLAGS := -fopenmp      # 仅SNAP程序需要的编译选项
SNAP_LDFLAGS := -L$(SNAP_CORE) -lsnap -lgomp  # 仅SNAP程序需要的链接选项

# 路径定义
SRC_DIR := src
BIN_DIR := bin
#存放公共头文件的文件夹
INCLUDE_DIR := $(SRC_DIR)/include
SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
EXECUTABLES := $(patsubst $(SRC_DIR)/%.cpp, $(BIN_DIR)/%, $(SOURCES))


#include组件
# 需要提前编译的C++源文件（自动扫描include目录下所有.cpp），纯头文件（如 cudacommon.h）无需特殊处理，代码包含就直接使用了
CPP_SRCS := $(wildcard $(INCLUDE_DIR)/*.cpp)  # 例如：OptionParser.cpp
CPP_OBJS := $(patsubst $(INCLUDE_DIR)/%.cpp, $(BIN_DIR)/%.o, $(CPP_SRCS))



# 程序分类
# OpenCL程序列表
OPENCL_PROGRAMS := bfs_ocl  # 例如您的程序名为 bfs_opencl.cpp
# OpenMP程序列表，需要使用openmp但不使用snap库
OPENMP_PROGRAMS := bfs_omp  bfs_omp_o1 bfs_omp_o2 # 需要OpenMP但不依赖SNAP的程序
# 使用SNAP的程序列表（匹配目标名，如bin/testsnap对应testsnap），空格分开
SNAP_PROGRAMS := testsnap graph_gen
# cuGraph 程序列表
CUGraph_CUDA_PROGRAMS := bfs_cugraph bfs_cugraph_v

# ------------------------- CUDA 程序分类 -------------------------
# 所有 CUDA 源文件
ALL_CUDA_SOURCES := $(wildcard $(SRC_DIR)/*.cu)
# 分类cuda源文件
CUGraph_CUDA_SOURCES := $(addprefix $(SRC_DIR)/, $(addsuffix .cu, $(CUGraph_CUDA_PROGRAMS)))
COMMON_CUDA_SOURCES := $(filter-out $(CUGraph_CUDA_SOURCES), $(ALL_CUDA_SOURCES))

# 生成可执行文件列表
CUGraph_CUDA_EXECUTABLES := $(patsubst $(SRC_DIR)/%.cu, $(BIN_DIR)/%, $(CUGraph_CUDA_SOURCES))
COMMON_CUDA_EXECUTABLES := $(patsubst $(SRC_DIR)/%.cu, $(BIN_DIR)/%, $(COMMON_CUDA_SOURCES))


# ------------------------- 编译规则 -------------------------
# 主目标：编译所有程序（包含CUDA）
all: $(EXECUTABLES) $(COMMON_CUDA_EXECUTABLES) $(CUGraph_CUDA_EXECUTABLES)

# #公共include文件
# 编译include目录下的C++实现文件（如OptionParser.cpp）
$(BIN_DIR)/%.o: $(INCLUDE_DIR)/%.cpp | $(BIN_DIR)
	@echo "=== 编译公共组件-C++对象文件：$@ ==="
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# # cuGraph 程序专用规则
# $(CUGraph_CUDA_EXECUTABLES): $(BIN_DIR)/%: $(SRC_DIR)/%.cu $(CPP_OBJS) | $(BIN_DIR)
# 	@echo "=== 编译 cuGraph 程序 [$@] (C++17+实验性Lambda) ==="
# 	$(NVCC) $(CUGRAPH_NVCCFLAGS) $< $(CPP_OBJS) -o $@ $(CUGRAPH_LDFLAGS)

# 修改cuGraph的编译规则，添加环境检测
$(CUGraph_CUDA_EXECUTABLES): $(BIN_DIR)/%: $(SRC_DIR)/%.cu $(CPP_OBJS) | $(BIN_DIR)
	@echo "=== 编译 cuGraph 程序 [$@] (需要Conda环境) ==="
	@# 仅在编译cuGraph时检查Conda环境
	@test -n "$(CONDA_PREFIX)" || { \
    echo "=========================================="; \
    echo " 编译cuGraph程序需要激活Conda环境！"; \
    echo " 请执行：conda activate cugraph_env"; \
    echo "=========================================="; \
    exit 1;}
	$(NVCC) $(CUGRAPH_NVCCFLAGS) $< $(CPP_OBJS) -o $@ $(CUGRAPH_LDFLAGS)


# 普通 CUDA 程序规则
$(COMMON_CUDA_EXECUTABLES): $(BIN_DIR)/%: $(SRC_DIR)/%.cu $(CPP_OBJS) | $(BIN_DIR)
	@echo "=== 编译普通 CUDA 程序 [$@] ==="
	$(NVCC) $(CUDA_NVCCFLAGS) $< $(CPP_OBJS) -o $@ $(CUDA_LDFLAGS)

# OpenCL程序规则，包含include内头文件
$(foreach prog,$(OPENCL_PROGRAMS),$(BIN_DIR)/$(prog)): $(BIN_DIR)/%: $(SRC_DIR)/%.cpp $(CPP_OBJS) | $(BIN_DIR)
	@echo "=== 编译OpenCL程序：$@ ==="
	$(CXX) $(CXXFLAGS) $(OPENCL_CXXFLAGS) -I$(INCLUDE_DIR) $< $(CPP_OBJS) -o $@ $(OPENCL_LDFLAGS) $(LDFLAGS)

# OpenMP程序规则
$(foreach prog,$(OPENMP_PROGRAMS),$(BIN_DIR)/$(prog)): $(BIN_DIR)/%: $(SRC_DIR)/%.cpp | $(BIN_DIR)
	@echo "=== 编译OpenMP程序：$@ ==="
	$(CXX) $(CXXFLAGS) $(OMP_CXXFLAGS) $< -o $@ $(OMP_LDFLAGS) $(LDFLAGS)

# SNAP程序规则
$(foreach prog,$(SNAP_PROGRAMS),$(BIN_DIR)/$(prog)): $(BIN_DIR)/%: $(SRC_DIR)/%.cpp $(SNAP_LIB) | $(BIN_DIR)
	@echo "=== 编译SNAP程序：$@ ==="
	$(CXX) $(CXXFLAGS) $(OMP_CXXFLAGS) $(SNAP_INCLUDES) $< -o $@ $(SNAP_LDFLAGS) $(LDFLAGS)

# 普通程序规则（排除其他类别）
$(filter-out $(foreach p,$(SNAP_PROGRAMS) $(OPENMP_PROGRAMS) $(OPENCL_PROGRAMS),$(BIN_DIR)/$(p)),$(EXECUTABLES)): $(BIN_DIR)/%: $(SRC_DIR)/%.cpp | $(BIN_DIR)
	@echo "=== 编译普通程序：$@ ==="
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# 创建输出目录
$(BIN_DIR):
	mkdir -p $@

# 编译SNAP静态库
$(SNAP_LIB):
	@echo "=== 编译SNAP库 ==="
	$(MAKE) -C $(SNAP_CORE) lib


# 清理所有生成文件
clean:
	rm -rf $(BIN_DIR)
	$(MAKE) -C $(SNAP_CORE) clean

.PHONY: all clean
