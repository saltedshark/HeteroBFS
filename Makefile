# 编译器配置
CXX := g++
CXXFLAGS := -std=c++11 -Wall
LDFLAGS := -lpthread

# OpenMP编译选项
OMP_CXXFLAGS := -fopenmp
OMP_LDFLAGS := -lgomp

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
SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
EXECUTABLES := $(patsubst $(SRC_DIR)/%.cpp, $(BIN_DIR)/%, $(SOURCES))


# OpenMP程序列表，需要使用openmp但不使用snap库
OPENMP_PROGRAMS := bfs_omp   # 需要OpenMP但不依赖SNAP的程序
# 需要链接SNAP的程序列表（匹配目标名，如bin/testsnap对应testsnap）
SNAP_PROGRAMS := testsnap graph_gen

# 主目标：编译所有程序
all: $(EXECUTABLES)

# OpenMP程序规则
$(foreach prog,$(OPENMP_PROGRAMS),$(BIN_DIR)/$(prog)): $(BIN_DIR)/%: $(SRC_DIR)/%.cpp | $(BIN_DIR)
	@echo "=== 编译OpenMP程序：$@ ==="
	$(CXX) $(CXXFLAGS) $(OMP_CXXFLAGS) $< -o $@ $(OMP_LDFLAGS) $(LDFLAGS)

# SNAP程序规则
$(foreach prog,$(SNAP_PROGRAMS),$(BIN_DIR)/$(prog)): $(BIN_DIR)/%: $(SRC_DIR)/%.cpp $(SNAP_LIB) | $(BIN_DIR)
	@echo "=== 编译SNAP程序：$@ ==="
	$(CXX) $(CXXFLAGS) $(OMP_CXXFLAGS) $(SNAP_INCLUDES) $< -o $@ $(SNAP_LDFLAGS) $(LDFLAGS)

# 普通程序规则（排除前两类）
$(filter-out $(foreach p,$(SNAP_PROGRAMS) $(OPENMP_PROGRAMS),$(BIN_DIR)/$(p)),$(EXECUTABLES)): $(BIN_DIR)/%: $(SRC_DIR)/%.cpp | $(BIN_DIR)
	@echo "=== 编译普通程序：$@ ==="
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)


# 创建输出目录
$(BIN_DIR):
	mkdir -p $@

# 编译SNAP静态库
$(SNAP_LIB):
	@echo "=== 编译SNAP库 ==="
	$(MAKE) -C $(SNAP_CORE) lib
# 编译SNAP静态库（确保SNAP自身启用OpenMP）
# $(SNAP_LIB):
# 	@echo "=== 编译SNAP库（启用OpenMP） ==="
# 	$(MAKE) -C $(SNAP_CORE) CXXFLAGS="-std=c++11 -Wall -fPIC -fopenmp" lib

# 清理所有生成文件
clean:
	rm -rf $(BIN_DIR)
	$(MAKE) -C $(SNAP_CORE) clean

.PHONY: all clean