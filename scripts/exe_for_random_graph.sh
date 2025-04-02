#!/bin/bash
#脚本功能
#   根据输入的路径filepath，对路径下的图进行遍历执行，结果存放到../exe_result下，也按图所在文件夹的组织结构进行组织
#   filepath下所存图的组织结构:filepath/graph1K/degree10/graph1K_d10_1_gunrock.csr.bin
#   程序bfs.seq对graph1K/degree10/内的5个文件的结果均存放到../exe_result/graph1K/degree10/bfs_seq.txt内
#用法：./exe_for_random_graph.sh filepath


# 检查输入参数
if [ $# -ne 1 ]; then
    echo "Usage: $0 <filepath>"
    exit 1
fi

# 用户自定义区（只需修改这里）=========================================
# 13种执行配置
#   seq 1
#   omp 1
#   cuda 4
#   cuda_edge 4
#   ocl 3
#   cuda_edge_opt 1

# input_file统一在遍历文件里面添加,所以有-i参数单独也需要放在最后

# 程序1配置
# Usage: ./bfs_seq <passes> <input_file>
PROG1_NAME="bfs_seq"      # 程序1名称
PROG1_ARGS="10"             # 程序1专用参数
PROG1_OUT="seq"         # 程序1输出文件名

# 程序2配置 
# Usage: ./bfs_omp <passes> <num_omp_threads> <input_file> 
# 使用16线程
PROG2_NAME="bfs_omp"         # 程序2名称
PROG2_ARGS="10 16"        # 程序2专用参数
PROG2_OUT="omp"

#bfs_cuda
#./bfs_cuda -n 10 -i filepath
# 程序3配置
PROG3_NAME="bfs_cuda"
PROG3_ARGS="-n 10 -i"
PROG3_OUT="cuda"

# 程序4配置
PROG4_NAME="bfs_cuda"
PROG4_ARGS="-n 10 --uvm -i"
PROG4_OUT="cuda-uvm"

# 程序5配置
PROG5_NAME="bfs_cuda"
PROG5_ARGS="-n 10 --uvm-advise -i"
PROG5_OUT="cuda-uvm-advise"

# 程序6配置
PROG6_NAME="bfs_cuda"
PROG6_ARGS="-n 10 --uvm-prefetch-advise -i"
PROG6_OUT="cuda-uvm-prefetch-advise"

#bfs_cuda_edge
#./bfs_cuda_edge -n 10 -i filepath
# 程序7配置
PROG7_NAME="bfs_cuda_edge"
PROG7_ARGS="-n 10 -i"
PROG7_OUT="cuda-edge"

# 程序8配置
PROG8_NAME="bfs_cuda_edge"
PROG8_ARGS="-n 10 --uvm -i"
PROG8_OUT="cuda-edge-uvm"

# # 程序9配置
# PROG9_NAME="bfs_cuda_edge"
# PROG9_ARGS="-n 10 --uvm-advise -i"
# PROG9_OUT="cuda-edge-uvm-advise"

# # 程序10配置
# PROG10_NAME="bfs_cuda_edge"
# PROG10_ARGS="-n 10 --uvm-prefetch-advise -i"
# PROG10_OUT="cuda-edge-uvm-prefetch-advise"


#bfs_ocl
#./bfs_ocl -p 1/2/3 -d 0 -n 10 -i filepath
# 这里的123需根据clinfo -l进行调整
# 程序11配置
PROG11_NAME="bfs_ocl"
PROG11_ARGS="-p 1 -d 0 -n 10 -i"
PROG11_OUT="ocl-cpu"

# 程序12配置
PROG12_NAME="bfs_ocl"
PROG12_ARGS="-p 2 -d 0 -n 10 -i"
PROG12_OUT="ocl-igpu"

# 程序13配置
PROG13_NAME="bfs_ocl"
PROG13_ARGS="-p 3 -d 0 -n 10 -i"
PROG13_OUT="ocl-ngpu"

# 程序14配置
# bfs_cuda_edge优化版本，设备排序
PROG14_NAME="bfs_cuda_edge_opt"
PROG14_ARGS="-n 10 -i"
PROG14_OUT="cuda-edge-opt"
# 程序15配置
# bfs_cuda_edge优化版本，设备排序+thrust设备查找
PROG15_NAME="bfs_cuda_edge_opt_thrust"
PROG15_ARGS="-n 10 -i"
PROG15_OUT="cuda-edge-opt-thrust"
# 程序16配置
# bfs_cuda优化版本，thrust设备查找
PROG16_NAME="bfs_cuda_thrust"
PROG16_ARGS="-n 10 -i"
PROG16_OUT="cuda-thrust"

# # 程序17配置
# # omp优化版本1,分离
# PROG17_NAME="bfs_omp_o1"
# PROG17_ARGS="10 16"
# PROG17_OUT="omp-o1"
# # 程序18配置
# #omp 优化版本2
# PROG18_NAME="bfs_omp_o2"
# PROG18_ARGS="10 16"
# PROG18_OUT="omp-o2"


# ===================================================================

filepath="$1"
script_dir=$(cd "$(dirname "$0")" && pwd)
bin_dir="${script_dir}/../bin"
result_dir="${script_dir}/../exe_result"

# 遍历文件结构
for graph_dir in "${filepath}"/graph*; do
    graph_name=$(basename "$graph_dir")
    
    for degree_dir in "${graph_dir}"/d*; do
        degree_name=$(basename "$degree_dir")
        
        # 根据遍历文件夹名在存放结果处创建对应文件夹
        exe_result="${result_dir}/${graph_name}/${degree_name}"
        mkdir -p "$exe_result"
        
        # 遍历数据文件,10个副本
        for data_file in "${degree_dir}"/*_gunrock.csr.bin; do
            filename=$(basename "$data_file")
            
            # 遍历所有程序配置
            for prog_num in {1..16}; do
                # 动态获取变量值
                prog_name_var="PROG${prog_num}_NAME"
                prog_args_var="PROG${prog_num}_ARGS"
                prog_out_var="PROG${prog_num}_OUT"
                program_name=${!prog_name_var}
                program_args=${!prog_args_var}
                program_out=${!prog_out_var}
                
                # 跳过未定义的配置
                [ -z "$program_name" ] && continue
                
                #构建输出文件,每副随机图的执行结果均统一放到对应程序结果内
                output_file="${exe_result}/${program_out}.txt"
                
                #程序执行,每个程序执行多少遍已通过程序参数进行控制
                echo "-----------------------------------"
                echo "[RUN] ${program_name} $program_args at ${filename}"
                echo "start at $(date)"
                
                #执行结果直接放到文件内，不在屏幕显示
                "${bin_dir}/${program_name}" $program_args "$data_file" >> "$output_file" 2>&1
                # "${bin_dir}/${program_name}" $program_args "$data_file" 2>&1 | tee -a "$output_file"

                echo "-----------------------------------"
                echo "end at $(date)"
                
            done
        done
    done
done

echo "All tasks completed. Results in: ${result_dir}"