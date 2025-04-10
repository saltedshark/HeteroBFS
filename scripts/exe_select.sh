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

num=10 #控制程序执行次数的参数

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
PROG1_NAME="select_seq_omp_cuda"      # 程序1名称
PROG1_ARGS="-n ${num} -i"             # 程序1专用参数
PROG1_OUT="select"         # 程序1输出文件名



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
            for prog_num in {1..2}; do
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