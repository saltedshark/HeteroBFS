#!/bin/bash
#脚本功能：用于删除低平均度的多余文件夹，需要删除d0.8,d0.9,d1.1,d1.5这4个点
set -euo pipefail

# 检查参数
if [ $# -ne 1 ]; then
    echo "用法: $0 <目标目录>"
    exit 1
fi

# 路径配置
TARGET_DIR=$(realpath "$1")
BIN_DIR="../bin"
GENERATOR="$BIN_DIR/graph_gen"


# 生成节点序列
generate_nodes() {
    # 微型 [1k, 5k]
    seq 1000 1000 5000
    # 小型 [1w, 5w]
    seq 10000 10000 50000
    # 中型 [10w,10w,100w]
    seq 100000 100000 1000000
    # 大型 [200w, 2000w]
    seq 2000000 1000000 18000000
    #实际图只到1700w
}


# 修改后的generate_degrees函数
generate_degrees() {
    local nodes=$1
    local -a degrees
    
    # # 固定系数数组
    # local fixed_factors=(0.5 0.8 0.9 1.0 1.1 1.5 2.0)
    # 固定系数数组，待删除
    # local fixed_factors=(0.8 0.9 1.1 1.5)
    # d0.5也删除
    local fixed_factors=(0.5)
    # # 对数系数数组
    # local log_factors=(0.5 0.8 0.9 1.0 1.1 1.5 2.0)
    
    # 前7个固定值（保留1位小数）
    for factor in "${fixed_factors[@]}"; do
        degrees+=($(printf "%.1f" $factor))
    done
    
    # # 计算对数基数（保留4位精度）
    # local logn=$(echo "scale=4; l($nodes)/1" | bc -l)  # 添加/1强制除法运算
    
    # # 后7个对数相关值（保留1位小数）
    # for factor in "${log_factors[@]}"; do
    #     local val=$(echo "scale=5; $factor * $logn" | bc)
    #     degrees+=($(printf "%.1f" $val))  # 保留1位小数
    # done
    
    # 去重并排序输出
    printf "%s\n" "${degrees[@]}" | sort -nu
}

# 节点数格式化函数
format_nodes() {
    local nodes=$1
    case $nodes in
        *000000) echo "$(($nodes/1000000))M" ;;
        *000)    echo "$(($nodes/1000))K" ;;
        *)       echo $nodes ;;
    esac
}

# 修改后的主循环片段
total_files=0

while IFS= read -r nodes; do
    formatted_nodes=$(format_nodes $nodes)
    
    # 生成当前节点数对应的度数序列
    while IFS= read -r degree; do
        
        output_dir="$TARGET_DIR/graph${formatted_nodes}/d${degree}"
        
        # #这里要删除编号为6——10的文件
        # for ((i=6; i<=10; i++)); do
            
        #     #构造待删除文件名
        #     file_to_rm="graph${formatted_nodes}_d${degree}_${i}_gunrock.csr.bin"
        #     full_path="${output_dir}/${file_to_rm}"
        #     # 验证并删除文件
        #     if [[ -d "$output_dir" ]]; then
        #         if [[ -f "$full_path" ]]; then
        #             rm -v "$full_path"
        #             echo "▶ 已删除：节点数=${nodes} 平均度=${degree} 副本=${i}"
        #         fi
        #     else
        #         echo "⚠ 目录不存在：${output_dir}，跳过..."
        #     fi
        # done
        #构造待删除文件夹名，就是outputdir，这里生成的度都是要删除的
        dir_to_rm="$output_dir"
        
        # 验证并删除文件
        if [[ -d "$output_dir" ]]; then
            rm -rf "$dir_to_rm"
            echo "已删除$dir_to_rm"
        else
            echo "⚠ 目录不存在：${output_dir}，跳过..."
        fi



    done < <(generate_degrees "$nodes")
done < <(generate_nodes)

echo "delete done!"

