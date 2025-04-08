#!/bin/bash
#脚本功能：控制不同节点数和平均度以生成图
#节点数分为小，中，大三种规模，小：[1k,10k)步长1k+[1万，10万)步长1万,中：[10万，100万）步长10万，大[100万，300万]步长50万
#度变化为[10,150]步长10,同时每种平均度下需要生成5副图
#使用方法 ./graphgen.sh path2saveallfiles
#先注释了中规模和大规模，用小规模测试
#nohup ./generate_graphs.sh "/media/ss/ss/my_bfs_bp" > graph_gen.log 2>&1 &
#可通过htop命令监控生成进程的资源使用情况

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

# 验证生成程序
if [[ ! -x "$GENERATOR" ]]; then
    echo "错误：找不到 graph_gen 程序"
    exit 1
fi

# 生成节点序列
generate_nodes() {
    # # 微型 [1k, 5k]
    seq 1000 1000 5000
    # # 小型 [1w, 5w]
    seq 10000 10000 50000
    # # 中型 [10w,10w,100w]
    seq 100000 100000 1000000
    # 大型 [200w, 1000w]
    seq 2000000 1000000 10000000
    # 超大型[2000w, 1亿]
    # seq 90000000 10000000 90000000
}

# 修改后的generate_degrees函数
generate_degrees() {
    local nodes=$1
    local -a degrees
    
    # # 固定系数数组
    # local fixed_factors=(0.5 0.8 0.9 1.0 1.1 1.5 2.0)
    # 对数系数数组
    # local log_factors=(0.5 0.8 0.9 1.0 1.1 1.5 2.0)

    # local log_factors=(0.5 0.7 0.85 0.95 1.0 1.05 1.15 1.3 1.6 2.0)
    # #脚本1
    # local log_factors=(0.5 0.7)
    # #脚本2
    # # local log_factors=(0.85 0.95)
    # #脚本3
    local log_factors=(1.0 1.05)
    # #脚本4
    # local log_factors=(1.15 1.3)
    # #脚本5
    # local log_factors=(1.6 2.0)

    # # 前7个固定值（保留1位小数）
    # for factor in "${fixed_factors[@]}"; do
    #     degrees+=($(printf "%.1f" $factor))
    # done
    
    # 计算对数基数（保留4位精度）
    local logn=$(echo "scale=4; l($nodes)/1" | bc -l)  # 添加/1强制除法运算
    
    # 后7个对数相关值（保留1位小数）
    for factor in "${log_factors[@]}"; do
        local val=$(echo "scale=5; $factor * $logn" | bc)
        degrees+=($(printf "%.2f" $val))  # 保留2位小数
    done
    
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
        # 创建目录名替换小数点,无需替换
        # safe_degree=$(echo "$degree" | tr '.' '_')
        # output_dir="$TARGET_DIR/graph${formatted_nodes}/deg${safe_degree}"
        output_dir="$TARGET_DIR/graph${formatted_nodes}/d${degree}"
        mkdir -p "$output_dir"
        
        # echo "测试双while运行 nodes : $nodes, degree : $degree"
        # 生成10个副本
        #测试2个
        for ((i=1; i<=10; i++)); do
            seed=$((1000 * i))
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "▶ 开始生成：节点数=${nodes} 平均度=${degree}  rndseed=${seed} 副本=${i}"
            echo "start at $(date)"
            # 执行生成命令（带小数）
            "$GENERATOR" -n $nodes -d $degree -s $seed
            
            # 处理浮点文件名
            original_file="graph${formatted_nodes}_d${degree}_gunrock.csr.bin"

            new_file="graph${formatted_nodes}_d${degree}_${i}_gunrock.csr.bin"


            # 验证并移动文件
            #原文件存在才进行移动，移动成功后才进行计数
            if [[ -f "$original_file" ]]; then
                if mv "$original_file" "$output_dir/$new_file"; then
                    ((total_files += 1))  # 仅在移动成功时计数
                    echo "✓ 文件已移动至：$output_dir/$new_file"
                fi
            else
                echo "错误：找不到生成文件 $original_file"
                exit 3
            fi
            echo "end at $(date)"
        done
    done < <(generate_degrees "$nodes")
done < <(generate_nodes)

# 最终统计
cat <<EOF

══════════════════════════════════════════════
 全部完成！成功生成 $total_files 个图文件
 存储路径：$TARGET_DIR
══════════════════════════════════════════════
EOF

