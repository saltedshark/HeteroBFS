#!/bin/bash
#脚本功能
#   移除结果文件夹内的所有uvm相关.txt文件，因为结果不正确
#用法：./rm_uvm_result.sh filepath


# 检查输入参数
if [ $# -ne 1 ]; then
    echo "Usage: $0 <filepath>"
    exit 1
fi

# ===================================================================

filepath="$1"
script_dir=$(cd "$(dirname "$0")" && pwd)


# 遍历文件结构
for graph_dir in "${filepath}"/graph*; do
    graph_name=$(basename "$graph_dir")
    
    for degree_dir in "${graph_dir}"/d*; do
        degree_name=$(basename "$degree_dir")
        
        for data_file in "${degree_dir}"/cuda-uvm*.txt; do
            filename=$(basename "$data_file")
            
            # echo "$filename"
            full_path="${filepath}/${graph_name}/${degree_name}/${filename}"
            # echo "full path is $full_path"
            
            # 验证并删除文件
            if [[ -f "$full_path" ]]; then
                rm -rf "$full_path"
                echo "已删除$full_path"
            else
                echo "⚠ 文件不存在：${filename}，跳过..."
            fi
        done
    done
done

echo "All cuda-uvm*.txt were deleted."