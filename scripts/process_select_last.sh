#!/bin/bash

#脚本功能，遍历待处理文件夹，然后使用py脚本处理e_select.txt文件,并将输出结果移动到相应文件夹

#用法：./time_stats.sh filepath

# 检查输入参数
if [ $# -ne 1 ]; then
    echo "Usage: $0 <filepath>"
    exit 1
fi

# ===================================================================

filepath="$1"
script_dir=$(cd "$(dirname "$0")" && pwd)
#存放去除异常值结果的文件夹路径
result_dir=${script_dir}/../result_handle/result_last

# 遍历文件结构
for graph_dir in "${filepath}"/graph*; do
    graph_name=$(basename "$graph_dir")
    
    for degree_dir in "${graph_dir}"/d*; do
        degree_name=$(basename "$degree_dir")
        
        # 根据遍历文件夹名在存放结果处创建对应文件夹
        result_last="${result_dir}/${graph_name}/${degree_name}"
        mkdir -p "$result_last"
        
        filename="e_select.txt"
        
        #构造文件完整路径
        fullpath=${filepath}/${graph_name}/${degree_name}/${filename}
        # echo "full path is $fullpath"
        file_to_mv=${filepath}/${graph_name}/${degree_name}/last_${filename}
        # echo "file_to_mv is $file_to_mv"

        #执行python时间提取脚本,该脚本会在当前路径下创建e_filename.txt文件
        # 验证并执行脚本
        if [[ -f "$fullpath" ]]; then
            python3 select_time_handle.py $fullpath
        else
            echo "⚠ 文件不存在：${filename}，跳过..."
        fi
        
        # 验证并移动文件
        if [[ -f "$file_to_mv" ]]; then
            #移动e_filename.txt到对应文件夹
            mv $file_to_mv $result_last
            echo "mv s_${filename}"
        else
            echo "⚠ 文件不存在：rmo_${filename}，跳过..."
        fi
        
    done
done

echo "All tasks completed. Results in: ${result_dir}"