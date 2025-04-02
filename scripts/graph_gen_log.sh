#!/bin/bash

# 带时间戳的日志文件名
LOG_FILE="$(date +%Y%m%d_%H%M%S).log"
touch $LOG_FILE

# 启动日志记录
{
    echo "====== 脚本开始执行 $(date) ======"
    # 主要逻辑
    # ./generate_graphs.sh /media/ss/ss/my_bfs_bp
    echo "====== 脚本执行结束 $(date) ======"
} 2>&1 | tee -a "$LOG_FILE"