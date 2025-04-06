#!/bin/bash

# 生成临时剩余3个图
echo $(date)
../bin/graph_gen -n 90000000 -d 21.06 -s 3000
echo $(date)
mv graph90M_d21.06_gunrock.csr.bin graph90M_d21.06_3_gunrock.csr.bin
mv graph90M_d21.06_3_gunrock.csr.bin /media/ss/ss/my_bfs_bp/graph90M/d21.06/

echo $(date)
../bin/graph_gen -n 90000000 -d 21.06 -s 4000
mv graph90M_d21.06_gunrock.csr.bin graph90M_d21.06_4_gunrock.csr.bin
mv graph90M_d21.06_4_gunrock.csr.bin /media/ss/ss/my_bfs_bp/graph90M/d21.06/
echo $(date)
../bin/graph_gen -n 90000000 -d 21.06 -s 5000
mv graph90M_d21.06_gunrock.csr.bin graph90M_d21.06_5_gunrock.csr.bin
mv graph90M_d21.06_5_gunrock.csr.bin /media/ss/ss/my_bfs_bp/graph90M/d21.06/
echo $(date)

# 启动剩余执行脚本，预计执行5h不到
./generate_graphs_9kw_1.sh /media/ss/ss/my_bfs_bp/