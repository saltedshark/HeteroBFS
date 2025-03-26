# HeteroBFS
本项目用于记录基于图参数的BFS硬件调度感知项目，代码参考rodinia、altis和opendwarfs2025等项目，计划实现广度优先遍历的顺序队列、openmp、opencl以及cuda版本代码，针对不同类型图运行一些数据，然后使用机器学习方法对后续图执行进行最优方法选择，输入执行图的特征，然后自动选择最优方法进行执行。

## 程序文件结构
程序分为7个部分：
1. data，存放输入数据;
2. bin，存放编译后的执行程序;
3. exe_result, 存放执行结果;
4. analyze, 存放结果处理相关文件;
5. src, 存放源代码，包括图生成，图属性获取，bfs各实现等;
6. extern,存放外部内容，比如斯坦福的snap库;
7. scripts, 存放执行脚本

## 编译说明
1. 编译某个程序时，执行make bin/program即可，makefile内有具体程序分类，分为使用snap库的程序，opencl程序，使用openmp的程序，使用cugraph的程序，需在对应类别上添加程序名才能进行编译指定编译
2. cugraph相关安装使用conda进行管理，执行命令是conda create -n cugraph_env -c rapidsai -c conda-forge -c nvidia rapids=25.02 python=3.12 cuda-version=11.8,可参考[https://rapids.ai/#quick-start]链接安装

## 免责声明
本项目引用的第三方库/代码的版权归属原作者所有。
若您发现任何许可证违规问题，请通过 issue 或邮件联系我们，我们将立即处理。
