#include "Snap.h"
#include <iostream>

int main() {
    // 创建一个随机图
    PUNGraph Graph = TSnap::GenRndGnm<PUNGraph>(1000, 5000);
    std::cout << "节点数: " << Graph->GetNodes() << std::endl;
    std::cout << "边数: " << Graph->GetEdges() << std::endl;
    return 0;
}