
__kernel void kernel1(__global const uint32_t* g_offsets,
		__global uint32_t* g_edges,
		__global bool* g_graph_mask,
		__global bool* g_updating_graph_mask,
		__global bool* g_graph_visited,
		__global int* g_cost,
		int no_of_nodes) 
{
	unsigned int tid = get_global_id(0);

	if(tid < no_of_nodes && g_graph_mask[tid])
	{
		g_graph_mask[tid] = false;//移除当前层
		//获取当前节点在edges数组内的邻居偏移量
		//邻居在edges内的偏移量是[start, end)
		uint32_t start = g_offsets[tid];
		uint32_t end = g_offsets[tid + 1];
		for(int i = start; i < end; i++)
		{
			uint32_t neighbor = g_edges[i];
			if(!g_graph_visited[neighbor]){
				g_cost[neighbor]=g_cost[tid]+1;
				g_updating_graph_mask[neighbor]=true;
			}
		}
	}
}

__kernel void kernel2(__global bool* g_graph_mask,
		__global bool* g_updating_graph_mask,
		__global bool* g_graph_visited,
		__global bool* g_over,
		int no_of_nodes)
{
	unsigned int tid = get_global_id(0);
	if(tid < no_of_nodes && g_updating_graph_mask[tid])
	{
		g_graph_mask[tid] = true;
		g_graph_visited[tid] = true;
		*g_over = true;
		g_updating_graph_mask[tid] = false;
	}	
}
