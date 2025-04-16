import networkx as nx
import matplotlib.pyplot as plt

# 创建一个小世界网络（Watts-Strogatz模型）
n = 9  # 节点数
k = 4  # 每个节点初始的连接数
p = 0.3  # 重连的概率（表示小世界效应）

G = nx.watts_strogatz_graph(n, k, p)

# 计算图的直径
if nx.is_connected(G):
    # 获取所有节点对的最短路径
    all_pairs_shortest_path = dict(nx.all_pairs_shortest_path_length(G))

    # 计算直径
    diameter = 0
    print("计算所有节点对之间的最短路径长度：")
    # 遍历所有最短路径，找出最大值
    for source in all_pairs_shortest_path:
        for target in all_pairs_shortest_path[source]:
            path_length = all_pairs_shortest_path[source][target]
            print(f"节点 {source} 和 节点 {target} 之间的最短路径长度: {path_length}")
            if path_length > diameter:
                diameter = path_length

    print(f"\n图的直径为: {diameter}")
else:
    print("图不连通，无法计算直径。")

# 绘制小世界网络拓扑图，黑白配色，节点为小黑点
plt.figure(figsize=(6, 6), dpi=300)  # 设置高分辨率
pos = nx.spring_layout(G, seed=42)  # 使用Spring布局来调整节点的位置，使其更清晰
nx.draw(G, pos, with_labels=True, node_size=50, node_color='black', font_size=10, font_weight='bold',
        edge_color='black')

# 设置图像样式
plt.title("Small-World Network Topology with 9 Nodes", fontsize=14, fontweight='bold')
plt.axis('off')  # 去掉坐标轴
plt.tight_layout()  # 自动调整布局以避免标签重叠

# 保存为高分辨率的PNG文件
plt.savefig('small_world_topology_9_nodes_bw.png', format='PNG', dpi=300)

# 显示图形
plt.show()
