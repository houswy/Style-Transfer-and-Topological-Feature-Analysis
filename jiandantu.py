import networkx as nx

# 创建图并添加节点和加权边
def create_weighted_graph():
    # 定义图的节点和边（包含了9个节点的边及其权重）
    edges = [
        ('a', 'b', 10),
        ('a', 'f', 11),
        ('b', 'c', 18),
        ('b', 'g', 12),
        ('b', 'h', 12),

        ('c', 'd', 22),
        ('c', 'h', 8),
        ('i', 'd', 16),
        ('d', 'g', 24),
        ('d', 'h', 21),
        ('d', 'e', 20),

        ('e', 'f', 26),

        ('f', 'g', 17),

        ('i', 'g', 19),
        ('i', 'e', 7)
    ]

    # 创建一个加权图
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    return G, edges

# 计算网络直径
def calculate_network_diameter(G, edges):
    # 计算图中所有节点对之间的最短路径
    print("Calculating shortest paths between all node pairs...")
    all_pairs_shortest_path = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

    # 打印所有节点对之间的最短路径
    print("Shortest paths between all node pairs (with distances):")
    for node1, paths in all_pairs_shortest_path.items():
        for node2, distance in paths.items():
            print(f"From {node1} to {node2}: {distance}")

    # 创建一个字典，将边的权重存储为键值对
    edge_weights = {(u, v): w for u, v, w in edges}

    # 计算所有节点对的最短路径之和
    total_distance = 0
    pair_count = 0
    nodes = list(G.nodes())

    # 计算每一对节点的最短路径并打印细节
    print("\nCalculating total distances and printing individual calculations:")
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node1 = nodes[i]
            node2 = nodes[j]
            if node2 in all_pairs_shortest_path[node1]:
                distance = all_pairs_shortest_path[node1][node2]
                total_distance += distance
                pair_count += 1
                print(f"\nFrom {node1} to {node2}: {distance}")
                # 打印路径上每条边的计算过程
                path = nx.dijkstra_path(G, node1, node2, weight='weight')  # 获取最短路径
                print(f"Path from {node1} to {node2}: {path}")
                path_weight = 0
                path_str = ""
                for k in range(len(path) - 1):
                    edge = (path[k], path[k + 1])  # 获取边
                    if edge not in edge_weights:  # 检查是否是反向边
                        edge = (path[k + 1], path[k])
                    edge_weight = edge_weights[edge]  # 获取边的权重
                    path_weight += edge_weight
                    path_str += f"{edge_weight} + "
                # 去掉最后一个多余的 "+"
                path_str = path_str.rstrip(" + ")
                print(f"  Edge weights calculation: {path_str} = {path_weight}")

    # 输出计算过程中的中间结果
    print("\nTotal distances of all node pairs:", total_distance)
    print("Number of node pairs:", pair_count)

    # 根据公式计算网络直径
    if pair_count > 0:
        network_diameter = max(max(path_length.values()) for path_length in all_pairs_shortest_path.values())
        return network_diameter
    else:
        return None


# 创建加权图
G, edges = create_weighted_graph()

# 计算网络直径
diameter = calculate_network_diameter(G, edges)
print(f"\nNetwork Diameter: {diameter}")
