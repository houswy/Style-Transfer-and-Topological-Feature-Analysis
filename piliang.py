import cv2
import numpy as np
import networkx as nx
import os
import random
from heapq import heappop, heappush

# 计算图像的网络直径
def calculate_diameter(image_path):
    # 加载图像并检查是否成功
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    # 图像预处理：Canny边缘检测
    edges = cv2.Canny(image, 100, 200)

    # 使用 NumPy 构建图，避免循环中的图节点删除
    graph = nx.grid_2d_graph(edges.shape[0], edges.shape[1])

    # 只保留边缘像素的节点
    edge_pixels = np.where(edges == 255)
    nodes_to_keep = list(zip(edge_pixels[0], edge_pixels[1]))
    graph.remove_nodes_from(set(graph.nodes) - set(nodes_to_keep))

    # 计算连通组件
    connected_components = list(nx.connected_components(graph))
    largest_component = max(connected_components, key=len)

    # 获取最大连通组件的子图
    subgraph = graph.subgraph(largest_component)

    # 使用 Dijkstra 算法来计算最长路径
    def dijkstra_max_distance(start_node, subgraph):
        distances = {node: float('inf') for node in subgraph.nodes}
        distances[start_node] = 0
        priority_queue = [(0, start_node)]  # (distance, node)
        farthest_node = start_node
        max_distance = 0

        while priority_queue:
            current_distance, node = heappop(priority_queue)
            if current_distance > distances[node]:
                continue

            for neighbor in subgraph.neighbors(node):
                weight = 1  # 所有边的权重设置为1，可以根据需要调整
                new_distance = current_distance + weight

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    heappush(priority_queue, (new_distance, neighbor))
                    if new_distance > max_distance:
                        max_distance = new_distance
                        farthest_node = neighbor

        return farthest_node, max_distance

    # 1. 从随机节点开始进行 Dijkstra
    random_node = random.choice(list(largest_component))
    farthest_node, _ = dijkstra_max_distance(random_node, subgraph)

    # 2. 从最远节点开始再进行 Dijkstra
    _, network_diameter = dijkstra_max_distance(farthest_node, subgraph)

    return network_diameter


# 批量处理文件夹中的指定数量的图像，计算平均网络直径
def calculate_average_diameter(folder_path, num_images_to_process):
    # 获取文件夹中所有的图像文件
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # 限制处理的图像数量
    image_files = image_files[:num_images_to_process]

    total_diameter = 0
    valid_image_count = 0

    # 对每个图像文件进行处理
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        # 计算图像的网络直径
        diameter = calculate_diameter(image_path)
        if diameter is not None:
            total_diameter += diameter
            valid_image_count += 1

    # 计算平均网络直径
    if valid_image_count > 0:
        average_diameter = total_diameter / valid_image_count
        print(f"Average Network Diameter (using {valid_image_count} images): {average_diameter}")
    else:
        print("No valid images processed.")


# 设置文件夹路径和处理的图像数量
folder_path = 'G:/GAN/data and pth/captcha_3000_after_conversion/TF/style_vangogh'  # 修改为你的图像文件夹路径
num_images_to_process = 200  # 设置要处理的图像数量

# 执行批量计算平均网络直径
calculate_average_diameter(folder_path, num_images_to_process)
