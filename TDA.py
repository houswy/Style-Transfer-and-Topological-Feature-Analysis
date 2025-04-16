import cv2
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

# 图像路径
image_path = 'G:/GAN/data and pth/captcha_3000_after_conversion/TF/style_cezanne/0asG.jpg'

# 加载图像，并检查是否成功
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print(f"Error: Unable to load image from {image_path}")
    exit()

# 图像预处理：二值化处理
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# 显示处理后的图像
plt.imshow(binary_image, cmap='gray')
plt.title("Preprocessed Image")
plt.axis('off')  # 关闭坐标轴
plt.show()

# 使用 OpenCV 进行边缘检测
edges = cv2.Canny(binary_image, 100, 200)

# 显示边缘检测结果
plt.imshow(edges, cmap='gray')
plt.title("Edge Detection")
plt.axis('off')  # 关闭坐标轴
plt.show()

# 使用 NumPy 构建图，避免循环中的图节点删除
graph = nx.grid_2d_graph(binary_image.shape[0], binary_image.shape[1])

# 只保留白色像素的节点
white_pixels = np.where(binary_image == 255)
nodes_to_keep = list(zip(white_pixels[0], white_pixels[1]))
graph.remove_nodes_from(set(graph.nodes) - set(nodes_to_keep))

# 计算连通组件
connected_components = list(nx.connected_components(graph))
largest_component = max(connected_components, key=len)

# 计算最大连通组件的大小
largest_component_size = len(largest_component)
print(f"Largest Connected Component Size: {largest_component_size}")

# 获取最大连通组件的子图
subgraph = graph.subgraph(largest_component)

# 方法 2: 使用两次 BFS 计算网络直径
def bfs_max_distance(start_node, subgraph):
    # 从 start_node 开始的广度优先搜索，返回最远距离
    distances = nx.single_source_shortest_path_length(subgraph, start_node)
    max_distance_node = max(distances, key=distances.get)
    max_distance = distances[max_distance_node]
    return max_distance_node, max_distance

# 1. 从随机节点开始进行 BFS
random_node = random.choice(list(largest_component))
farthest_node, _ = bfs_max_distance(random_node, subgraph)

# 2. 从最远节点开始再进行 BFS
_, network_diameter = bfs_max_distance(farthest_node, subgraph)

print(f"Estimated Network Diameter (using two BFS): {network_diameter}")
