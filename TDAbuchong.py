import cv2
import numpy as np
import networkx as nx
from skimage import measure
from skimage import io
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('G:/GAN/data and pth/captcha_3000/0glB.jpg', cv2.IMREAD_GRAYSCALE)

# 图像预处理：二值化处理
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# 使用OpenCV进行边缘检测
edges = cv2.Canny(binary_image, 100, 200)

# 将二值图像转为图，节点是白色区域的像素，边是相邻的像素
graph = nx.grid_2d_graph(binary_image.shape[0], binary_image.shape[1])

# 创建图像的网络图
for i in range(binary_image.shape[0]):
    for j in range(binary_image.shape[1]):
        if binary_image[i, j] == 0:  # 黑色像素不参与
            graph.remove_node((i, j))








# 能量计算
energy = np.sum(binary_image ** 2)
print(f"Energy: {energy}")

# 显示原始图像和二值化图像
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 原始图像
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

# 二值化图像
axes[1].imshow(binary_image, cmap='gray')
axes[1].set_title("Binarized Image")
axes[1].axis('off')

plt.show()

# 绘制图的拓扑图
plt.figure(figsize=(8, 8))
pos = {node: (node[1], -node[0]) for node in graph.nodes()}
nx.draw(graph, pos, node_size=10, with_labels=False, node_color='black')


