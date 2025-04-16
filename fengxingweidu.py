import cv2
import os
import numpy as np
from sklearn.linear_model import LinearRegression

# 设置图像文件夹路径和输入图像数量
folder_path = 'G:\GAN\data and pth\captcha_3000'  # 修改为您的文件夹路径
max_images = 1  # 设置最大输入图像数量

# 获取文件夹中所有图像文件
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 只选择前 `max_images` 张图像
image_files = image_files[:max_images]

# 计算分形维度：盒计数法
def box_counting(binary_image, box_sizes):
    counts = []
    for box_size in box_sizes:
        count = 0
        # 对图像进行分块
        for i in range(0, binary_image.shape[0], box_size):
            for j in range(0, binary_image.shape[1], box_size):
                # 如果盒子内有任何白色像素
                if np.any(binary_image[i:i + box_size, j:j + box_size] == 255):
                    count += 1
        counts.append(count)
    return counts

# 选择不同的盒子大小（增加盒子大小的范围）
box_sizes = [2 ** i for i in range(1, 7)]  # 扩展盒子大小范围到 2, 4, 8, 16, 32, 64

# 存储所有图像的分形维度
fractal_dimensions = []

# 遍历文件夹中的所有图像
for image_file in image_files:
    # 拼接文件路径
    image_path = os.path.join(folder_path, image_file)

    # 加载图像，并检查是否成功
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        continue

    # 图像预处理：使用自适应阈值
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # 计算每个盒子大小下的覆盖数量
    counts = box_counting(binary_image, box_sizes)

    # 使用线性回归计算分形维度
    log_counts = np.log(np.array(counts) + 1e-10)  # 添加一个小的常数来避免log(0)
    log_box_sizes = np.log(box_sizes)

    # 进行线性回归，拟合 log(盒子数目) 与 log(盒子大小) 的关系
    regressor = LinearRegression()
    regressor.fit(log_box_sizes.reshape(-1, 1), log_counts)

    # 分形维度为回归线的斜率
    fractal_dimension = -regressor.coef_[0]

    # 存储分形维度
    fractal_dimensions.append(fractal_dimension)

# 计算所有图像的平均分形维度
average_fractal_dimension = np.mean(fractal_dimensions)
print(f"Average Fractal Dimension: {average_fractal_dimension}")
