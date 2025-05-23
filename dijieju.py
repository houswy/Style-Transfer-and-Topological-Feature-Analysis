import os
import numpy as np
from skimage import io, color
from skimage.util import img_as_ubyte

# 文件夹路径
folder_path = r'G:\GAN\data and pth\captcha_3000_after_conversion\CycleGAN\double\vangogh\vangogh_vangogh'

# 设置要计算的图像数量
num_images = 3000  # 你可以修改这个数字，控制计算多少张图像

# 存储所有图像的均值、方差和标准差
mean_values = []
variance_values = []
std_dev_values = []

# 遍历文件夹中的图像文件
count = 0
for filename in os.listdir(folder_path):
    if count >= num_images:
        break  # 达到指定数量后停止

    file_path = os.path.join(folder_path, filename)

    # 确保只处理图像文件
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # 读取图像
        image = io.imread(file_path)

        # 转换为灰度图像
        gray_image = color.rgb2gray(image)
        gray_image = img_as_ubyte(gray_image)  # 转换为8位无符号整数

        # 计算当前图像的均值
        mean_intensity = np.mean(gray_image)
        mean_values.append(mean_intensity)

        # 计算方差
        variance_intensity = np.var(gray_image)
        variance_values.append(variance_intensity)

        # 计算标准差
        std_dev_intensity = np.std(gray_image)
        std_dev_values.append(std_dev_intensity)

        # 增加计数
        count += 1

# 计算所有图像的均值、方差和标准差
if mean_values:
    overall_mean = np.mean(mean_values)
    overall_variance = np.mean(variance_values)
    overall_std_dev = np.mean(std_dev_values)

    print(f"选定的{num_images}张图像的整体均值：{overall_mean}")
    print(f"选定的{num_images}张图像的整体方差：{overall_variance}")
    print(f"选定的{num_images}张图像的整体标准差：{overall_std_dev}")
else:
    print("没有足够的图像进行计算。")
