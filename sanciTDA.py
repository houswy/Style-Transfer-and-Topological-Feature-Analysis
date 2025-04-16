import cv2
import numpy as np
import os
import networkx as nx
from concurrent.futures import ProcessPoolExecutor

# 加载图像
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法加载图像：{image_path}")
    return image

# 二值化处理
def preprocess_image(image, threshold_value=128):
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image

# 计算拓扑特征：图的直径
def compute_graph_diameter(binary_image):
    graph = nx.grid_2d_graph(binary_image.shape[0], binary_image.shape[1])
    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i, j] == 0:  # 黑色像素不参与
                graph.remove_node((i, j))
    try:
        diameter = nx.diameter(graph)
    except nx.NetworkXError:
        diameter = 0  # 如果图是断开的，直径无法计算，设为0
    return diameter

# 计算分形维度
def compute_fractal_dimension(binary_image):
    Z = binary_image > 0
    sizes = range(1, min(Z.shape[0], Z.shape[1]) // 2, 1)  # 调整尺寸范围
    counts = []
    for size in sizes:
        count = 0
        for i in range(0, Z.shape[0], size):
            for j in range(0, Z.shape[1], size):
                sub_image = Z[i:i + size, j:j + size]
                if np.sum(sub_image) > 0:
                    count += 1
        counts.append(count)

    counts = np.array(counts)

    if len(counts) > 1:
        try:
            slopes = np.polyfit(np.log(sizes), np.log(counts), 1)
            return slopes[0]  # 返回分形维度
        except np.linalg.LinAlgError:
            return 0
    else:
        return 0

# 计算能量
def compute_energy(binary_image):
    return np.sum(binary_image ** 2)

# 计算每个图像的拓扑特征
def process_image(path, label):
    image = load_image(path)
    binary_image = preprocess_image(image)

    diameter = compute_graph_diameter(binary_image)
    fractal_dimension = compute_fractal_dimension(binary_image)
    energy = compute_energy(binary_image)

    return [diameter, fractal_dimension, energy, label]

# 提取图像的拓扑特征（使用并行处理）
def extract_features(image_paths, labels):
    features = []
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_image, image_paths, labels)
        features = list(results)
    return features

# 加载文件夹中的图像
def load_images_from_folder(folder_path, label, limit=3000):
    image_paths = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if img_path.endswith(".png") or img_path.endswith(".jpg"):
            image_paths.append(img_path)
        if len(image_paths) >= limit:  # 限制加载3000张图像
            break
    return image_paths, [label] * len(image_paths)

# 计算每个类别的特征统计数据
def compute_category_statistics(features, categories):
    category_features = {category: {'diameter': [], 'fractal_dimension': [], 'energy': []} for category in categories}

    for feature in features:
        category = feature[3]
        category_features[category]['diameter'].append(feature[0])
        category_features[category]['fractal_dimension'].append(feature[1])
        category_features[category]['energy'].append(feature[2])

    for category, data in category_features.items():
        avg_diameter = np.mean(data['diameter'])
        avg_fractal_dimension = np.mean(data['fractal_dimension'])
        avg_energy = np.mean(data['energy'])
        print(f"{category} 类别的拓扑特征：")
        print(f"  平均直径: {avg_diameter:.2f}")
        print(f"  平均分形维度: {avg_fractal_dimension:.2f}")
        print(f"  平均能量: {avg_energy:.2f}")
        print("-" * 30)

# 主函数：处理图像并提取特征
def main():
    original_folder = 'G:/GAN/data and pth/captcha_3000'
    first_migration_folder = 'G:/GAN/data and pth/captcha_3000_after_conversion/CycleGAN/style_monet'
    second_migration_folder = 'G:/GAN/data and pth/captcha_3000_after_conversion/CycleGAN/double/monet/monet_monet'

    # 加载文件夹中的图像，限制每类图像加载数
    original_images, original_labels = load_images_from_folder(original_folder, 'original', limit=3000)
    first_migration_images, first_migration_labels = load_images_from_folder(first_migration_folder, 'first_migration', limit=3000)
    second_migration_images, second_migration_labels = load_images_from_folder(second_migration_folder, 'second_migration', limit=3000)

    # 合并图像路径和标签
    image_paths = original_images + first_migration_images + second_migration_images
    labels = original_labels + first_migration_labels + second_migration_labels

    # 提取图像的拓扑特征
    features = extract_features(image_paths, labels)

    # 计算并打印每个类别的拓扑特征统计数据
    categories = ['original', 'first_migration', 'second_migration']
    compute_category_statistics(features, categories)

if __name__ == "__main__":
    main()
