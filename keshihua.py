import matplotlib.pyplot as plt
import numpy as np

# 设置字体为微软雅黑，支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据
categories = ['原始', '第一次风格迁移', '第二次风格迁移']
avg_topology = [214.97, 0.00, 8.27]
avg_fractal_dimension = [-1.77, -1.74, -1.74]
avg_accuracy = [13374.83, 10831.03, 10716.50]

# 创建图表
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制拓扑值的柱状图
ax1.bar(categories, avg_topology, color='b', alpha=0.6, label='平均拓扑值')
ax1.set_xlabel('类别')
ax1.set_ylabel('平均拓扑值', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# 绘制分形维度的折线图
ax2 = ax1.twinx()
ax2.plot(categories, avg_fractal_dimension, color='g', marker='o', label='平均分形维度', linestyle='--')
ax2.set_ylabel('平均分形维度', color='g')
ax2.tick_params(axis='y', labelcolor='g')

# 绘制准确度的折线图
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # 将第三个y轴向右移动
ax3.plot(categories, avg_accuracy, color='r', marker='x', label='平均准确度', linestyle='-.')
ax3.set_ylabel('平均准确度', color='r')
ax3.tick_params(axis='y', labelcolor='r')

# 添加标题并显示图表
plt.title('拓扑值、分形维度与准确度数据可视化')
fig.tight_layout()
plt.show()
