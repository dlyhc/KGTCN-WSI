"""
画图-PSNR

Author: 罗涛
Date: 2024-10-12
"""

import pandas as pd
import matplotlib.pyplot as plt


csv_files = ['result/12_LR+0_5/psnr_updated.csv','result/LR+0_5 without blur kernel/psnr_updated.csv']  # 多个CSV文件的路径
label = ["A","B"]

# 创建一个绘图区域
plt.figure(figsize=(10, 6))

i = 0
# 遍历所有CSV文件，读取数据并绘制曲线
for file in csv_files:
    # 读取CSV文件
    data = pd.read_csv(file, nrows=150)

    # 绘制曲线，使用提取的标签
    plt.plot(data['Epoch'], data['PSNR'], label=label[i])
    i = i + 1

# 设置标题和标签
plt.title('PSNR Curve')
plt.xlabel('Epoch')
plt.ylabel('PSNR')

# 添加图例，位置设置为右下角
plt.legend(loc='lower right')

# 显示图形
plt.grid(True)
plt.show()