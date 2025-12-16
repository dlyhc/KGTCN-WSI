import pandas as pd
import matplotlib.pyplot as plt

csv_files = ['loss_log.csv']  # 多个CSV文件的路径
label = ["train loss"]

# 创建一个绘图区域
plt.figure(figsize=(10, 6))

# 遍历所有CSV文件，读取数据并绘制曲线
i =0
for file in csv_files:
    # 读取CSV文件
    data = pd.read_csv(file, nrows=250)

    # 绘制曲线，使用提取的标签
    plt.plot(data['Epoch'], data['Loss'], label=label[i])
    i = i+1

# 设置标题和标签
plt.title('Train Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 添加图例，位置设置为右下角
plt.legend(loc='upper right')

# 显示图形
plt.grid(True)
plt.show()