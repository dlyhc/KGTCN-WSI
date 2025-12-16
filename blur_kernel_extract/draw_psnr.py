import pandas as pd
import matplotlib.pyplot as plt

csv_files = ['./result/LR+5_10/psnr_log.csv','./result/LR-5_10/psnr_log.csv','./result/LR+0_5/psnr_log.csv', './result/LR-0_5/psnr_log.csv']  # 多个CSV文件的路径

# 创建一个绘图区域
plt.figure(figsize=(10, 6))

# 遍历所有CSV文件，读取数据并绘制曲线
for file in csv_files:
    # 读取CSV文件
    data = pd.read_csv(file)

    # 提取中间部分作为图例标签
    label = file.split('/')  # 假设路径格式为 './result/LR-0_5/psnr_log.csv'

    # 绘制曲线，使用提取的标签
    plt.plot(data['Epoch'], data['PSNR'], label=label[2])

# 设置标题和标签
plt.title('PSNR Curve')
plt.xlabel('Epoch')
plt.ylabel('PSNR')

# 添加图例，位置设置为右下角
plt.legend(loc='lower right')

# 显示图形
plt.grid(True)
plt.show()