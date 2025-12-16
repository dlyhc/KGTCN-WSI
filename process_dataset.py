"""
用来处理数据集的

Author: 罗涛
Date: 2024-12-18
"""

# import os
# import re
# from shutil import move
#
#
# def classify_images(root_dir, new_root):
#     # 定义目标文件夹
#     target_folders = ['HR', 'LR-0_5', 'LR-5_10', 'LR+0_5', 'LR+5_10']
#     for folder in target_folders:
#         os.makedirs(os.path.join(new_root, folder), exist_ok=True)
#
#     # 遍历根目录下的子目录
#     for parent_dir_name in os.listdir(root_dir):
#         current_dir = os.path.join(root_dir, parent_dir_name)
#         # 跳过非目录和目标文件夹
#         if not os.path.isdir(current_dir) or parent_dir_name in target_folders:
#             continue
#
#         # 收集当前目录下的所有jpg文件并按Seg分组
#         seg_groups = {}
#         pattern = re.compile(r'(Seg\d+)_defocus(-?\d+)\.jpg$')
#         for filename in os.listdir(current_dir):
#             if not filename.lower().endswith('.jpg'):
#                 continue
#             match = pattern.match(filename)
#             if not match:
#                 continue
#             seg_name, defocus_str = match.groups()
#             defocus = int(defocus_str)
#             if seg_name not in seg_groups:
#                 seg_groups[seg_name] = []
#             seg_groups[seg_name].append((filename, defocus))
#
#         # 处理每个Seg组
#         for seg_name, files in seg_groups.items():
#             if not files:
#                 continue
#             # 按defocus绝对值排序
#             sorted_files = sorted(files, key=lambda x: abs(x[1]))
#             # 处理HR文件
#             hr_filename, hr_defocus = sorted_files[0]
#             hr_new_name = f"{parent_dir_name}_{hr_filename}"
#             hr_dest = os.path.join(new_root, 'HR', hr_new_name)
#             move(os.path.join(current_dir, hr_filename), hr_dest)
#
#             # 处理剩余文件
#             for file_info in sorted_files[1:]:
#                 filename, defocus = file_info
#                 absolute = abs(defocus)
#                 # 确定正负和范围
#                 sign = '+' if defocus >= 0 else '-'
#                 if absolute < 5500:
#                     lr_range = '0_5'
#                 else:
#                     lr_range = '5_10'
#                 # 目标文件夹
#                 target_folder = f'LR{sign}{lr_range}'
#                 # 新文件名
#                 new_name = f"{parent_dir_name}_{filename}"
#                 dest_path = os.path.join(new_root, target_folder, new_name)
#                 # 移动文件
#                 move(os.path.join(current_dir, filename), dest_path)
#
#
# if __name__ == '__main__':
#     root_directory = '/mnt/sda/lt/new_dataset/incoherent_RGBchannels/train_incoherent_RGBChannels'
#     new_root = "/mnt/sda/lt/new_dataset/incoherent_RGBchannels"
#     classify_images(root_directory, new_root)
#     print("分类完成！")



#
# import os
# import shutil
#
#
# def organize_hr_lr_pairs(original_hr_dir, original_lr_dir, new_hr_dir, new_lr_dir):
#     # 创建新的文件夹
#     os.makedirs(new_hr_dir, exist_ok=True)
#     os.makedirs(new_lr_dir, exist_ok=True)
#
#     # 构建HR前缀到文件路径的映射
#     hr_prefix_map = {}
#     for hr_filename in os.listdir(original_hr_dir):
#         if hr_filename.lower().endswith('.jpg'):
#             parts = hr_filename.split('_defocus')
#             if len(parts) < 2:
#                 continue  # 跳过不符合格式的文件名
#             prefix = parts[0] + '_defocus'
#             hr_path = os.path.join(original_hr_dir, hr_filename)
#             hr_prefix_map[prefix] = hr_path
#
#     # 处理每个LR文件
#     for lr_filename in os.listdir(original_lr_dir):
#         if lr_filename.lower().endswith('.jpg'):
#             parts = lr_filename.split('_defocus')
#             if len(parts) < 2:
#                 continue  # 跳过不符合格式的文件名
#             prefix = parts[0] + '_defocus'
#             if prefix in hr_prefix_map:
#                 # 对应的HR文件路径
#                 hr_source_path = hr_prefix_map[prefix]
#                 # 新HR文件路径（重命名为当前LR文件名）
#                 hr_dest_path = os.path.join(new_hr_dir, lr_filename)
#                 # 复制HR文件到新目录
#                 shutil.copy2(hr_source_path, hr_dest_path)
#                 # 移动LR文件到新目录
#                 lr_source_path = os.path.join(original_lr_dir, lr_filename)
#                 lr_dest_path = os.path.join(new_lr_dir, lr_filename)
#                 # 确保目标目录存在（理论上已存在）
#                 shutil.copy2(lr_source_path, lr_dest_path)
#             else:
#                 print(f"No corresponding HR file found for LR file: {lr_filename}")
#                 break
#
# type = "LR+5_10"
#
# # 使用示例
# organize_hr_lr_pairs(
#     original_hr_dir='/mnt/sda/lt/new_dataset/same/HR',
#     original_lr_dir='/mnt/sda/lt/new_dataset/same/{}'.format(type),
#     new_hr_dir='/mnt/sda/lt/new_dataset/test_dataset/same/{}/HR'.format(type),
#     new_lr_dir='/mnt/sda/lt/new_dataset/test_dataset/same/{}/LR'.format(type)
# )






# import os
# import random
# import shutil
#
# def filter_and_split_dataset(hr_dir, lr_dir, output_dir, total_samples=5000, train_ratio=0.8):
#     """
#     从数据集中随机选取指定数量的样本，并按比例划分训练集和验证集
#     :param hr_dir: HR 图片目录
#     :param lr_dir: LR 图片目录
#     :param output_dir: 输出目录
#     :param total_samples: 总样本数（默认 5000）
#     :param train_ratio: 训练集比例（默认 0.8）
#     """
#     # 获取所有文件名（假设 HR 和 LR 的文件名完全一致）
#     hr_files = [f for f in os.listdir(hr_dir) if f.endswith('.jpg')]
#     lr_files = [f for f in os.listdir(lr_dir) if f.endswith('.jpg')]
#
#     # 确保 HR 和 LR 的文件名一致
#     assert set(hr_files) == set(lr_files), "HR 和 LR 的文件名不匹配！"
#
#     # 如果总样本数超过实际文件数，则调整 total_samples
#     if total_samples > len(hr_files):
#         print(f"警告：总样本数 {total_samples} 超过实际文件数 {len(hr_files)}，将使用全部文件。")
#         total_samples = len(hr_files)
#
#     # 随机选取指定数量的样本
#     random.shuffle(hr_files)
#     selected_files = hr_files[:total_samples]
#
#     # 计算训练集和验证集的分割点
#     split_idx = int(total_samples * train_ratio)
#     train_files = selected_files[:split_idx]
#     val_files = selected_files[split_idx:]
#
#     # 创建输出目录
#     os.makedirs(os.path.join(output_dir, 'train', 'HR'), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, 'train', 'LR'), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, 'val', 'HR'), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, 'val', 'LR'), exist_ok=True)
#
#     # 复制训练集
#     for file in train_files:
#         shutil.copy(os.path.join(hr_dir, file), os.path.join(output_dir, 'train', 'HR', file))
#         shutil.copy(os.path.join(lr_dir, file), os.path.join(output_dir, 'train', 'LR', file))
#
#     # 复制验证集
#     for file in val_files:
#         shutil.copy(os.path.join(hr_dir, file), os.path.join(output_dir, 'val', 'HR', file))
#         shutil.copy(os.path.join(lr_dir, file), os.path.join(output_dir, 'val', 'LR', file))
#
#     print(f"数据集处理完成！总样本数: {total_samples} 张")
#     print(f"训练集: {len(train_files)} 张（HR 和 LR 各 {len(train_files)} 张）")
#     print(f"验证集: {len(val_files)} 张（HR 和 LR 各 {len(val_files)} 张）")
#
# # 示例调用
# filter_and_split_dataset(
#     hr_dir='/mnt/sda/lt/new_dataset/LR+5_10/HR',
#     lr_dir='/mnt/sda/lt/new_dataset/LR+5_10/LR',
#     output_dir='/mnt/sda/lt/new_dataset/LR+5_10_5000',
#     total_samples=5000,
#     train_ratio=0.8
# )




import os
import random
import shutil

def split_dataset(hr_dir, lr_dir, output_dir, train_ratio=0.8):
    """
    划分数据集为训练集和验证集
    :param hr_dir: HR 图片目录
    :param lr_dir: LR 图片目录
    :param output_dir: 输出目录
    :param train_ratio: 训练集比例（默认 0.8）
    """
    # 获取所有文件名（假设 HR 和 LR 的文件名完全一致）
    hr_files = [f for f in os.listdir(hr_dir) if f.endswith('.jpg')]
    lr_files = [f for f in os.listdir(lr_dir) if f.endswith('.jpg')]

    # 确保 HR 和 LR 的文件名一致
    assert set(hr_files) == set(lr_files), "HR 和 LR 的文件名不匹配！"

    # 随机打乱文件名
    random.shuffle(hr_files)

    # 计算训练集和验证集的分割点
    split_idx = int(len(hr_files) * train_ratio)
    train_files = hr_files[:split_idx]
    val_files = hr_files[split_idx:]

    # 创建输出目录
    os.makedirs(os.path.join(output_dir, 'train', 'HR'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'LR'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'HR'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'LR'), exist_ok=True)

    # 复制训练集
    for file in train_files:
        shutil.copy(os.path.join(hr_dir, file), os.path.join(output_dir, 'train', 'HR', file))
        shutil.copy(os.path.join(lr_dir, file), os.path.join(output_dir, 'train', 'LR', file))

    # 复制验证集
    for file in val_files:
        shutil.copy(os.path.join(hr_dir, file), os.path.join(output_dir, 'test', 'HR', file))
        shutil.copy(os.path.join(lr_dir, file), os.path.join(output_dir, 'test', 'LR', file))

    print(f"数据集划分完成！训练集: {len(train_files)} 张，验证集: {len(val_files)} 张")

type = "LR+0_5"

split_dataset(
    hr_dir='/mnt/sda/lt/new_dataset/{}/HR'.format(type),
    lr_dir='/mnt/sda/lt/new_dataset/{}/LR'.format(type),
    output_dir='/mnt/sda/lt/new_dataset/{}_newdeblur'.format(type)
)




# import os
# import re
# import shutil
#
#
# def classify_images(root_dir, new):
#     # 定义目标文件夹
#     target_dirs = ['HR', 'LR+0_5', 'LR+5_10', 'LR-0_5', 'LR-5_10']
#     for dir_name in target_dirs:
#         os.makedirs(os.path.join(new, dir_name), exist_ok=True)
#
#     # 遍历根目录下的每个子文件夹
#     for subdir_name in os.listdir(root_dir):
#         subdir_path = os.path.join(root_dir, subdir_name)
#         if not os.path.isdir(subdir_path) or subdir_name in target_dirs:
#             continue  # 跳过文件和目标文件夹
#
#         # 收集并解析所有jpg文件
#         parsed_files = []
#         for fname in os.listdir(subdir_path):
#             if not fname.lower().endswith('.jpg'):
#                 continue
#             match = re.match(r'defocus(-?\d+)\.jpg$', fname, re.IGNORECASE)
#             if not match:
#                 continue
#             num = int(match.group(1))
#             parsed_files.append((fname, num))
#
#         if not parsed_files:
#             continue
#
#         # 处理HR文件（最小绝对值）
#         min_abs = min(abs(num) for (_, num) in parsed_files)
#         hr_candidates = [(f, n) for (f, n) in parsed_files if abs(n) == min_abs]
#
#         # 复制HR文件
#         for fname, _ in hr_candidates:
#             src = os.path.join(subdir_path, fname)
#             dst = os.path.join(new, 'HR', f"{subdir_name}_{fname}")
#             shutil.copy2(src, dst)
#
#         # 处理其他文件
#         remaining = [item for item in parsed_files if item not in hr_candidates]
#         for fname, num in remaining:
#             if num >= 0:
#                 target = 'LR+5_10' if num >= 5500 else 'LR+0_5'
#             else:
#                 abs_num = abs(num)
#                 target = 'LR-5_10' if abs_num >= 5500 else 'LR-0_5'
#
#             src = os.path.join(subdir_path, fname)
#             dst = os.path.join(new, target, f"{subdir_name}_{fname}")
#             shutil.copy2(src, dst)
#
#
# if __name__ == '__main__':
#     root_directory = '/mnt/sda/lt/new_dataset/incoherent_RGBchannels/testRawData_incoherent_diffProtocol'
#     new_root = "/mnt/sda/lt/new_dataset/diff"
#     classify_images(root_directory, new_root)
#     print("分类完成！")

