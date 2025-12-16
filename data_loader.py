"""
自定义数据载入

Author: 罗涛
Date: 2025-5-15
"""

import numpy as np
from utils import load_img, img2tensor
import torch
from torch.utils.data import Dataset, DataLoader
import os


class PairedData(Dataset):
    def __init__(self, root, target='train'):
        super(Dataset, self).__init__()
        name_list = os.listdir(os.path.join(root, target, "HR"))

        self.HR_path = []
        self.LR_path = []
        self.LR_kernel = []

        num = 0
        for i, name in enumerate(name_list):
            self.HR_path.append(os.path.join(root, target, "HR", name))
            self.LR_path.append(os.path.join(root, target, "LR", name))
            self.LR_kernel.append(os.path.join(root, target, "LR_kernel", name.split(".")[0] + ".npy"))

            # 这里主要用来控制载入多少数据量，需要控制的话解开注释，不需要就别管
            # num = num + 1
            # if target == "train" and num == 2000:
            #     break
            #
            # if target == "test" and num == 300:
            #     break

        self.length = len(self.HR_path)
        self.target = target

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        hr_img = img2tensor(load_img(self.HR_path[idx], grayscale=False))
        lr_img = img2tensor(load_img(self.LR_path[idx], grayscale=False))

        kernel_path = self.LR_kernel[idx]
        kernel = np.load(kernel_path)
        kernel = torch.from_numpy(kernel).float()
        # 加入batch_size维度，以匹配输出
        kernel = kernel.squeeze(0)

        _, file_name = os.path.split(self.LR_path[idx])

        return hr_img, lr_img, kernel, file_name

# 测试
if __name__ == '__main__':
    dataset_path = "/mnt/sda/lt/new_dataset/{}_deblur".format("LR+0_5")
    # 创建 PairedData 实例
    dataset = PairedData(root=dataset_path, target="test")

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    print(len(dataloader))
    # 遍历 DataLoader 进行测试
    for i, (hr_img, lr_img, kernel, file_name) in enumerate(dataloader):

        print(hr_img)
        print(f"Batch {i + 1}:")
        print(f"HR Image Shape: {hr_img.shape}")
        print(f"LR Image Shape: {lr_img.shape}")
        print(f"Kernel Shape: {kernel.shape}")
        print(f"File Names: {file_name}")

        # 限制只输出前几个批次
        if i == 2:
            break
