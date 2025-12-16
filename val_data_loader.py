import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import numpy as np
from utils import load_img, img2tensor  # 假设这些函数已正确实现
from PIL import Image


class ValidData(Dataset):
    def __init__(self, dataset_path):
        super(Dataset, self).__init__()
        # 从文件夹加载数据
        self.hr_folder = os.path.join(dataset_path, 'hr_img')  # 高分辨率图片文件夹
        self.lr_folder = os.path.join(dataset_path, 'lr_img')  # 低分辨率图片文件夹
        self.mask_folder = os.path.join(dataset_path, 'mask')  # mask文件夹
        self.reg_mask_folder = os.path.join(dataset_path, 'reg_mask')  # reg_mask文件夹

        # 获取文件名列表
        self.hr_files = sorted(os.listdir(self.hr_folder))  # 获取高分辨率图片的文件名列表
        self.lr_files = sorted(os.listdir(self.lr_folder))  # 获取低分辨率图片的文件名列表
        self.mask_files = sorted(os.listdir(self.mask_folder))  # 获取mask文件的文件名列表
        self.reg_mask_files = sorted(os.listdir(self.reg_mask_folder))  # 获取reg_mask文件的文件名列表

        # 确保所有文件夹中数据数量一致
        assert len(self.hr_files) == len(self.lr_files) == len(self.mask_files) == len(self.reg_mask_files), "文件数量不匹配!"

        self.length = len(self.hr_files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 加载文件路径
        hr_path = os.path.join(self.hr_folder, self.hr_files[idx])
        lr_path = os.path.join(self.lr_folder, self.lr_files[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_files[idx])
        reg_mask_path = os.path.join(self.reg_mask_folder, self.reg_mask_files[idx])
        file_name = self.hr_files[idx]

        # 加载图片和掩码
        hr_img = load_img(hr_path)
        lr_img = load_img(lr_path)
        mask = np.load(mask_path)
        reg_mask = np.load(reg_mask_path)

        # 转换为PyTorch张量
        hr_img = img2tensor(hr_img)
        lr_img = img2tensor(lr_img)
        mask = torch.tensor(mask, dtype=torch.int).squeeze(0)
        reg_mask = torch.tensor(reg_mask, dtype=torch.int).squeeze(0)

        return hr_img, lr_img, mask, reg_mask, file_name
