import os
import random

import torch
from torch.utils.data import Dataset

from utils import img2tensor, load_img
from torchvision.transforms import Resize


class BlurDataset(Dataset):
    def __init__(self, root, target='train', level=""):
        super(Dataset, self).__init__()
        name_list = [filename for filename in os.listdir(os.path.join(root, level, target, "HR"))]

        self.HR_path = []
        self.LR_path = []

        for i, name in enumerate(name_list):
            self.HR_path.append(os.path.join(root, level, target, "HR", name))
            self.LR_path.append(os.path.join(root, level, target, "LR", name))

        self.length = len(self.HR_path)
        self.target = target
        self.resize = Resize((256, 256))
        # print(self.HR_path)
        # print(self.LR_path)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        hr_img = img2tensor(load_img(self.HR_path[idx], grayscale=False))
        lr_img = img2tensor(load_img(self.LR_path[idx], grayscale=False))

        _, file_name = os.path.split(self.LR_path[idx])
        # 调整图像大小
        hr_img = self.resize(hr_img)
        lr_img = self.resize(lr_img)

        # 稍微数据增强，旋转
        if self.target == "train":
            # 随机选择旋转次数 (1-4)
            i = random.choice([1, 2, 3, 4])

            # 对 HR 图像、LR 图像同步旋转
            hr_img = torch.rot90(hr_img, i, [1, 2])
            lr_img = torch.rot90(lr_img, i, [1, 2])

        return hr_img, lr_img, file_name
