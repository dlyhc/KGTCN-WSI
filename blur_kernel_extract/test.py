"""
模糊核提取测试，以及npy生成

Author: 罗涛
Date: 2024-12-16
"""

import os
import numpy as np
import torch
from blur_kernel_extract.blur_kernel_estimator_net import KernelWizard
from utils import save_img, load_img, img2tensor

def save_multiple_img(img, name_list, save_path):
    B = img.shape[0]
    for i in range(B):
        save_img(img[i], os.path.join(save_path, name_list[i]))

# 设置CUDA设备
torch.cuda.set_device(1)
device = torch.device("cuda:1")

# 路径
item = "LR+0_5"
dataset = "test"

model_path = "/home/lt/python/eventually/blur_kernel_extract/result_new/{}/model_best.pth".format(item)
# 测试图像所在的文件夹
test_folder = "/mnt/sda/lt/new_dataset/{}_newdeblur/{}/LR".format(item,dataset)
#
save_kernel_path = "/mnt/sda/lt/new_dataset/{}_newdeblur/{}/LR_kernel".format(item,dataset)
# save_SR_path = "/mnt/sda/lt/real/RAT/{}/test/SR".format(item)


# 加载模型
model = KernelWizard()
model.eval()
model.load_state_dict(torch.load(model_path))
model = model.to(device)

# 获取文件夹中的所有图像
test_images = [f for f in os.listdir(test_folder) if f.endswith('.png') or f.endswith('.jpg')]
# 保存模糊核为 .npy 文件
if not os.path.exists(save_kernel_path):
    os.makedirs(save_kernel_path)

# if not os.path.exists(save_SR_path):
#     os.makedirs(save_SR_path)

with torch.no_grad():
    # 处理每个图像
    for test_img_name in test_images:
        test_path = os.path.join(test_folder, test_img_name)
        test_img = load_img(test_path)
        test_tensor = img2tensor(test_img).unsqueeze(0).to(device)

        kernel_mean, kernel_sigma = model(test_tensor)

        kernel = kernel_mean.detach().cpu().numpy()

        # 保存模糊核为 .npy 文件
        kernel_name = os.path.splitext(test_img_name)[0] + ".npy"
        np.save(os.path.join(save_kernel_path, kernel_name), kernel)

        # 模糊核模糊清晰图像
        # test_LR_tensor = model.adaptKernel(test_tensor, kernel_mean)
        #
        # LQ_img = tensor2img(test_LR_tensor.detach().cpu())
        #
        # # 保存结果
        # save_multiple_img(LQ_img, [test_img_name], save_SR_path)
