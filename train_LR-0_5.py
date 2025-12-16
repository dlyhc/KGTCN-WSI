import csv
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
import math

from tqdm import tqdm

from blur_kernel_extract.blur_kernel_estimator_net import KernelWizard
from blur_kernel_extract.blur_kernel_loss import CharbonnierLoss
from blur_kernel_extract.dataloader import BlurDataset
from utils import tensor2img, calculate_ssim

import sys
import os

# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def save_model(model, save_path, suf=''):
    save_path = os.path.join(save_path, 'model_{}.pth'.format(suf))
    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()


def cal_psnr_numpy(img1, img2, data_range=255):
    B, H, W, C = img1.shape
    mse = (img1 - img2) ** 2
    mse = np.mean(mse.reshape(B, -1), axis=1)
    return list(10 * np.log10(data_range ** 2 / mse))


def mkdir(p):
    isExists = os.path.exists(p)
    if isExists:
        pass
    else:
        os.makedirs(p)
        print("make directory successfully:{}".format(p))


def save_img(img, filepath):
    c = img.shape[-1]
    if c == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(filepath, img)


def save_multiple_img(img, name_list, save_path):
    B = img.shape[0]
    for i in range(B):
        save_img(img[i], os.path.join(save_path, name_list[i]))


epochs = 300
lr = 0.0001
batch_size = 64
torch.cuda.set_device(0)
device = torch.device('cuda:0')
model_save_path = "/home/lt/python/eventually/blur_kernel_extract/result_new/LR-0_5"
eval_interval = 10
level = "LR-0_5_split"

dataset_root = "/mnt/sda/lt/new_dataset"

# loss = nn.L1Loss().to(device)
loss = CharbonnierLoss().to(device)
model = KernelWizard()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, lr * 0.05)

train_dataset = BlurDataset(root=dataset_root, target="train", level=level)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
valid_dataset = BlurDataset(root=dataset_root, target="test", level=level)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)

model = model.to(device)

# 创建CSV文件并写入表头
loss_csv_path = os.path.join(model_save_path, "loss_log.csv")
psnr_csv_path = os.path.join(model_save_path, "psnr_log.csv")

# 初始化Loss CSV文件
with open(loss_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Loss", "ValLoss"])

# 初始化PSNR CSV文件
with open(psnr_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "PSNR", "SSIM"])

# 训练过程
max_psnr = 0
min_valLoss = math.inf
for epoch in range(epochs + 1):
    epoch_loss = 0
    model.train()
    for iteration, (hr_img, lr_img, _) in enumerate(train_dataloader):
        optimizer.zero_grad()
        hr_img = hr_img.type(torch.FloatTensor).to(device)
        lr_img = lr_img.type(torch.FloatTensor).to(device)
        # 模型输出的模糊核
        kernel_mean, kernel_sigma = model.forward(lr_img)
        kernel = kernel_mean.detach() + kernel_sigma.detach() * torch.randn_like(kernel_mean.detach())
        fake_LQ = model.adaptKernel(hr_img, kernel)

        # 计算L1损失
        batch_loss = loss(fake_LQ, lr_img)
        epoch_loss += batch_loss.item()

        # 反向传播
        batch_loss.backward()
        optimizer.step()

    scheduler.step()
    # 打印损失
    epoch_loss = epoch_loss / len(train_dataloader)

    # 验证过程
    val_loss = 0
    if (epoch + 1) % eval_interval == 0:
        print("begin valid...")
        model.eval()
        with torch.no_grad():
            psnr = []
            ssim = []
            for _, data in enumerate(tqdm(valid_dataloader)):
                val_hr_img, val_lr_img, name_list = data
                val_hr_img = val_hr_img.type(torch.FloatTensor).to(device)
                val_lr_img = val_lr_img.type(torch.FloatTensor).to(device)
                kernel_mean, kernel_sigma = model(val_lr_img)
                # 使用输出的模糊核对清晰图像进行模糊
                kernel = kernel_mean.detach() + kernel_sigma.detach() * torch.randn_like(kernel_mean.detach())
                fake_LQ = model.adaptKernel(val_hr_img, kernel)

                loss_ = loss(fake_LQ, val_lr_img)
                val_loss += loss_.item()

                fake_LQ = tensor2img(fake_LQ.detach().cpu())
                val_lr_img = tensor2img(val_lr_img.detach().cpu())

                file_path = os.path.join(model_save_path, 'val_img')
                mkdir(file_path)

                save_multiple_img(fake_LQ, name_list, file_path)

                psnr += cal_psnr_numpy(fake_LQ, val_lr_img, data_range=255)
                # ssim += calculate_ssim(fake_LQ, val_lr_img, crop_border=0, input_order='HWC', test_y_channel=False)

            val_loss = val_loss / len(valid_dataloader)
            psnr_mean = np.mean(psnr)
            # ssim_mean = np.mean(ssim)
            print(f"psnr: {psnr_mean:.4f}")
            # print(f"ssim: {ssim_mean:.4f}")

            # 保存PSNR到CSV文件
            with open(psnr_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, psnr_mean, 0])

            if min_valLoss > val_loss:
                min_valLoss = min(min_valLoss, val_loss)
                print("save best to model_minLoss.pth...")
                save_model(model, model_save_path, suf=str("minLoss"))

            # 保存最佳模型
            if psnr_mean > max_psnr:
                max_psnr = max(psnr_mean, max_psnr)
                print("save best to model_best.pth...")
                save_model(model, model_save_path, suf=str("best"))

        print("end valid...")

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, valLoss: {val_loss:.4f}")
    # 保存Loss到CSV文件
    with open(loss_csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, epoch_loss, val_loss])
