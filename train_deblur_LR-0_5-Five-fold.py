import csv
import math
import os
import random
import sys
from datetime import datetime
from sklearn.model_selection import KFold  # 交叉验证工具

import numpy as np
import torch
import matplotlib

from pytorch_msssim import MS_SSIM
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset
import lpips  # 新增LPIPS库

from loss import PerceptualLoss

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from model import Generator, Discriminator
from data_loader import PairedData
from utils import tensor2img, save_img, calculate_ssim


def mkdir(p):
    isExists = os.path.exists(p)
    if not isExists:
        os.makedirs(p)
        print(f"make directory successfully:{p}")


def cal_psnr_numpy(img1, img2, data_range=255):
    B, H, W, C = img1.shape
    mse = (img1 - img2) ** 2
    mse = np.mean(mse.reshape(B, -1), axis=1)
    return list(10 * np.log10(data_range ** 2 / mse))


# 新增：计算MAE（Mean Absolute Error）
def cal_mae_numpy(img1, img2, data_range=255):
    """基于numpy计算批量MAE，返回每个样本的MAE列表"""
    B, H, W, C = img1.shape
    abs_error = np.abs(img1 - img2)
    mae = np.mean(abs_error.reshape(B, -1), axis=1)  # 每个样本的平均绝对误差
    return list(mae)


def calculate_ssim_batch(SRs, HRs):
    batch_ssim = 0.0
    for i in range(SRs.shape[0]):
        SR = SRs[i]
        HR = HRs[i]
        batch_ssim += calculate_ssim(SR, HR, crop_border=0, input_order='HWC', test_y_channel=False)
    return batch_ssim / SRs.shape[0]


# 新增：计算LPIPS（Learned Perceptual Image Patch Similarity）
def calculate_lpips_batch(SRs, HRs, lpips_model, device, data_range=255):
    """
    批量计算LPIPS，注意输入格式转换：
    - SRs/HRs: numpy数组 (B, H, W, C)，范围[0, 255]
    - 输出：批量LPIPS均值
    """
    # 转换为torch张量并调整维度为(C, H, W)，范围归一化到[-1, 1]（LPIPS要求）
    sr_tensor = torch.from_numpy(SRs).permute(0, 3, 1, 2).float() / (data_range / 2) - 1.0
    hr_tensor = torch.from_numpy(HRs).permute(0, 3, 1, 2).float() / (data_range / 2) - 1.0
    
    sr_tensor = sr_tensor.to(device)
    hr_tensor = hr_tensor.to(device)
    
    # 计算LPIPS（每个样本的LPIPS值）
    lpips_values = lpips_model(sr_tensor, hr_tensor)
    return torch.mean(lpips_values).item()  # 返回批量均值


def save_multiple_img(img, name_list, save_path):
    B = img.shape[0]
    for i in range(B):
        save_img(img[i], os.path.join(save_path, name_list[i]))


def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_model(model, save_path, suf=''):
    save_path = os.path.join(save_path, f'model_{suf}.pth')
    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()


def print_fixed_log(epoch, num_epochs, iteration, len_dataloader, loss_D, loss_pixel, loss_GAN, loss_G):
    message = (
        '[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Identity/Advers/Total): %.4f/%.4f/%.4f'
        % (epoch, num_epochs, iteration, len_dataloader, loss_D, loss_pixel, loss_GAN, loss_G)
    )
    sys.stdout.write("\r" + message)
    sys.stdout.flush()


def print_fixed_log_without_gan(epoch, num_epochs, iteration, len_dataloader, loss_pixel, loss_G):
    message = (
        '[%d/%d][%d/%d] Generator_Loss (Identity/Total): %.4f/%.4f'
        % (epoch, num_epochs, iteration, len_dataloader, loss_pixel, loss_G)
    )
    sys.stdout.write("\r" + message)
    sys.stdout.flush()


# 新增：通用绘图函数（支持所有指标）
def plot_metric(data, label, save_path):
    l = len(data)
    axis = np.linspace(1, l, l)

    fig = plt.figure()
    plt.title(label)
    plt.plot(axis, data, label=label, color='blue' if label != 'LPIPS' else 'red')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close(fig)


# 超参数设置
run_from = None
type = "LR-0_5"
home_path = f"/home/lt/python/eventually/result_new_ssim/{type}"
g_lr = 0.0001
d_lr = 0.00001
eval_interval = 1
start_epoch = 1
num_epochs = 300
l = 0.1  # 对抗损失占比
p = 0.1  # 感知损失占比
gan = False

patch_size = 224
batch_size = 16
middle_block_num = 12
base_channel = 32

dataset_path = f"/mnt/sda/lt/new_dataset/{type}_newdeblur"

torch.cuda.set_device(1)
device = torch.device('cuda:1')
tensor = torch.cuda.FloatTensor
patch = (1, patch_size // 2 ** 4, patch_size // 2 ** 4)
set_seeds()

# 新增：初始化LPIPS模型（使用预训练的AlexNet，适用于图像恢复任务）
lpips_model = lpips.LPIPS(net='alex', verbose=False).to(device)
lpips_model.eval()  # 仅用于评估，固定参数

# 五折交叉验证设置
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# 加载完整数据集
full_dataset = PairedData(root=dataset_path, target='train')
dataset_size = len(full_dataset)
indices = list(range(dataset_size))


# 定义每折的训练函数（更新指标相关部分）
def train_fold(fold, train_indices, val_indices):
    # 创建当前折的结果目录
    run = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    fold_home = os.path.join(home_path, f'fold_{fold}')
    fold_run_home = os.path.join(fold_home, run)
    
    # 创建必要目录
    file_path = os.path.join(fold_run_home, 'checkpoint')
    mkdir(file_path)
    sr_path = os.path.join(fold_run_home, 'sr_img')
    mkdir(sr_path)
    
    # 初始化日志文件（新增MAE和LPIPS列）
    loss_csv_path = os.path.join(fold_run_home, "loss_log.csv")
    metric_csv_path = os.path.join(fold_run_home, "metric_log.csv")  # 统一存储所有指标
    
    with open(loss_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss"])
    
    with open(metric_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "PSNR", "SSIM", "MAE", "LPIPS"])  # 新增MAE和LPIPS
    
    # 初始化模型（每个折独立初始化）
    generator = Generator(base_channel=base_channel, middle_blk_num=middle_block_num)
    generator.to(device)
    if gan:
        discriminator = Discriminator()
        discriminator.to(device)
    
    # 创建当前折的数据集和数据加载器
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    
    dataloader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=4
    )
    
    dataloader_val = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=4
    )
    
    # 优化器设置
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=g_lr)
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, num_epochs, g_lr * 0.05)
    
    if gan:
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=d_lr)
        scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, num_epochs, d_lr * 0.05)
    
    # 损失函数
    criterionL = nn.L1Loss().to(device)
    criterionSSIM = MS_SSIM(data_range=1, size_average=True, channel=3).to(device)
    criterionMSE = nn.MSELoss().to(device)
    
    # 指标记录（新增MAE和LPIPS）
    record_metrics = {
        "psnr": [],
        "ssim": [],
        "mae": [],
        "lpips": [],
        "loss": []
    }
    
    # 最佳指标记录（新增MAE和LPIPS）
    best_metrics = {
        "psnr": 0,
        "ssim": 0,
        "mae": math.inf,  # MAE越小越好
        "lpips": math.inf  # LPIPS越小越好
    }
    loss_min = math.inf
    
    print(f"\n>>>> Starting Fold {fold}/{num_folds} Training....")
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_loss = 0
        gan_loss = 0
        total_loss = 0
        dis_loss = 0
        generator.train()
        if gan:
            discriminator.train()
        
        epoch_begin_time = datetime.now()
        for iteration, (hr_img, lr_img, lr_kernel, filename) in enumerate(dataloader):
            real_mid = Variable(lr_img.type(tensor).to(device), requires_grad=False)
            real_high = Variable(hr_img.type(tensor).to(device), requires_grad=False)
            lr_kernel = Variable(lr_kernel.type(tensor).to(device), requires_grad=False)
            
            if gan:
                valid = Variable(tensor(np.ones((real_mid.size(0), *patch))).to(device), requires_grad=False)
                fake = Variable(tensor(np.zeros((real_mid.size(0), *patch))).to(device), requires_grad=False)
            
            # 训练生成器
            optimizer_G.zero_grad()
            fake_high = generator(real_mid, lr_kernel)
            
            if gan:
                pred_fake = discriminator(fake_high, real_mid)
                loss_GAN = criterionMSE(pred_fake, valid)
            
            lossL1 = criterionL(fake_high, real_high)
            loss_pixel = lossL1
            
            if gan:
                loss_G = l * loss_GAN + (1 - l) * loss_pixel
            else:
                loss_G = loss_pixel
            
            loss_G.backward()
            optimizer_G.step()
            total_loss += loss_G.item()
            
            if gan:
                gan_loss += loss_GAN.item()
            
            # 训练判别器
            if gan:
                optimizer_D.zero_grad()
                pred_real = discriminator(real_high, real_mid)
                loss_real = criterionMSE(pred_real, valid)
                pred_fake = discriminator(fake_high.detach(), real_mid)
                loss_fake = criterionMSE(pred_fake, fake)
                loss_D = 0.5 * (loss_real + loss_fake)
                loss_D.backward()
                optimizer_D.step()
                dis_loss += loss_D.item()
            
            epoch_loss += loss_pixel.item()
            
            # 打印日志
            if gan:
                print_fixed_log(
                    epoch, num_epochs, iteration, len(dataloader),
                    loss_D.item(), loss_pixel.item(), loss_GAN.item(), loss_G.item()
                )
            else:
                print_fixed_log_without_gan(
                    epoch, num_epochs, iteration, len(dataloader),
                    loss_pixel.item(), loss_G.item()
                )
        
        epoch_end_time = datetime.now()
        print(f"\n>>> Fold {fold} Epoch {epoch} time cost: {(epoch_end_time - epoch_begin_time).total_seconds()} seconds")
        
        # 验证过程（更新指标计算，新增MAE和LPIPS）
        if epoch % eval_interval == 0:
            psnr_list = []
            ssim_total = 0.0
            mae_list = []
            lpips_total = 0.0
            
            generator.eval()
            if gan:
                discriminator.eval()
            
            with torch.no_grad():
                for batch_idx, data in enumerate(tqdm(dataloader_val, desc=f"Fold {fold} Validation")):
                    hr_img, lr_img, lr_kernel, name_list = data
                    lr_img = lr_img.type(torch.FloatTensor).to(device)
                    lr_kernel = lr_kernel.type(torch.FloatTensor).to(device)
                    
                    sr_img = generator(lr_img, lr_kernel)
                    sr_img = sr_img.detach().cpu()
                    
                    # 转换为numpy数组（HWC格式，0-255）
                    sr_img_np = tensor2img(sr_img)
                    hr_img_np = tensor2img(hr_img)
                    
                    # 计算各指标
                    psnr_list += cal_psnr_numpy(sr_img_np, hr_img_np, data_range=255)
                    mae_list += cal_mae_numpy(sr_img_np, hr_img_np, data_range=255)
                    ssim_total += calculate_ssim_batch(sr_img_np, hr_img_np)
                    lpips_total += calculate_lpips_batch(sr_img_np, hr_img_np, lpips_model, device)
                    
                    # 保存恢复图像（仅首折前3个batch，避免占用过多空间）
                    if fold == 1 and batch_idx < 3:
                        save_multiple_img(sr_img_np, name_list, sr_path)
            
            # 计算批量均值
            psnr_mean = np.mean(psnr_list)
            ssim_mean = ssim_total / len(dataloader_val)
            mae_mean = np.mean(mae_list)
            lpips_mean = lpips_total / len(dataloader_val)
            
            # 记录指标
            record_metrics['psnr'].append(psnr_mean)
            record_metrics['ssim'].append(ssim_mean)
            record_metrics['mae'].append(mae_mean)
            record_metrics['lpips'].append(lpips_mean)
            
            # 打印验证结果
            print(f" ===> Fold {fold} Epoch {epoch} Validation Results:")
            print(f"      PSNR: {psnr_mean:.4f} dB | SSIM: {ssim_mean:.4f}")
            print(f"      MAE: {mae_mean:.4f} | LPIPS: {lpips_mean:.4f}")
            
            # 保存指标日志
            with open(metric_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, psnr_mean, ssim_mean, mae_mean, lpips_mean])
            
            # 保存最佳模型（基于不同指标）
            if psnr_mean > best_metrics["psnr"]:
                best_metrics["psnr"] = psnr_mean
                save_model(generator, file_path, suf=f'psnr_best_epoch{epoch}')
                if gan:
                    save_model(discriminator, file_path, suf=f'dis_psnr_best_epoch{epoch}')
            
            if ssim_mean > best_metrics["ssim"]:
                best_metrics["ssim"] = ssim_mean
                save_model(generator, file_path, suf=f'ssim_best_epoch{epoch}')
                if gan:
                    save_model(discriminator, file_path, suf=f'dis_ssim_best_epoch{epoch}')
            
            if mae_mean < best_metrics["mae"]:
                best_metrics["mae"] = mae_mean
                save_model(generator, file_path, suf=f'mae_best_epoch{epoch}')
                if gan:
                    save_model(discriminator, file_path, suf=f'dis_mae_best_epoch{epoch}')
            
            if lpips_mean < best_metrics["lpips"]:
                best_metrics["lpips"] = lpips_mean
                save_model(generator, file_path, suf=f'lpips_best_epoch{epoch}')
                if gan:
                    save_model(discriminator, file_path, suf=f'dis_lpips_best_epoch{epoch}')
            
            # 绘制指标曲线（新增MAE和LPIPS）
            plot_metric(record_metrics['psnr'], 'Val PSNR (dB)', os.path.join(fold_run_home, 'val_psnr.pdf'))
            plot_metric(record_metrics['ssim'], 'Val SSIM', os.path.join(fold_run_home, 'val_ssim.pdf'))
            plot_metric(record_metrics['mae'], 'Val MAE', os.path.join(fold_run_home, 'val_mae.pdf'))
            plot_metric(record_metrics['lpips'], 'Val LPIPS', os.path.join(fold_run_home, 'val_lpips.pdf'))
        
        # 更新学习率
        scheduler_G.step()
        if gan:
            scheduler_D.step()
        
        # 记录训练损失
        cur_epoch_loss = epoch_loss / len(dataloader)
        record_metrics['loss'].append(cur_epoch_loss)
        plot_metric(record_metrics['loss'], 'Train Loss', os.path.join(fold_run_home, 'train_loss.pdf'))
        
        # 保存最新模型
        save_model(generator, file_path, suf=f"gen_last_epoch{epoch}")
        if gan:
            save_model(discriminator, file_path, suf=f'dis_last_epoch{epoch}')
        
        # 保存损失最小模型
        if cur_epoch_loss < loss_min:
            loss_min = cur_epoch_loss
            save_model(generator, file_path, suf="gen_loss_min")
            if gan:
                save_model(discriminator, file_path, suf='dis_loss_min')
        
        print(f" ===> Fold {fold} Epoch {epoch} Complete: Avg. Loss: {cur_epoch_loss:.6f}")
        
        # 保存损失日志
        with open(loss_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, cur_epoch_loss])
    
    # 返回当前折的最佳指标
    return (best_metrics["psnr"], best_metrics["ssim"], 
            best_metrics["mae"], best_metrics["lpips"], loss_min)


# 执行五折交叉验证
fold_results = []
for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
    print(f"\n===== Starting Fold {fold + 1}/{num_folds} =====")
    best_psnr, best_ssim, best_mae, best_lpips, best_loss = train_fold(fold + 1, train_idx, val_idx)
    fold_results.append({
        'fold': fold + 1,
        'best_psnr': best_psnr,
        'best_ssim': best_ssim,
        'best_mae': best_mae,
        'best_lpips': best_lpips,
        'best_loss': best_loss
    })

# 汇总五折结果
print("\n===== 5-Fold Cross Validation Final Results =====")
avg_psnr = 0.0
avg_ssim = 0.0
avg_mae = 0.0
avg_lpips = 0.0
avg_loss = 0.0

for result in fold_results:
    print(f"\nFold {result['fold']} Best Metrics:")
    print(f"  PSNR: {result['best_psnr']:.4f} dB")
    print(f"  SSIM: {result['best_ssim']:.4f}")
    print(f"  MAE: {result['best_mae']:.4f}")
    print(f"  LPIPS: {result['best_lpips']:.4f}")
    print(f"  Loss: {result['best_loss']:.6f}")
    
    avg_psnr += result['best_psnr']
    avg_ssim += result['best_ssim']
    avg_mae += result['best_mae']
    avg_lpips += result['best_lpips']
    avg_loss += result['best_loss']

# 计算平均值
avg_psnr /= num_folds
avg_ssim /= num_folds
avg_mae /= num_folds
avg_lpips /= num_folds
avg_loss /= num_folds

print("\n===== Average Results Across 5 Folds =====")
print(f"Average Best PSNR: {avg_psnr:.4f} dB")
print(f"Average Best SSIM: {avg_ssim:.4f}")
print(f"Average Best MAE: {avg_mae:.4f}")
print(f"Average Best LPIPS: {avg_lpips:.4f}")
print(f"Average Best Loss: {avg_loss:.6f}")

# 保存交叉验证汇总结果（包含所有指标）
summary_path = os.path.join(home_path, "cv_summary_with_all_metrics.csv")
with open(summary_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Fold", "Best PSNR (dB)", "Best SSIM", "Best MAE", "Best LPIPS", "Best Loss"])
    for result in fold_results:
        writer.writerow([
            result['fold'], 
            round(result['best_psnr'], 4),
            round(result['best_ssim'], 4),
            round(result['best_mae'], 4),
            round(result['best_lpips'], 4),
            round(result['best_loss'], 6)
        ])
    writer.writerow([
        "Average", 
        round(avg_psnr, 4),
        round(avg_ssim, 4),
        round(avg_mae, 4),
        round(avg_lpips, 4),
        round(avg_loss, 6)
    ])

print(f"\nSummary saved to: {summary_path}")
