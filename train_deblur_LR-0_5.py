import csv
import math
import os
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import sys
from datetime import datetime

import numpy as np
import torch
import matplotlib

from pytorch_msssim import MS_SSIM
from torch.autograd import Variable

from loss import PerceptualLoss

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Generator, Discriminator
from data_loader import PairedData
from utils import tensor2img, save_img, calculate_ssim


def mkdir(p):
    isExists = os.path.exists(p)
    if isExists:
        pass
    else:
        os.makedirs(p)
        print("make directory successfully:{}".format(p))


def cal_psnr_numpy(img1, img2, data_range=255):
    B, H, W, C = img1.shape
    mse = (img1 - img2) ** 2
    mse = np.mean(mse.reshape(B, -1), axis=1)
    return list(10 * np.log10(data_range ** 2 / mse))

def calculate_ssim_batch(SRs, HRs):
    # SRs 和 HRs 是 [B, H, W, C]
    batch_ssim = 0.0
    for i in range(SRs.shape[0]):
        SR = SRs[i]
        HR = HRs[i]
        batch_ssim += calculate_ssim(SR, HR, crop_border=0, input_order='HWC', test_y_channel=False)
    return batch_ssim / SRs.shape[0]


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
    save_path = os.path.join(save_path, 'model_{}.pth'.format(suf))
    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()


def print_fixed_log(epoch, num_epochs, iteration, len_dataloader, loss_D, loss_pixel, loss_GAN, loss_G):
    message = (
            '[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Identity/Advers/Total): %.4f/%.4f/%.4f'
            % (epoch, num_epochs, iteration, len_dataloader, loss_D, loss_pixel, loss_GAN, loss_G)
    )
    sys.stdout.write("\r" + message)  # 使用 \r 回到当前行首
    sys.stdout.flush()  # 刷新输出缓冲区


def print_fixed_log_without_gan(epoch, num_epochs, iteration, len_dataloader, loss_pixel, loss_G):
    message = (
            '[%d/%d][%d/%d] Generator_Loss (Identity/Total): %.4f/%.4f'
            % (epoch, num_epochs, iteration, len_dataloader, loss_pixel, loss_G)
    )
    sys.stdout.write("\r" + message)  # 使用 \r 回到当前行首
    sys.stdout.flush()  # 刷新输出缓冲区


def plot_psnr(data, label, save_path):
    l = len(data)
    axis = np.linspace(1, l, l)

    fig = plt.figure()
    plt.title(label)

    plt.plot(
        axis,
        data,
        label=label
    )
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close(fig)


# 超参数设置
run_from = None
type = "LR-0_5"
home_path = "/home/lt/python/eventually/result_new_ssim/{}".format(type)
g_lr = 0.0001  # 学习率
d_lr = 0.00001
eval_interval = 1  # 每多少个epoch验证一次
start_epoch = 1
num_epochs = 300  # 训练epoch数
l = 0.1  # 对抗损失占比
p = 0.1  # 感知损失占比
gan = False

patch_size = 224
batch_size = 16
middle_block_num = 12
base_channel=32

dataset_path = "/mnt/sda/lt/new_dataset/{}_newdeblur".format(type)

# warnings.filterwarnings('ignore')
torch.cuda.set_device(1)
device = torch.device('cuda:1')
tensor = torch.cuda.FloatTensor
run = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
patch = (1, patch_size // 2 ** 4, patch_size // 2 ** 4)
set_seeds()

record_metrics = {
    "psnr": [],
    "loss": [],
    "ssim": []
}

# 创建CSV文件并写入表头
loss_csv_path = os.path.join(home_path, run, "loss_log.csv")
psnr_csv_path = os.path.join(home_path, run, "psnr_log.csv")
mkdir(os.path.join(home_path, run))

file_path = os.path.join(home_path, run, 'checkpoint')
mkdir(file_path)
sr_path = os.path.join(home_path, run, 'sr_img')
mkdir(sr_path)

# 初始化Loss CSV文件
with open(loss_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Loss"])

# 初始化PSNR CSV文件
with open(psnr_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "PSNR", "SSIM"])

# 模型
generator = Generator(base_channel=base_channel,middle_blk_num=middle_block_num)
generator.to(device)
if gan:
    discriminator = Discriminator()
    discriminator.to(device)

# total_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
# print(total_params)
# if torch.cuda.is_available():
#     generator.cuda()
#     discriminator.cuda()
# generator = nn.DataParallel(generator, device_ids=range(torch.cuda.device_count()))
# discriminator = nn.DataParallel(discriminator, device_ids=range(torch.cuda.device_count()))

# 创建数据集
dataset = PairedData(root=dataset_path, target='train')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

dataset_test = PairedData(root=dataset_path, target='test')
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

# 是否载入已有的参数
if run_from is not None:
    generator.load_state_dict(
        torch.load(os.path.join(home_path, run_from, 'checkpoint', 'model_gen_last.pth')))
    try:
        discriminator.load_state_dict(
            torch.load(os.path.join(home_path, run_from, 'checkpoint', 'model_dis_last.pth')))
    except:
        print('Discriminator weights not found!')
        pass

# 损失
criterionL = nn.L1Loss().to(device)
criterionSSIM = MS_SSIM(data_range=1, size_average=True, channel=3).to(device)
criterionMSE = nn.MSELoss().to(device)
# perceptualLoss = PerceptualLoss()

# 优化器设置
optimizer_G = torch.optim.Adam(generator.parameters(), lr=g_lr)
scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, num_epochs, g_lr * 0.05)

if gan:
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=d_lr)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, num_epochs, d_lr * 0.05)

psnr_max = 0
ssim_max = 0
loss_min = math.inf

print(">>>> Training....")
for epoch in range(start_epoch, num_epochs + 1):
    epoch_loss = 0
    gan_loss = 0
    total_loss = 0
    dis_loss = 0
    generator.train()
    if gan:
        discriminator.train()

    ######## Epoch 开始 ########
    epoch_begin_time = datetime.now()
    for iteration, (hr_img, lr_img, lr_kernel, filename) in enumerate(dataloader):
        real_mid = Variable(lr_img.type(tensor).to(device), requires_grad=False)
        real_high = Variable(hr_img.type(tensor).to(device), requires_grad=False)
        lr_kernel = Variable(lr_kernel.type(tensor).to(device), requires_grad=False)

        # Adversarial ground truths
        if gan:
            valid = Variable(tensor(np.ones((real_mid.size(0), *patch))).to(device), requires_grad=False)
            fake = Variable(tensor(np.zeros((real_mid.size(0), *patch))).to(device), requires_grad=False)

        # ---------------
        #  Train Generator
        # ---------------
        # GAN loss
        optimizer_G.zero_grad()  # 清空过往梯度

        fake_high = generator(real_mid, lr_kernel)

        # GAN_loss
        if gan:
            pred_fake = discriminator(fake_high, real_mid)
            loss_GAN = criterionMSE(pred_fake, valid)

        # L1损失
        lossL1 = criterionL(fake_high, real_high)

        # SSIM损失
        # ssim_loss = 1 - criterionSSIM(fake_high, real_high)

        # 感知损失
        # perceptual_loss = perceptualLoss.get_loss(fake_high, real_high)

        # Total loss
        loss_pixel = lossL1

        if gan:
            loss_G = l * loss_GAN + (1 - l) * loss_pixel
        else:
            loss_G = loss_pixel
        loss_G.backward()
        optimizer_G.step()

        total_loss = total_loss + loss_G.item()

        if gan:
            gan_loss = gan_loss + loss_GAN.item()

        # ---------------
        #  Train Discriminator
        # ---------------
        if gan:
            optimizer_D.zero_grad()
            # Real loss
            pred_real = discriminator(real_high, real_mid)
            loss_real = criterionMSE(pred_real, valid)
            # Fake loss
            pred_fake = discriminator(fake_high.detach(), real_mid)
            loss_fake = criterionMSE(pred_fake, fake)
            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()

            dis_loss = dis_loss + loss_D.item()
        epoch_loss = epoch_loss + loss_pixel.item()

        if gan:
            print_fixed_log(
                epoch, num_epochs, iteration, len(dataloader),
                loss_D.item(), loss_pixel.item(), loss_GAN.item(), loss_G.item()
            )
        else:
            print_fixed_log_without_gan(epoch, num_epochs, iteration, len(dataloader), loss_pixel.item(), loss_G.item())

    epoch_end_time = datetime.now()
    print(f"\n>>>one epoch time cost: {(epoch_end_time - epoch_begin_time).total_seconds()} seconds")
    ######## Epoch 结束 ########

    ######## Valid 开始 ########
    print(">>>> Valid....")
    if epoch % eval_interval == 0:
        psnr = []
        ssim = 0.0
        generator.eval()
        if gan:
            discriminator.eval()
        with torch.no_grad():
            for _, data in enumerate(tqdm(dataloader_test)):
                hr_img, lr_img, lr_kernel, name_list = data
                lr_img = lr_img.type(torch.FloatTensor).to(device)
                lr_kernel = lr_kernel.type(torch.FloatTensor).to(device)

                sr_img = generator(lr_img, lr_kernel)
                sr_img = sr_img.detach().cpu()
                lr_img = lr_img.detach().cpu()

                sr_img = tensor2img(sr_img)
                hr_img = tensor2img(hr_img)
                lr_img = tensor2img(lr_img)
                psnr += cal_psnr_numpy(sr_img, hr_img, data_range=255)
                ssim += calculate_ssim_batch(sr_img, hr_img)

                save_multiple_img(sr_img, name_list, sr_path)

        # 根据总的PSNR，计算平均PSNR
        psnr_mean = np.mean(psnr)
        ssim_mean = ssim / len(dataloader_test)

        record_metrics['psnr'].append(psnr_mean)
        record_metrics['ssim'].append(ssim_mean)

        print(" ===> Epoch {} Complete: Avg. psnr: {:.4f}, ssim: {:.4f}".format(epoch, psnr_mean, ssim_mean))
        # 保存PSNR到CSV文件
        with open(psnr_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, psnr_mean, ssim_mean])

        # 更新最好的PSNR，并保存PSNR表现最好的模型
        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            save_model(generator, file_path, suf='psnr_best')
            if gan:
                save_model(discriminator, file_path, suf='dis_psnr_best')

        # 更新最好的SSIM，并保存SSIM表现最好的模型
        if ssim_mean > ssim_max:
            ssim_max = ssim_mean
            save_model(generator, file_path, suf='ssim_best')
            if gan:
                save_model(discriminator, file_path, suf='dis_ssim_best')

        # 画图-PSNR/SSIM
        plot_psnr(record_metrics['psnr'], 'val psnr', os.path.join(home_path, run, 'val_psnr.pdf'))
        plot_psnr(record_metrics['ssim'], 'val ssim', os.path.join(home_path, run, 'val_ssim.pdf'))

        print(">>>> End valid")
    ######## Valid 结束 ########

    # 更新学习率
    scheduler_G.step()
    if gan:
        scheduler_D.step()

    # 画图-loss
    record_metrics['loss'].append(epoch_loss / len(dataloader))
    plot_psnr(record_metrics['loss'], 'train loss', os.path.join(home_path, run, 'train_loss.pdf'))

    # 保存最后一个模型参数
    save_model(generator, file_path, suf=str("gen_last"))
    if gan:
        save_model(discriminator, file_path, suf='dis_last')

    # 保存最小的loss模型参数
    cur_epoch_loss = epoch_loss / len(dataloader)
    if loss_min > cur_epoch_loss:
        loss_min = cur_epoch_loss
        save_model(generator, file_path, suf=str("gen_loss_min"))
        if gan:
            save_model(discriminator, file_path, suf='dis_loss_min')

    print(" ===> Epoch {} Complete: Avg. Loss: {:.6f}".format(epoch, cur_epoch_loss))

    # 保存Loss到CSV文件
    with open(loss_csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, cur_epoch_loss])
