"""
模型性能测试

Author: 罗涛
Date: 2024-1-6
"""

import io
import os
import shutil
import imagej
import numpy as np
import torch
from PIL import Image
from numpy import pad
from skimage import io, img_as_float, img_as_ubyte
from skimage.util import view_as_windows
from torchvision.transforms import Resize
from blur_kernel_extract.blur_kernel_estimator_net import KernelWizard
from model import Generator


# ImageJ处理拼接的临时文件夹
srTempPath = "/mnt/sda/lt/new_dataset/incoherent_RGBchannels/test/output2"

tempPatchPath = os.path.join(srTempPath, "temp_patch")
tempChannel = os.path.join(srTempPath, "temp_channel")
tile_config_path = os.path.join(tempPatchPath, "TileConfiguration.txt")
print('loading ImageJ......')
# ImageJ的主文件夹
fijiPath = '/home/lt/python/image_SR/fiji/Fiji.app'
ij = imagej.init(fijiPath)


def clear_cache():
    shutil.rmtree(srTempPath)
    os.makedirs(srTempPath, exist_ok=True)
    os.makedirs(os.path.join(srTempPath, "lr"), exist_ok=True)
    os.makedirs(os.path.join(srTempPath, "hr"), exist_ok=True)
    os.makedirs(os.path.join(srTempPath, "sr"), exist_ok=True)
    os.makedirs(os.path.join(srTempPath, "temp_patch"), exist_ok=True)
    os.makedirs(os.path.join(srTempPath, "temp_patch_low"), exist_ok=True)
    os.makedirs(os.path.join(srTempPath, "temp_patch_target"), exist_ok=True)
    os.makedirs(os.path.join(srTempPath, "temp_channel"), exist_ok=True)


def sr(imgPath, generator, kernel_model, sr_path, stitching=True, device=torch.device('cuda:0')):
    step = 192
    patch_size = 224

    clear_cache()
    img = Image.open(imgPath)
    print(img.size)
    img_array = img_as_float(np.array(img))
    pad_h = int((np.floor(img_array.shape[0] / step) * step + patch_size) - img_array.shape[0])
    pad_w = int((np.floor(img_array.shape[1] / step) * step + patch_size) - img_array.shape[1])
    img_array_padded = pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    img_wd = view_as_windows(img_array_padded, (patch_size, patch_size, 3), step=step)
    img_wd = np.squeeze(img_wd)

    with open(tile_config_path, 'w') as text_file:
        print('dim = {}'.format(2), file=text_file)
        with torch.no_grad():
            generator.eval()
            kernel_model.eval()
            for i in range(0, img_wd.shape[1]):
                for j in range(0, img_wd.shape[0]):
                    # low = img_wd[j, i]
                    # patch_lr_path = os.path.join(srTempPath, "hr", "s700_{}_{}.jpg").format(j, i)
                    # io.imsave(patch_lr_path, img_as_ubyte(low))

                    patch = img_wd[j, i].transpose((2, 0, 1))[None, :]
                    patch_tensor = torch.from_numpy(patch).float().to(device)
                    resize = Resize((256, 256))
                    k, _ = kernel_model(resize(patch_tensor))
                    prediction = generator(patch_tensor, k)
                    patch_path = os.path.join(tempPatchPath, "{}_{}.tiff").format(j, i)
                    io.imsave(patch_path, img_as_ubyte(np.clip(prediction.cpu().numpy()[0], 0, 1)))
                    print('{}_{}.tiff; ; ({}, {})'.format(j, i, i * step, j * step), file=text_file)

    if stitching:
        # 这里调用了软件imageJ 里面的拼接插件
        params = {'type': 'Positions from file', 'order': 'Defined by TileConfiguration',
                  'directory': tempPatchPath,
                  'layout_file': 'TileConfiguration.txt',
                  'fusion_method': 'Linear Blending', 'regression_threshold': '0.30',
                  'max/avg_displacement_threshold': '2.50', 'absolute_displacement_threshold': '3.50',
                  'compute_overlap': False, 'computation_parameters': 'Save computation time (but use more RAM)',
                  'image_output': 'Write to disk',
                  'output_directory': tempChannel}
        plugin = "Grid/Collection stitching"
        ij.py.run_plugin(plugin, params)
        c1 = io.imread(os.path.join(tempChannel, 'img_t1_z1_c1'))
        c2 = io.imread(os.path.join(tempChannel, 'img_t1_z1_c2'))
        c3 = io.imread(os.path.join(tempChannel, 'img_t1_z1_c3'))
        c1 = c1.astype(float) / 255.0
        c2 = c2.astype(float) / 255.0
        c3 = c3.astype(float) / 255.0
        c1 = c1[:img.size[1], :img.size[0]]
        c2 = c2[:img.size[1], :img.size[0]]
        c3 = c3[:img.size[1], :img.size[0]]
        img_to_save = np.clip(np.stack((c1, c2, c3)).transpose((1, 2, 0)), 0, 1)
        io.imsave(os.path.join(sr_path, os.path.basename(imgPath)), img_as_ubyte(img_to_save))


# 加载模糊核提取模型 和 图像恢复模型
device = torch.device('cuda:0')
model = KernelWizard()
model.load_state_dict(
    torch.load("/home/lt/python/eventually/blur_kernel_extract/result_new/{}/model_best.pth".format("LR+5_10")))
model = model.to(device)

generator = Generator(base_channel=32, middle_blk_num=12)
generator.load_state_dict(
    torch.load(r"/home/lt/python/eventually/result_new_ssim/LR+5_10/2025-05-11--10-13-45/epoch100-model_psnr_best.pth"))
generator.to(device)


# 恢复图像的保存文件夹
srOutput = r"/mnt/sda/lt/new_dataset/test_dataset/diff/LR+5_10/SR"
# 测试模糊图的文件夹
folder_path = r"/mnt/sda/lt/new_dataset/test_dataset/diff/LR+5_10/LR"

# 遍历当前目录下的文件和子目录名（不递归）
for item in os.listdir(folder_path):
    img_path = os.path.join(folder_path, item)
    sr(img_path, generator, model, srOutput, device=device)



# s16_l6/defocus3700
# s17_l4/defocus-5150
# s101_l1/defocus6550
# testRawData_incoherent_diffProtocol/s204_l5/defocus5600.jpg
# testRawData_incoherent_diffProtocol/s209_l4/defocus-7000.jpg
