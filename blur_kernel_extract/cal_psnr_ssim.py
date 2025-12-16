from utils import calculate_psnr, calculate_ssim
import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import json


def save_json(data, path):
    with open(path, "w", encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, encoding='utf8') as f:
        data = json.load(f)
    return data


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


test_y_channel = True
# item = "LR-0_5"
# item = "LR+0_5"
# item = "LR-5_10"
item = "LR+5_10"
sr_path = r"/home/lt/python/eventually/blur_kernel_extract/result/{}/SR".format(item)
hr_path = r"/mnt/sda/lt/real/test/{}".format(item)
ychannel_path = "/home/lt/python/eventually/blur_kernel_extract/result/{}/result-ychannel-{}.csv".format(item,item)
color_path = "/home/lt/python/eventually/blur_kernel_extract/result/{}/result-color-{}.csv".format(item,item)

result = {
    'name': [],
    "psnr": [],
    "ssim": []
}

file_list = os.listdir(sr_path)
print("Data Size:", len(file_list))

for name in tqdm(file_list):
    SR = cv2.imread(os.path.join(sr_path, name), cv2.IMREAD_COLOR).astype(np.float32)
    HR = cv2.imread(os.path.join(hr_path, name), cv2.IMREAD_COLOR).astype(np.float32)
    result["name"].append(name)
    result["psnr"].append(calculate_psnr(SR, HR, crop_border=0, input_order='HWC', test_y_channel=test_y_channel))
    result["ssim"].append(calculate_ssim(SR, HR, crop_border=0, input_order='HWC', test_y_channel=test_y_channel))

result['psnr_mean'] = np.mean(result["psnr"])
result['psnr_std'] = np.std(result["psnr"])
result['ssim_mean'] = np.mean(result["ssim"])
result['ssim_std'] = np.std(result["ssim"])

df = pd.DataFrame({key: pd.Series(value) for key, value in result.items()})
if test_y_channel:
    df.to_csv(ychannel_path)
else:
    df.to_csv(color_path)
