"""
计算 psnr 和 ssim 的平均值和标准差
来源于RAT https://github.com/RAT
Author: 罗涛
Date: 2024-12-16
"""

from utils import calculate_psnr, calculate_ssim
import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


test_y_channel = True

sr_path = r"C:\Users\11139\Desktop\fsdownload\SR"
hr_path = r"C:\Users\11139\Desktop\fsdownload\result_old\diff\LR+5_10\HR"

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

result['psnr_mean'] = float(np.mean(result["psnr"]))
result['psnr_std'] = float(np.std(result["psnr"]))
result['ssim_mean'] = float(np.mean(result["ssim"]))
result['ssim_std'] = float(np.std(result["ssim"]))

df = pd.DataFrame({key: pd.Series(value) for key, value in result.items()})
if test_y_channel:
    df.to_csv("new-diff-test2-LR+5_10.csv")
else:
    df.to_csv("result-color.csv")
