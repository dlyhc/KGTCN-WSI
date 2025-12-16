"""
这个文件 用于计算推理时间

Author: 罗涛
Date: 2025-2-8
"""

import torch
import numpy as np
from PIL import Image
from skimage import img_as_float

from model import Generator

model = Generator(base_channel=32)
device = torch.device('cuda:0')
model.to(device)

model.load_state_dict(
            torch.load(r"/home/lt/python/eventually/result/LR-0_5/checkpoint/model_gen_best.pth"))

model.eval().cuda()
img = Image.open(r"/mnt/sda/lt/real/RAT/LR-0_5/test/LR/cropped_0_0_118.jpg")
img_hr_array = img_as_float(np.array(img)).transpose((2, 0, 1))
img_hr_array = np.expand_dims(img_hr_array, axis=0)
# 输入torch.randn(1, 3, 256, 256).cuda()
dummy_input = torch.from_numpy(img_hr_array).float().cuda()
# 载入模糊核信息
kernel = np.load(r"/mnt/sda/lt/real/RAT/LR-0_5/test/LR_kernel/cropped_0_0_118.npy")
kernel = torch.from_numpy(kernel).float()
# 加入batch_size维度
kernel = kernel.squeeze(0)

# 预热
for _ in range(100):
    _ = model(dummy_input, kernel)

# 时间测量
repetitions = 300
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
timings = np.zeros((repetitions, 1))

with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input,kernel)
        ender.record()
        torch.cuda.synchronize()
        timings[rep] = starter.elapsed_time(ender)

mean_time = np.mean(timings)
std_time = np.std(timings)
fps = 1000 / mean_time

print(f"Mean Inference Time: {mean_time:.3f}ms ± {std_time:.3f}ms")
print(f"FPS: {fps:.2f}")