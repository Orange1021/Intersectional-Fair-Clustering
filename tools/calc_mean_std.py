import os
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='统计图片数据集的mean和std')
parser.add_argument('--img_dir', type=str, required=True, help='图片文件夹路径')
parser.add_argument('--gray', action='store_true', help='是否为灰度图像')
args = parser.parse_args()

img_dir = args.img_dir
img_list = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
means = []
stds = []

for img_name in img_list:
    img_path = os.path.join(img_dir, img_name)
    img = Image.open(img_path)
    if args.gray:
        img = img.convert('L')
        img = np.array(img) / 255.0
        means.append(np.mean(img))
        stds.append(np.std(img))
    else:
        img = img.convert('RGB')
        img = np.array(img) / 255.0
        means.append(np.mean(img, axis=(0,1)))
        stds.append(np.std(img, axis=(0,1)))

means = np.array(means)
stds = np.array(stds)
if args.gray:
    mean = np.mean(means)
    std = np.mean(stds)
    print(f'Gray mean: {mean:.6f}, std: {std:.6f}')
else:
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)
    print(f'RGB mean: {mean}, std: {std}') 