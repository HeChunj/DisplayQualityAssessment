import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import csv

from .utils import cvtColor, preprocess_input
from .utils_aug import CenterCrop, ImageNetPolicy, RandomResizedCrop, Resize


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


class KADID10kDataset(Dataset):
    def __init__(self, input_shape, image_folder, dmos_file, random, autoaugment_flag=True):
        self.input_shape = input_shape
        self.image_folder = image_folder
        self.dmos_file = dmos_file

        self.random = random

        self.autoaugment_flag = autoaugment_flag
        if self.autoaugment_flag:
            self.resize_crop = RandomResizedCrop(input_shape)
            self.policy = ImageNetPolicy()

            self.resize = Resize(
                input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
            self.center_crop = CenterCrop(input_shape)

        # 读取 dmos.csv 标签文件
        self.dmos_data = []
        with open(dmos_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过标题行
            for row in reader:
                dist_img = row[0]   # 失真图像名称
                ref_img = row[1]  # 原始图像名称
                dmos = np.float32(row[2])  # DMOS 分数
                self.dmos_data.append((dist_img, ref_img, dmos))

    def __len__(self):
        return len(self.dmos_data)

    def __getitem__(self, index):
        """
        根据索引返回图像和相应的 DMOS 分数
        :param idx: 索引
        :return: 图像路径和 DMOS 标签
        """
        dist_img, ref_img, dmos = self.dmos_data[index]
        dist_img = os.path.join(self.image_folder, dist_img)
        ref_img = os.path.join(self.image_folder, ref_img)

        # 读取失真图像
        image1 = Image.open(dist_img)
        image1 = cvtColor(image1)
        if self.autoaugment_flag:
            image1 = self.AutoAugment(image1, random=self.random)
        else:
            image1 = self.get_random_data(
                image1, self.input_shape, random=self.random)
        image1 = preprocess_input(np.array(image1).astype(np.float32))
        image1 = np.transpose(image1, [2, 0, 1])

        # 读取参考图像
        image2 = Image.open(ref_img)
        image2 = cvtColor(image2)
        if self.autoaugment_flag:
            image2 = self.AutoAugment(image2, random=self.random)
        else:
            image2 = self.get_random_data(
                image2, self.input_shape, random=self.random)
        image2 = preprocess_input(np.array(image2).astype(np.float32))
        image2 = np.transpose(image2, [2, 0, 1])

        return image1, image2, dmos

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            # ---------------------------------#
            #   将图像多余的部分加上灰条
            # ---------------------------------#
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            return image_data

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter, 1+jitter) / \
            self.rand(1-jitter, 1+jitter)
        scale = self.rand(0.75, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        rotate = self.rand() < .5
        if rotate:
            angle = np.random.randint(-15, 15)
            a, b = w/2, h/2
            M = cv2.getRotationMatrix2D((a, b), angle, 1)
            image = cv2.warpAffine(
                np.array(image), M, (w, h), borderValue=[128, 128, 128])

        image_data = np.array(image, np.uint8)
        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge(
            (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data

    def AutoAugment(self, image, random=True):
        if not random:
            image = self.resize(image)
            image = self.center_crop(image)
            return image

        # ------------------------------------------#
        #   resize并且随即裁剪
        # ------------------------------------------#
        image = self.resize_crop(image)

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # ------------------------------------------#
        #   随机增强
        # ------------------------------------------#
        image = self.policy(image)
        return image

# DataLoader中collate_fn使用


def dataset_collate(batch):
    left_images = []
    right_images = []
    labels = []
    for img1, img2, pair_labels in batch:
        left_images.append(img1)
        right_images.append(img2)
        labels.append(pair_labels)

    images = torch.from_numpy(
        np.array([left_images, right_images])).type(torch.FloatTensor)
    labels = torch.from_numpy(np.array(labels)).type(torch.FloatTensor)
    return images, labels


if __name__ == "__main__":
    # 测试 DataLoader 是否正常工作
    image_folder = "/data/hechunjiang/KADID-10k/kadid10k/images"  # 图像文件夹路径
    dmos_file = "/data/hechunjiang/KADID-10k/kadid10k/dmos.csv"     # dmos.csv 文件路径
    batch_size = 32
    input_shape = [256, 256]
    # 创建数据集
    dataset = KADID10kDataset(input_shape=input_shape,
                              image_folder=image_folder, dmos_file=dmos_file, random=True, autoaugment_flag=True)

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    print(len(dataset))
    cnt = 0
    for iteration, batch in enumerate(dataloader):
    # for imgs, dmos_scores in dataloader:
        print(f"Batch of dist images: {batch[0].shape}")
        print(f"Batch of ref images: {batch[1].shape}")
        print(f"Batch of DMOS scores: {batch[2].shape}")
        break  # 仅加载一个批次进行测试
    # print(cnt)
