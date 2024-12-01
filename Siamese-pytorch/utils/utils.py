import json
import math
import os
import random
from functools import partial
from random import shuffle
import time

import cv2
import numpy as np
from PIL import Image

from .utils_aug import center_crop, resize

import matplotlib.pyplot as plt
from PIL import ImageDraw


def load_dataset(dataset_path, train_own_data, train_ratio):
    types = 0
    train_path = os.path.join(dataset_path, 'images_background')
    lines = []
    labels = []

    if train_own_data:
        # -------------------------------------------------------------#
        #   自己的数据集，遍历大循环
        # -------------------------------------------------------------#
        for character in os.listdir(train_path):
            # -------------------------------------------------------------#
            #   对每张图片进行遍历
            # -------------------------------------------------------------#
            character_path = os.path.join(train_path, character)
            for image in os.listdir(character_path):
                lines.append(os.path.join(character_path, image))
                labels.append(types)
            types += 1
    else:
        # -------------------------------------------------------------#
        #   Omniglot数据集，遍历大循环
        # -------------------------------------------------------------#
        for alphabet in os.listdir(train_path):
            alphabet_path = os.path.join(train_path, alphabet)
            # -------------------------------------------------------------#
            #   Omniglot数据集，遍历小循环
            # -------------------------------------------------------------#
            for character in os.listdir(alphabet_path):
                character_path = os.path.join(alphabet_path, character)
                # -------------------------------------------------------------#
                #   对每张图片进行遍历
                # -------------------------------------------------------------#
                for image in os.listdir(character_path):
                    lines.append(os.path.join(character_path, image))
                    labels.append(types)
                types += 1

    # -------------------------------------------------------------#
    #   将获得的所有图像进行打乱。
    # -------------------------------------------------------------#
    random.seed(1)
    shuffle_index = np.arange(len(lines), dtype=np.int32)
    shuffle(shuffle_index)
    random.seed(None)
    lines = np.array(lines, dtype=np.object)
    labels = np.array(labels)
    lines = lines[shuffle_index]
    labels = labels[shuffle_index]

    # -------------------------------------------------------------#
    #   将训练集和验证集进行划分
    # -------------------------------------------------------------#
    num_train = int(len(lines)*train_ratio)

    val_lines = lines[num_train:]
    val_labels = labels[num_train:]

    train_lines = lines[:num_train]
    train_labels = labels[:num_train]
    return train_lines, train_labels, val_lines, val_labels


# ---------------------------------------------------#
#   对输入图像进行resize
# ---------------------------------------------------#
def letterbox_image(image, size, letterbox_image):
    w, h = size
    iw, ih = image.size
    if letterbox_image:
        '''resize image with unchanged aspect ratio using padding'''
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        if h == w:
            new_image = resize(image, h)
        else:
            new_image = resize(image, [h, w])
        new_image = center_crop(new_image, [h, w])
    return new_image


# ---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
# ---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


# ----------------------------------------#
#   预处理训练图片
# ----------------------------------------#
def preprocess_input(x):
    x /= 255.0
    return x


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1, no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
                                              ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters,
                       warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url

    download_urls = {
        'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    }
    url = download_urls[backbone]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)


def calculate_iou(box1, box2):
    """
    计算红色框在绿色框中的占比
    box1是红色框, box2是绿色框
    Parameters
    """

    x_left = max(box1['left'], box2['x1'])
    y_top = max(box1['top'], box2['y1'])
    x_right = min(box1['right'], box2['x2'])
    y_bottom = min(box1['bottom'], box2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left + 1.0) * (y_bottom - y_top + 1.0)

    box1_area = (box1['right'] - box1['left'] + 1.0) * \
        (box1['bottom'] - box1['top'] + 1.0)
    box2_area = (box2['x2'] - box2['x1'] + 1.0) * \
        (box2['y2'] - box2['y1'] + 1.0)
    # return intersection_area / (box1_area + box2_area - intersection_area)

    # 计算红色框和绿色框相交面积占红色框的比例
    iou = intersection_area / box1_area
    return iou


def map_model_to_opinion(b_value, b_min=-4, b_max=4, a_min=10, a_max=90):
    # model to opinion
    if type(b_value) != list:
        return round(a_min + (b_value - b_min) / (b_max - b_min) * (a_max - a_min), 2)
    return [round(a_min + (b - b_min) / (b_max - b_min) * (a_max - a_min), 2) for b in b_value]


def map_opinion_to_model(a_value, a_min=10, a_max=90, b_min=-4, b_max=4):
    # opinion to model
    if type(a_value) != list:
        return round(b_min + (a_value - a_min) / (a_max - a_min) * (b_max - b_min), 2)
    return [round(b_min + (a - a_min) / (a_max - a_min) * (b_max - b_min), 2) for a in a_value]


# 绘制直方图
def pltHistogram(data):
    plt.figure(figsize=(8, 5))
    unique_values = np.unique(data)
    # 计算频率的平均值
    n, bins, patches = plt.hist(
        data, bins=20, color='skyblue', edgecolor='black')
    mean_frequency = len(data) / len(unique_values)
    # 绘制水平参考线表示平均值
    plt.axhline(mean_frequency, color='red', linestyle='--',
                linewidth=1.5, label=f'Mean Frequency: {mean_frequency:.2f}')
    plt.text(bins[len(bins) // 2], mean_frequency + 1,
             f'Avg: {mean_frequency:.2f}', color='red', ha='center')
    plt.title('Value Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()


# 将label box绘制到图片上
def show_box(img1, coordinates_ref, label_ref):
    image_1 = Image.open(img1)
    draw = ImageDraw.Draw(image_1)
    for coordinate in coordinates_ref:
        draw.rectangle([coordinate['left'], coordinate['top'],
                       coordinate['right'], coordinate['bottom']], outline="red")
    # 将label_ref中的框绘制到image_1上
    for label in label_ref:
        draw.rectangle([label['x1'], label['y1'], label['x2'],
                       label['y2']], outline="green", width=4)
    image_1.show()


# 根据坐标裁剪图块
def crop_image(image_path, coordinates_path, j, img_idx, demo_name="", result_dir=None):

    # 检查图像路径是否存在
    if not os.path.exists(image_path):
        raise ValueError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)

    # 检查坐标文件是否存在
    if not os.path.exists(coordinates_path):
        raise ValueError(f"Coordinates file not found: {coordinates_path}")
    with open(coordinates_path, 'r') as json_file:
        coordinates = json.load(json_file)

    # 记录开始的时间
    start_time = time.time()

    # 记录裁剪的图块的地址
    cropped_image_paths = []

    if result_dir is None:
        result_dir = f'/home/hechunjiang/gradio/GeoFormer/finetune_data/{demo_name}/{img_idx}/param_{j}/'

    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for i, data in enumerate(coordinates['coordinates']):
        # 根据坐标裁剪图块
        left = data['left']
        top = data['top']
        right = data['right']
        bottom = data['bottom']

        left = round(left)
        top = round(top)
        right = round(right)
        bottom = round(bottom)

        cropped_image = image[top:bottom, left:right]
        cropped_image_save_path = os.path.join(
            result_dir, f"cropped_image_{i}.png")

        cv2.imwrite(cropped_image_save_path, cropped_image)
        cropped_image_paths.append(cropped_image_save_path)

    # 记录结束的时间
    end_time = time.time()

    print(
        f"Processing image: {image_path}, time: {end_time - start_time} seconds")

    return cropped_image_paths
