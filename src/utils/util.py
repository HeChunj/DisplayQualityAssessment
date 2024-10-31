import time
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor

import cv2


def crop_image(image_path, centers, window_size, model_name=""):
    """
    image_path: 图片路径
    centers: 一系列中心坐标 (x, y) 的列表
    window_size: 切割图块的大小（正方形的边长）
    """
    # 检查图像路径是否存在
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    height, width, _ = image.shape

    half_window = window_size // 2  # 窗口的一半

    # 记录开始的时间
    start_time = time.time()

    # 记录裁剪的图块的地址
    cropped_image_paths = []

    # 获取当前工作目录
    current_directory = os.getcwd()

    # 提取文件名（不带扩展名）
    filename_without_ext = os.path.splitext(os.path.basename(image_path))[0]

    result_dir = f'{current_directory}/croped_result/{model_name}/{filename_without_ext}'

    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 使用多线程处理裁剪和保存
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for i, (x, y) in enumerate(centers):
            futures.append(
                executor.submit(crop_and_save, image_path, image, x, y,
                                half_window, width, height, i, result_dir)
            )

        # 获取结果
        for future in futures:
            cropped_image_paths.append(future.result())

    # 记录结束的时间
    end_time = time.time()
    print(f"Time elapsed, crop img: {end_time - start_time} seconds")

    return cropped_image_paths


def crop_and_save(image_path, image, x, y, half_window, width, height, i, result_dir):

    image = cv2.imread(image_path)

    # 确定裁剪区域的左上角和右下角
    left = max(0, x - half_window)
    top = max(0, y - half_window)
    right = min(width, x + half_window)
    bottom = min(height, y + half_window)

    left = round(left)
    top = round(top)
    right = round(right)
    bottom = round(bottom)

    # 裁剪图块
    cropped_image = image[top:bottom, left:right]

    # 保存裁剪的图块到指定目录
    save_path = os.path.join(result_dir, f"cropped_image_{i}.png")
    cv2.imwrite(save_path, cropped_image)

    return save_path
