import time
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt


def crop_image(image_path, centers, window_size, demo="", data_type=""):
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
    current_directory = "static"
    filename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
    result_dir = f'{current_directory}/croped_result_{demo}/{data_type}/{filename_without_ext}'

    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for i, (x, y) in enumerate(centers):
        # 裁剪图块
        cropped_image = crop_and_save(
            image, x, y, half_window, width, height, i, result_dir)
        cropped_image_paths.append(cropped_image)

    # 记录结束的时间
    end_time = time.time()
    # print(f"Time elapsed, crop img: {end_time - start_time} seconds")

    return cropped_image_paths


def crop_and_save(image, x, y, half_window, width, height, i, result_dir):

    # 确定裁剪区域的左上角和右下角
    left = max(0, x - half_window)
    top = max(0, y - half_window)
    right = min(width, x + half_window)
    bottom = min(height, y + half_window)

    left = round(left)
    top = round(top)
    right = round(right)
    bottom = round(bottom)

    # 确保裁剪区域在图像边界内
    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if right > width:
        right = width
    if bottom > height:
        bottom = height

    # 裁剪图块
    cropped_image = image[top:bottom, left:right]

    # 保存裁剪的图块的坐标信息到json文件
    json_path = os.path.join(result_dir, f"cropped_image_coordinate.json")
    res = {}
    res["coordinates"] = []
    res["coordinates"].append({
        "cx": x,
        "cy": y,
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom
    })
    save = {}
    save["coordinates"] = []

    if os.path.exists(json_path):
        # 删除已经存在的json文件
        # os.remove(json_path)
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
    else:
        data = {}
        data['coordinates'] = []
    save['coordinates'] = data['coordinates'] + res['coordinates']
    with open(json_path, 'w') as json_file:
        json.dump(save, json_file, indent=4)

    # 保存裁剪的图块到指定目录
    save_path = os.path.join(result_dir, f"cropped_image_{i}.png")
    cv2.imwrite(save_path, cropped_image)

    return save_path


def get_homography_res(img1_path, img2_path, kpts1, kpts2, matches, is_draw=False):
    # img1向img2对齐
    if len(matches) < 4:
        return None, None

    if isinstance(matches, list):
        matches = np.array(matches)

    if isinstance(kpts1, list):
        kpts1 = np.array(kpts1)

    if isinstance(kpts2, list):
        kpts2 = np.array(kpts2)

    img1_color = cv2.imread(img1_path)
    img2_color = cv2.imread(img2_path)

    src_pts = np.float32(matches[:, :2]).reshape(-1, 1, 2)
    dst_pts = np.float32(matches[:, 2:]).reshape(-1, 1, 2)

    matrix, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    height, width = img2_color.shape[:2]

    aligned_img1_color = cv2.warpPerspective(
        img1_color, matrix, (width, height))

    if is_draw:
        plt.figure(dpi=500)

        # 示例关键点和图像
        kp0 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kpts2]
        kp1 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kpts2]
        matches = [cv2.DMatch(_trainIdx=i, _queryIdx=i,
                              _distance=1, _imgIdx=-1) for i in range(len(kp0))]

        # 转换图像为 RGB 格式
        aligned_img1_rgb = cv2.cvtColor(aligned_img1_color, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB)

        # 创建拼接图像
        h1, w1 = aligned_img1_rgb.shape[:2]
        h2, w2 = img2_rgb.shape[:2]
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = aligned_img1_rgb
        canvas[:h2, w1:w1 + w2] = img2_rgb

        # 绘制匹配点和随机颜色连线
        for match in matches:
            pt1 = (int(kp0[match.queryIdx].pt[0]),
                   int(kp0[match.queryIdx].pt[1]))
            pt2 = (int(kp1[match.trainIdx].pt[0]) +
                   w1, int(kp1[match.trainIdx].pt[1]))
            random_color = tuple(np.random.randint(0, 256, 3).tolist())
            cv2.line(canvas, pt1, pt2, color=random_color, thickness=2)
            cv2.circle(canvas, pt1, radius=10,
                       color=random_color, thickness=-1)
            cv2.circle(canvas, pt2, radius=10,
                       color=random_color, thickness=-1)

        # 显示结果
        print("仿射变换后的匹配结果：")
        plt.imshow(canvas)
        plt.axis('off')
        plt.show()

    return cv2.cvtColor(aligned_img1_color, cv2.COLOR_BGR2RGB), matrix
