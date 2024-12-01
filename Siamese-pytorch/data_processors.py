'''
处理扩充数据的各种方法
'''
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np

from typing import Any, Callable, Dict
from functools import wraps


class ProcessorRegistry:
    """数据处理器注册中心"""
    _processors: Dict[str, Callable] = {}

    @classmethod
    def register(cls, key: str = None):
        """注册处理器的装饰器"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            # 如果没有指定key，使用函数名作为key
            nonlocal key
            if key is None:
                key = func.__name__

            cls._processors[key] = wrapper
            return wrapper
        return decorator

    @classmethod
    def get_processor(cls, key: str) -> Callable:
        """获取处理器"""
        processor = cls._processors.get(key)
        if processor is None:
            raise KeyError(f"No processor registered for key: {key}")
        return processor

    @classmethod
    def list_processors(cls) -> list:
        """列出所有已注册的处理器"""
        return list(cls._processors.keys())


# 清晰度
@ProcessorRegistry.register("qingxidu")
def expand_qingxidu(img_path: str, param: int):
    image = Image.open(img_path)
    gaussian_blur_img = image.filter(ImageFilter.GaussianBlur(radius=param))
    return gaussian_blur_img


# 白平衡
@ProcessorRegistry.register("baipingheng")
def expand_baipingheng(img_path: str, param: int):
    """
    调整图像色温
    :param image: 输入图像 (BGR)
    :param temperature: 色温变化值 (正值提高冷色，负值增强暖色)
    :return: 调整后的图像
    """
    image = cv2.imread(img_path)
    if param > 0:
        increase_matrix = np.array([1, 1, 1 + param / 100], dtype=float)
    else:
        increase_matrix = np.array([1 + abs(param) / 100, 1, 1], dtype=float)

    # 分别调整 BGR 通道
    b, g, r = cv2.split(image)
    b = np.clip(b * increase_matrix[0], 0, 255).astype(np.uint8)
    g = np.clip(g * increase_matrix[1], 0, 255).astype(np.uint8)
    r = np.clip(r * increase_matrix[2], 0, 255).astype(np.uint8)

    # 转为Image格式
    rgb_image = cv2.merge([r, g, b])
    return Image.fromarray(rgb_image)


# 灰阶
@ProcessorRegistry.register("huijie")
def expand_huijie(img_path: str, param: int):
    image = cv2.imread(img_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    adjusted_v = cv2.convertScaleAbs(v, alpha=param, beta=0)
    hsv_adjusted = cv2.merge([h, s, adjusted_v])

    # 转换回 RGB 空间
    adjusted_image = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2RGB)
    return Image.fromarray(adjusted_image)


# 彩色饱和度
@ProcessorRegistry.register("caisebaohedu")
def expand_caisebaohedu(img_path: str, param: int):
    image = cv2.imread(img_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 调整饱和度
    s = np.clip(s * param, 0, 255).astype(np.uint8)

    hsv_adjusted = cv2.merge([h, s, v])
    adjusted_image = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2RGB)
    return Image.fromarray(adjusted_image)


# 彩色准确性
@ProcessorRegistry.register("caisezhunquexing")
def expand_caisezhunquexing(img_path: str, param: int):
    image = cv2.imread(img_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # 调整色相
    h = (h + param) % 180
    hsv = cv2.merge([h, s, v])
    adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(adjusted_image)


# 对比度
@ProcessorRegistry.register("duibidu")
def expand_duibidu(img_path: str, param: int):
    image = Image.open(img_path)
    enhancer = ImageEnhance.Contrast(image)
    adjusted_image = enhancer.enhance(param)
    return adjusted_image


# 区域控光
@ProcessorRegistry.register("quyukongguang")
def expand_quyukongguang(img_path: str, param: int):
    image = cv2.imread(img_path)
    adjusted = cv2.convertScaleAbs(image, alpha=1, beta=param)
    adjusted_image = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
    return Image.fromarray(adjusted_image)
