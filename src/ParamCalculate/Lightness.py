# Lightness
import cv2
from Strategy.AbstractStrategy import AbstractStrategy
import numpy as np


class ParamLightness(AbstractStrategy):

    def __new__(cls, *args, **kwargs):
        for arg in args:
            print("arg: ", arg)
        for k, v in kwargs.items():
            print(f"key: {k}, value: {v}")
        if not hasattr(ParamLightness, "_instance"):
            print("ParamLightness __new__")
            ParamLightness._instance = super(ParamLightness, cls).__new__(cls)
            cls.register(ParamLightness._instance, "Lightness")
        return ParamLightness._instance

    def getLightness(self, img_path):
        img = cv2.imread(img_path)
        # 把图片转换为单通道的灰度图
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 获取形状以及长宽
        img_shape = gray_img.shape
        height, width = img_shape[0], img_shape[1]
        size = gray_img.size
        # 灰度图的直方图
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        # 计算灰度图像素点偏离均值(128)程序
        ma = 0
        # np.full 构造一个数组，用指定值填充其元素
        reduce_matrix = np.full((height, width), 128)
        shift_value = gray_img - reduce_matrix
        shift_sum = np.sum(shift_value)
        da = shift_sum / size
        # 计算偏离128的平均偏差
        for i in range(256):
            ma += (abs(i-128-da) * hist[i])
        m = abs(ma / size)
        # 亮度系数
        k = abs(da) / m
        return k
        # print(k)
        # if k[0] > 1:
        #     # 过亮
        #     if da > 0:
        #         print("过亮")
        #     else:
        #         print("过暗")
        # else:
        #     print("亮度正常")

    def calculate(self, img1=None, img2=None):
        data = {"img1": img1, "img2": img2}
        print("ParamLightness calculate: ", data)
        k1 = self.getLightness(img1)
        k2 = self.getLightness(img2)

        return k1, k2
