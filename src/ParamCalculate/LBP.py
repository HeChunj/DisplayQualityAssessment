# LBP
import cv2
from Strategy.AbstractStrategy import AbstractStrategy
from skimage.feature import local_binary_pattern
from skimage import data, filters


class ParamLBP(AbstractStrategy):

    def __new__(cls, *args, **kwargs):
        for arg in args:
            print("arg: ", arg)
        for k, v in kwargs.items():
            print(f"key: {k}, value: {v}")
        if not hasattr(ParamLBP, "_instance"):
            print("ParamLBP __new__")
            ParamLBP._instance = super(ParamLBP, cls).__new__(cls)
            cls.register(ParamLBP._instance, "LBP")
        return ParamLBP._instance

    def calculate(self, img1=None, img2=None):
        data = {"img1": img1, "img2": img2}
        print("ParamLBP calculate: ", data)
        # settings for LBP
        radius = 3  # LBP算法中范围半径的取值
        n_points = 8 * radius  # 领域像素点数
        image1 = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2GRAY)
        lbp1 = local_binary_pattern(image1, n_points, radius)

        image2 = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2GRAY)
        lbp2 = local_binary_pattern(image2, n_points, radius)
        return lbp1, lbp2
