# 灰度值
import cv2
from Strategy.AbstractStrategy import AbstractStrategy


class ParamGray(AbstractStrategy):

    def __new__(cls, *args, **kwargs):
        for arg in args:
            print("arg: ", arg)
        for k, v in kwargs.items():
            print(f"key: {k}, value: {v}")
        if not hasattr(ParamGray, "_instance"):
            print("ParamGray __new__")
            ParamGray._instance = super(ParamGray, cls).__new__(cls)
            cls.register(ParamGray._instance, "Gray")
        return ParamGray._instance

    def calculate(self, img1=None, img2=None):
        data = {"img1": img1, "img2": img2}
        print("ParamGray calculate: ", data)
        image1 = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2GRAY)
        m1 = cv2.mean(image1)

        image2 = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2GRAY)
        m2 = cv2.mean(image2)

        return m1[0], m2[0]
