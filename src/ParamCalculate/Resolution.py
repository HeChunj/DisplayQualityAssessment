# 清晰度
import cv2
from Strategy.AbstractStrategy import AbstractStrategy


class ParamResolution(AbstractStrategy):

    def __new__(cls, *args, **kwargs):
        for arg in args:
            print("arg: ", arg)
        for k, v in kwargs.items():
            print(f"key: {k}, value: {v}")
        if not hasattr(ParamResolution, "_instance"):
            print("ParamResolution __new__")
            ParamResolution._instance = super(
                ParamResolution, cls).__new__(cls)
            cls.register(ParamResolution._instance, "Resolution")
        return ParamResolution._instance

    def calculate(self, img1=None, img2=None):
        data = {"img1": img1, "img2": img2}
        print("ParamResolution calculate: ", data)
        image1 = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2GRAY)
        m1 = cv2.Laplacian(image1, cv2.CV_64F).var()

        image2 = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2GRAY)
        m2 = cv2.Laplacian(image2, cv2.CV_64F).var()

        return m1, m2
