# HSV
import cv2
from Strategy.AbstractStrategy import AbstractStrategy


class ParamHSV(AbstractStrategy):

    def __new__(cls, *args, **kwargs):
        for arg in args:
            print("arg: ", arg)
        for k, v in kwargs.items():
            print(f"key: {k}, value: {v}")
        if not hasattr(ParamHSV, "_instance"):
            print("ParamHSV __new__")
            ParamHSV._instance = super(ParamHSV, cls).__new__(cls)
            cls.register(ParamHSV._instance, "HSV")
        return ParamHSV._instance

    def calculate(self, img1=None, img2=None):
        data = {"img1": img1, "img2": img2}
        print("ParamHSV calculate: ", data)
        image1 = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2HSV)
        m1 = cv2.mean(image1)

        image2 = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2HSV)
        m2 = cv2.mean(image2)

        return m1[0:3], m2[0:3]
