# 灰度共生矩阵参数计算
import cv2
from Strategy.AbstractStrategy import AbstractStrategy
from skimage.feature import graycomatrix, graycoprops


class ParamGLCM(AbstractStrategy):

    def __new__(cls, *args, **kwargs):
        for arg in args:
            print("arg: ", arg)
        for k, v in kwargs.items():
            print(f"key: {k}, value: {v}")
        if not hasattr(ParamGLCM, "_instance"):
            print("ParamGLCM __new__")
            ParamGLCM._instance = super(
                ParamGLCM, cls).__new__(cls)
            cls.register(ParamGLCM._instance, "GLCM")
        return ParamGLCM._instance

    def calculate(self, img1=None, img2=None):
        data = {"img1": img1, "img2": img2}
        print("ParamGLCM calculate: ", data)

        # 读取图片
        image1 = cv2.imread(img1)
        gray_img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        # 计算灰度共生矩阵
        glcm1 = graycomatrix(gray_img1, [1], [0], symmetric=True, normed=True)
        contrast1 = graycoprops(glcm1, prop='contrast')
        dissimilarity1 = graycoprops(glcm1, prop='dissimilarity')
        homogeneity1 = graycoprops(glcm1, prop='homogeneity')
        energy1 = graycoprops(glcm1, prop='energy')
        correlation1 = graycoprops(glcm1, prop='correlation')
        # 读取图片
        image2 = cv2.imread(img2)
        gray_img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        # 计算灰度共生矩阵
        glcm2 = graycomatrix(gray_img2, [1], [0], symmetric=True, normed=True)
        contrast2 = graycoprops(glcm2, prop='contrast')
        dissimilarity2 = graycoprops(glcm2, prop='dissimilarity')
        homogeneity2 = graycoprops(glcm2, prop='homogeneity')
        energy2 = graycoprops(glcm2, prop='energy')
        correlation2 = graycoprops(glcm2, prop='correlation')

        return [glcm1, contrast1, dissimilarity1, homogeneity1, energy1, correlation1], [glcm2, contrast2, dissimilarity2, homogeneity2, energy2, correlation2]
        # return [glcm1.tolist(), contrast1, dissimilarity1, homogeneity1, energy1, correlation1], [glcm2.tolist(), contrast2, dissimilarity2, homogeneity2, energy2, correlation2]
