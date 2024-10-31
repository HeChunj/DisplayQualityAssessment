from abc import ABC, abstractmethod


# 策略接口
class Strategy(ABC):

    @abstractmethod
    def match(self, img1, img2):
        pass

    @abstractmethod
    def calculate(self, img1, img2):
        pass
