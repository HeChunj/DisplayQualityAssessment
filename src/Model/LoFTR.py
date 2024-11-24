from Strategy.AbstractStrategy import AbstractStrategy
from Strategy.Strategy import Strategy
import time
import requests


class ModelLoFTR(AbstractStrategy):

    def __new__(cls, *args, **kwargs):
        for arg in args:
            print("arg: ", arg)
        for k, v in kwargs.items():
            print(f"key: {k}, value: {v}")
        if not hasattr(ModelLoFTR, "_instance"):
            print("ModelLoFTR __new__")
            ModelLoFTR._instance = super(ModelLoFTR, cls).__new__(cls)
            cls.register(ModelLoFTR._instance, "LoFTR")
        return ModelLoFTR._instance

    def match(self, img1=None, img2=None):
        data = {"img1": img1, "img2": img2}
        print("ModelLoFTR match: ", data)
        start_time = time.time()
        response = requests.post('http://localhost:5002/match', json=data)
        print(f"Time elapsed, rpc: {time.time() - start_time} seconds")
        r = response.json()
        return r
