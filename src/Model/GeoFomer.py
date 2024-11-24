from Strategy.AbstractStrategy import AbstractStrategy
from Strategy.Strategy import Strategy
import requests
import time


class ModelGeoFomer(AbstractStrategy):

    def __new__(cls, *args, **kwargs):
        for arg in args:
            print("arg: ", arg)
        for k, v in kwargs.items():
            print(f"key: {k}, value: {v}")
        if not hasattr(ModelGeoFomer, "_instance"):
            print("ModelGeoFomer __new__")
            ModelGeoFomer._instance = super(ModelGeoFomer, cls).__new__(cls)
            cls.register(ModelGeoFomer._instance, "GeoFormer")
        return ModelGeoFomer._instance

    def match(self, img1=None, img2=None):
        data = {"img1": img1, "img2": img2}
        print("ModelGeoFomer match: ", data)
        start_time = time.time()
        response = requests.post('http://localhost:5001/match', json=data)
        print(f"Time elapsed, rpc: {time.time() - start_time} seconds")
        r = response.json()
        return r
