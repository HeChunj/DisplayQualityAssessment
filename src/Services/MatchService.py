from Strategy.Context import Context
from Model.GeoFomer import ModelGeoFomer
from Model.LoFTR import ModelLoFTR


class MatchService:
    def __init__(self):
        model_list = {"GeoFormer", "LoFTR", "SIFT"}
        print(f"init MatchService, these models are available: {model_list}")
        ModelGeoFomer()
        ModelLoFTR()

    def get_match_result(self, model_name, img1, img2):
        strategy = Context.get_strategy(model_name)
        if strategy is None:
            raise ValueError(f"Invalid model name: {model_name}")
        return strategy.match(img1, img2)
