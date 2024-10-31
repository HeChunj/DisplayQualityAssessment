from ParamCalculate.GLCM import ParamGLCM
from ParamCalculate.Gray import ParamGray
from ParamCalculate.HSV import ParamHSV
from ParamCalculate.LBP import ParamLBP
from ParamCalculate.Lightness import ParamLightness
from ParamCalculate.RGB import ParamRGB
from ParamCalculate.Resolution import ParamResolution
from Strategy.Context import Context
import json
import os


class ParamCalService:
    def __init__(self):
        self.param_list = {"RGB", "HSV", "Gray",
                           "Resolution", "Lightness", "GLCM", "LBP"}
        print(
            f"init ParamCalculateService, these params are available: {self.param_list}")
        ParamRGB()
        ParamHSV()
        ParamGray()
        ParamResolution()
        ParamLightness()
        ParamGLCM()
        ParamLBP()

    def get_cal_result(self, model_name, img1, img2):
        # 获取当前工作目录
        current_directory = os.getcwd()

        # 提取文件名（不带扩展名）
        filename_without_ext_1 = os.path.splitext(
            os.path.basename(img1))[0]

        filename_without_ext_2 = os.path.splitext(
            os.path.basename(img2))[0]

        result_dir_1 = f'{current_directory}/croped_result/{model_name}/{filename_without_ext_1}'
        result_dir_2 = f'{current_directory}/croped_result/{model_name}/{filename_without_ext_2}'

        filepath1 = []
        filepath2 = []
        for filename in os.listdir(result_dir_1):
            filepath1.append(os.path.join(result_dir_1, filename))

        for filename in os.listdir(result_dir_2):
            filepath2.append(os.path.join(result_dir_2, filename))

        # 将结果保存为一个dict, 写入到json文件
        json_file_path = f'{current_directory}/result/{model_name}/{filename_without_ext_1}_{filename_without_ext_2}.json'
        features = {}
        features["model_name"] = model_name
        features[filename_without_ext_1] = {}
        features[filename_without_ext_2] = {}
        for image1, image2 in zip(filepath1, filepath2):
            features[filename_without_ext_1][image1] = {}
            features[filename_without_ext_2][image2] = {}
            for param in self.param_list:
                strategy = Context.get_strategy(param)
                if strategy is None:
                    raise ValueError(f"Invalid param name: {param}")

                # 遍历param_list列表，获取对应的参数计算结果
                r1, r2 = strategy.calculate(image1, image2)
                features[filename_without_ext_1][image1][param] = str(r1)
                features[filename_without_ext_2][image2][param] = str(r2)
        with open(json_file_path, 'w') as json_file:
            json.dump(features, json_file, indent=4)
        return json_file_path
        # return r1, r2
