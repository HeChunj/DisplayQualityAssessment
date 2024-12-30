import json
from PIL import Image
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import argparse
import os

from tqdm import tqdm
from siamese import Siamese
from utils.utils import calculate_iou, map_model_to_opinion

script_dir = os.path.dirname(os.path.abspath(__file__))


def get_final_output(model_ckpt_name: str, config: OmegaConf):
    '''
    保存最终的输出结果
    args:
        model_ckpt_name: str, 模型ckpt的名称
        config: OmegaConf, 配置文件
    '''
    # 1. 准备配置
    config_list = {
        "model_path": f"model_ckpt/{model_ckpt_name}",
        "model_name": config['pretrained_model'],
        "image_size": config['image_size'],
    }
    config_ = OmegaConf.create(config_list)

    # 2. 加载模型
    model = Siamese(config_)

    # 3. 计算推理的结果
    target_index = config['target_index']
    begin_idx = config['target_index_to_img_idx'][target_index][0]
    end_idx = config['target_index_to_img_idx'][target_index][1]
    demo_list = config['demo_list']

    # 保存结果文件，判断保存结果的文件是否存在
    final_output_path = config['final_output_path']
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)
        res_df = pd.DataFrame(data=np.zeros(21), index=range(0, 21), columns=demo_list)
    elif not os.path.exists(f"{final_output_path}/final_output.csv"):
        res_df = pd.DataFrame(data=np.zeros(21), index=range(0, 21), columns=demo_list)
    else:
        res_df = pd.read_csv(f"{final_output_path}/final_output.csv")

    pbar = tqdm(total=(end_idx - begin_idx + 1) * len(demo_list))
    for demo in demo_list:
        img_list_1 = []
        img_list_2 = []
        scores = [[] for _ in range(config['total_img_num'] + 1)]
        res = [[] for _ in range(config['total_img_num'] + 1)]
        for i in range(begin_idx, end_idx + 1):
            img_list_1 = os.listdir(
                f"static/croped_result_{demo}/finetune_dst/{i}/")
            img_list_2 = os.listdir(
                f"static/croped_result_{demo}/finetune_ref/{i}/")
            # 过滤掉.json文件
            img_list_1 = [x for x in img_list_1 if x.endswith(".png")]
            img_list_2 = [x for x in img_list_2 if x.endswith(".png")]
            img_list_1.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            img_list_2.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            for path1, path2 in zip(img_list_1, img_list_2):
                image_1 = Image.open(
                    f"static/croped_result_{demo}/finetune_dst/{i}/{path1}")
                image_2 = Image.open(
                    f"static/croped_result_{demo}/finetune_ref/{i}/{path2}")
                scores[i].append(model.detect_image(
                    image_1, image_2).cpu().numpy())

            json_ref = f"static/croped_result_{demo}/finetune_ref/{i}/cropped_image_coordinate.json"
            coordinates_ref = []
            with open(json_ref, "r") as f:
                coordinates_ref = json.load(f)["coordinates"]

            label_json_ref = f"result/attention_area/{i}.json"
            label_ref = []

            # 如果label_json_ref不存在，则跳过
            if not os.path.exists(label_json_ref):
                continue
            with open(label_json_ref, "r") as f:
                label_ref = json.load(f)["points"]

            # 对于每一个coordinates_ref中的框，计算其和label_ref中每一个框的iou
            for idx, coordinate in enumerate(coordinates_ref):
                for label in label_ref:
                    # 计算iou
                    iou = calculate_iou(coordinate, label)
                    if iou >= 0.5:
                        res[i].append(scores[i][idx])
                        break

            avg_score = np.array(0)
            if len(scores[i]) != 0:
                avg_score = sum(res[i]) / len(res[i])

            # 将avg_score[0]添加到res_df[demo]中
            res_df.loc[i - 1, demo] = map_model_to_opinion(avg_score[0])

            pbar.update()

    res_df.to_csv(f"{final_output_path}/final_output.csv",
                  mode='w', index=False)


def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=f"{script_dir}/configs/finetune_config.yaml")
    parser.add_argument("--demo_name", type=str, required=True)
    parser.add_argument("--i", type=str, required=True)
    args = parser.parse_args()

    demo_name = args.demo_name
    i = int(args.i)  # i是指标序号

    config = OmegaConf.load(args.config)

    index_list = ["", "qingxidu", "baipingheng", "huijie",
                  "caisebaohedu", "caisezhunquexing", "duibidu", "quyukongguang"]

    model_ckpt_list = os.listdir("../src/model_ckpt")
    model_ckpt_list.sort()
    model_ckpt_list.insert(0, "")

    model = model_ckpt_list[i].split('_')[1]
    config['target_index'] = index_list[i]
    config['demo_list'] = [demo_name]
    config['pretrained_model'] = model
    config['final_output_path'] = f"static/final_output_{demo_name}"

    if model == 'vit':
        config['image_size'] = [224, 224]
    else:
        config['image_size'] = [256, 256]

    # 判断是否需要进入下面的逻辑
    target_index = config['target_index']
    begin_idx = config['target_index_to_img_idx'][target_index][0]
    end_idx = config['target_index_to_img_idx'][target_index][1]
    for j in range(begin_idx, end_idx + 1):
        dst_path = f"static/croped_result_{demo_name}/finetune_dst/{j}/"
        ref_path = f"static/croped_result_{demo_name}/finetune_ref/{j}/"
        if not os.path.exists(dst_path) or not os.path.exists(ref_path):
            return

        img_list_1 = os.listdir(dst_path)
        img_list_2 = os.listdir(ref_path)
        if len(img_list_1) == 0 or len(img_list_2) == 0:
            return

    get_final_output(model_ckpt_list[i], config)


if __name__ == "__main__":
    inference()
