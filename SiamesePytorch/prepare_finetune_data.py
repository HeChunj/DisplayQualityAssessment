'''
准备微调的数据集
'''

import argparse
import json
import os

from omegaconf import OmegaConf
import pandas as pd
from utils.utils import crop_image
from utils.utils import calculate_iou, map_opinion_to_model
from data_processors import ProcessorRegistry
from sklearn.utils import resample


def generate_original_dataset(config: OmegaConf):
    begin_idx = 1
    end_idx = config['total_img_num'] + 1
    demo_list = config['demo_list']
    scores = config['scores_original']
    all_index = config['target_index_to_img_idx']

    # 每一个demo的每一张图片，生成对应的csv文件
    for demo in demo_list:
        for i in range(begin_idx, end_idx):
            # TODO: 考虑使用数据库存储
            dst_image_path = f"/home/hechunjiang/gradio/GeoFormer/croped_result_{demo}/finetune_dst/{i}/"
            ref_image_path = f"/home/hechunjiang/gradio/GeoFormer/croped_result_{demo}/finetune_ref/{i}/"
            dst_image_list = os.listdir(dst_image_path)
            ref_image_list = os.listdir(ref_image_path)
            dst_image_list = [os.path.join(dst_image_path, x)
                              for x in dst_image_list if x.endswith(".png")]
            ref_image_list = [os.path.join(ref_image_path, x)
                              for x in ref_image_list if x.endswith(".png")]
            dst_image_list.sort(key=lambda x: int(
                x.split("_")[-1].split(".")[0]))
            ref_image_list.sort(key=lambda x: int(
                x.split("_")[-1].split(".")[0]))

            # 读取ref image的坐标文件
            json_ref = f"/home/hechunjiang/gradio/GeoFormer/croped_result_{demo}/finetune_ref/{i}/cropped_image_coordinate.json"
            coordinates_ref = []
            with open(json_ref, "r") as f:
                coordinates_ref = json.load(f)["coordinates"]

            # 读取label_ref，即attention区域
            label_json_ref = f"/home/hechunjiang/gradio/src/result/attention_area/{demo}/transformed_attention_{i}.json"
            label_ref = []
            if os.path.exists(label_json_ref):
                with open(label_json_ref, "r") as f:
                    label_ref = json.load(f)["points"]

            # 保存符合条件的dst_img_list和ref_image_list
            # 即选择iou大于0.5的图块进入数据集
            dst_image_list_filtered = []
            ref_image_list_filtered = []
            for idx, coordinate in enumerate(coordinates_ref):
                for label in label_ref:
                    # 计算iou
                    iou = calculate_iou(coordinate, label)
                    if iou >= 0.5:
                        dst_image_list_filtered.append(dst_image_list[idx])
                        ref_image_list_filtered.append(ref_image_list[idx])
                        break

            print(
                f"original: dst_filtered = {len(dst_image_list_filtered)}, ref_filtered = {len(ref_image_list_filtered)}")

            dst_image_list = dst_image_list_filtered
            ref_image_list = ref_image_list_filtered

            df = pd.DataFrame()
            df['dist_img'] = dst_image_list
            df['ref_img'] = ref_image_list
            # 这里要注意，图片的序号是从1开始的，所以scores的索引也是从1开始的
            df['dmos'] = map_opinion_to_model(scores[demo][i])
            if not os.path.exists(f"/home/hechunjiang/gradio/GeoFormer/finetune_data/{demo}/"):
                os.makedirs(
                    f"/home/hechunjiang/gradio/GeoFormer/finetune_data/{demo}/")
            df.to_csv(
                f"/home/hechunjiang/gradio/GeoFormer/finetune_data/{demo}/all_attention_original_{i}.csv", index=False)

        # 根据指标生成不同的csv文件
        for index in all_index.keys():
            df_all = pd.DataFrame()
            s_idx = all_index[index][0]
            e_idx = all_index[index][1] + 1
            for i in range(s_idx, e_idx):
                df = pd.read_csv(
                    f"/home/hechunjiang/gradio/GeoFormer/finetune_data/{demo}/all_attention_original_{i}.csv")
                df_all = pd.concat([df_all, df])
                # 删除radius_.csv文件
                # os.remove(f"/home/hechunjiang/gradio/GeoFormer/finetune_data/{demo}/all_attention{i}.csv")
            df_all.to_csv(
                f"/home/hechunjiang/gradio/GeoFormer/finetune_data/{demo}/all_attention_original_{index}.csv", index=False)

        print(f"prepare original data, demo: {demo} finished!")

    # 将每个样品的同一指标合并
    for index in all_index.keys():
        df_all = pd.DataFrame()
        for demo in demo_list:
            df = pd.read_csv(
                f"/home/hechunjiang/gradio/GeoFormer/finetune_data/{demo}/all_attention_original_{index}.csv")
            df_all = pd.concat([df_all, df])

        if not os.path.exists("/home/hechunjiang/gradio/GeoFormer/finetune_data/original_data/"):
            os.makedirs(
                "/home/hechunjiang/gradio/GeoFormer/finetune_data/original_data")
        df_all.to_csv(
            f"/home/hechunjiang/gradio/GeoFormer/finetune_data/original_data/all_attention_original_data_{index}.csv", index=False)


def generate_augmented_dataset(config: OmegaConf):
    '''
    扩充数据集
    只扩充得分最低的那张图片, 且是往下扩充
    对于每个指标, 基本上有三个阶梯的扩充
    扩充图片的最终结果保存在GeoFormer/finetune_data/{demo}/{index}下面
    '''

    scores = config['scores_agumented']
    all_index = config['target_index_to_img_idx']
    augment_data_param = config['augment_data_param']
    augment_data_demo = config['augment_data_demo']
    image_path_list = {
        "LG": "/home/hechunjiang/gradio/样品1 LG 65UF8580/华为P50手机采集图像/样品1采集图像/",
        "SONY": "/home/hechunjiang/gradio/样品2 SONY 43吋/华为P50手机采集图像/",
        "AMAZON": "/home/hechunjiang/gradio/样品3 亚马逊 43吋/华为P50手机采集图像/"
    }

    for key, _ in all_index.items():
        df_all = pd.DataFrame()
        augmented_method = ProcessorRegistry.get_processor(key)
        print(f"prepare augment data, index: {key} begin!")
        param_list = augment_data_param[key]
        demo = augment_data_demo[key][0]
        i = augment_data_demo[key][1]
        image_path = image_path_list[demo] + f"{i}.jpg"

        # 只处理第i张图片
        coordinate_path = f"/home/hechunjiang/gradio/GeoFormer/croped_result_{demo}/finetune_dst/{i}/cropped_image_coordinate.json"
        # 读取ref的图片，形成文件名称的list
        ref_image_path = f"/home/hechunjiang/gradio/GeoFormer/croped_result_{demo}/finetune_ref/{i}/"
        ref_image_list_org = os.listdir(ref_image_path)
        ref_image_list_org = [
            x for x in ref_image_list_org if x.endswith(".png")]
        ref_image_list_org.sort(key=lambda x: int(
            x.split("_")[-1].split(".")[0]))

        # 读取label_ref，即attention区域
        label_json_ref = f"/home/hechunjiang/gradio/src/result/attention_area/{demo}/transformed_attention_{i}.json"
        label_ref = []
        if os.path.exists(label_json_ref):
            with open(label_json_ref, "r") as f:
                label_ref = json.load(f)["points"]

        # 读取ref的坐标文件
        json_ref = f"/home/hechunjiang/gradio/GeoFormer/croped_result_{demo}/finetune_ref/{i}/cropped_image_coordinate.json"
        coordinates_ref = []
        with open(json_ref, "r") as f:
            coordinates_ref = json.load(f)["coordinates"]

        for j in range(len(param_list)):
            augment_img = augmented_method(image_path, param_list[j])
            augment_img_save_path = f"/home/hechunjiang/gradio/GeoFormer/finetune_data/{demo}/{key}/{i}/param_{param_list[j]}.jpg"
            if not os.path.exists(os.path.dirname(augment_img_save_path)):
                os.makedirs(os.path.dirname(augment_img_save_path))
            augment_img.save(augment_img_save_path)

            # 读取dst坐标文件，按照坐标文件的坐标对图片进行裁剪
            dst_img_list = crop_image(
                augment_img_save_path, coordinate_path, param_list[j], i, demo_name=demo)

            # 判断ref和dst的数量是否相等
            assert len(dst_img_list) == len(ref_image_list_org)

            # 根据attention_area中的坐标筛选dst_img_list和ref_image_list
            # 保存符合条件的dst_img_list和ref_image_list
            dst_img_list_filtered = []
            ref_image_list_filtered = []
            for idx, coordinate in enumerate(coordinates_ref):
                for label in label_ref:
                    # 计算iou
                    iou = calculate_iou(coordinate, label)
                    if iou >= 0.5:
                        dst_img_list_filtered.append(dst_img_list[idx])
                        ref_image_list_filtered.append(
                            ref_image_list_org[idx])
                        break

            print(
                f"augment: dst_filtered = {len(dst_img_list_filtered)}, ref_filtered = {len(ref_image_list_filtered)}")

            dst_img_list = dst_img_list_filtered
            ref_image_list = ref_image_list_filtered

            assert len(dst_img_list) == len(ref_image_list)

            # 将r, ref_image_list, scores保存到csv文件中
            df = pd.DataFrame()
            df['dist_img'] = dst_img_list
            df['ref_img'] = [os.path.join(ref_image_path, x)
                             for x in ref_image_list]
            df['dmos'] = map_opinion_to_model(scores[key][j])

            if not os.path.exists(f"/home/hechunjiang/gradio/GeoFormer/finetune_data/augment_data/{key}"):
                os.makedirs(
                    f"/home/hechunjiang/gradio/GeoFormer/finetune_data/augment_data/{key}")

            df.to_csv(
                f"/home/hechunjiang/gradio/GeoFormer/finetune_data/augment_data/{key}/param_{param_list[j]}.csv", index=False)

            df_all = pd.concat([df_all, df])
            # 删除radius_.csv文件
            # os.remove(f"/home/hechunjiang/gradio/GeoFormer/finetune_data/{demo}/{i}/radius_attention_{radius}.csv")

        df_all.to_csv(
            f"/home/hechunjiang/gradio/GeoFormer/finetune_data/augment_data/all_attention_augment_data_{key}.csv", index=False)
        print(f"prepare augment data, index: {key} finished!")


def combine_data(config: OmegaConf):
    # 原始数据集: "/home/hechunjiang/gradio/GeoFormer/finetune_data/original_data/all_attention_original_data_{index}.csv"
    # 扩充数据集: "/home/hechunjiang/gradio/GeoFormer/finetune_data/augment_data/all_attention_augment_data_{index}.csv"
    # 取所有数据集的并集
    all_index = config['target_index_to_img_idx']
    target_size = config['target_size']
    for key, _ in all_index.items():
        df_org = pd.read_csv(
            f"/home/hechunjiang/gradio/GeoFormer/finetune_data/original_data/all_attention_original_data_{key}.csv")

        df_aug = pd.read_csv(
            f"/home/hechunjiang/gradio/GeoFormer/finetune_data/augment_data/all_attention_augment_data_{key}.csv")

        df_train = pd.concat([df_org, df_aug], ignore_index=True)
        df_train.to_csv(
            f"/home/hechunjiang/gradio/GeoFormer/finetune_data/final_train_finetune_data_{key}_all.csv", index=False)

        # 对于df_train，做如下逻辑处理
        # 1. 首先将df_train进行重采样，大于target_size的类别进行下采样；小于target_size的类别进行上采样
        # 2. 然后根据dmos进行分组，将频率的倒数作为权重，计算Loss的时候使用
        # 3. 保存权重到指定文件中
        balanced = None
        if len(df_train) > target_size:
            # 下采样：随机选择目标数量的数据
            downsampled = resample(df_train,
                                   replace=False,  # 不放回采样
                                   n_samples=target_size,
                                   random_state=42)
            balanced = downsampled
            print(f"{key} : 下采样到 {len(downsampled)}")
        elif len(df_train) < target_size:
            # 上采样：随机复制数据直到达到目标数量
            upsampled = resample(df_train,
                                 replace=True,  # 放回采样
                                 n_samples=target_size,
                                 random_state=42)
            balanced = upsampled
            print(f"{key} : 上采样到 {len(upsampled)}")
        else:
            # 如果数据量刚好等于目标数量，不做处理
            balanced = df_train
            print(f"{key} : 保持原样，数量为 {len(df_train)}")

        balanced.to_csv(
            f"/home/hechunjiang/gradio/GeoFormer/finetune_data/final_train_finetune_data_{key}.csv", index=False)

        # 计算权重
        # 1. 首先根据dmos进行分组
        # 2. 计算每个分组的频率
        # 3. 计算每个分组的权重，保存为json文件
        group = balanced.groupby("dmos").size()
        group = group / len(balanced)
        group = 1 / group
        group = group / group.sum()

        with open(f"/home/hechunjiang/gradio/Siamese-pytorch/loss_weights/weights_{key}.json", "w") as f:
            json.dump(group.to_dict(), f)


def main(config: OmegaConf):
    '''
    生成两部分数据集
    1. 扩充数据集
    2. 原始数据集
    '''

    # 首先根据原始数据图片生成对应的原始数据集
    generate_original_dataset(config)

    # 再根据扩充数据图片生成对应的扩充数据集
    generate_augmented_dataset(config)

    # 对原始数据集和扩充数据集进行组合
    combine_data(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="/home/hechunjiang/gradio/Siamese-pytorch/configs/finetune_data.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    main(config)
