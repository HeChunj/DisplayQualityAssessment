import gradio as gr
import pandas as pd
import os

all_scores = {
    "LG": pd.DataFrame(),
    "SONY": pd.DataFrame(),
    "AMAZON": pd.DataFrame()
}

all_match_res = {
    "LG": [],
    "SONY": [],
    "AMAZON": []
}

file_path = ["/home/hechunjiang/gradio/Siamese-pytorch/final_output_tanh-vgg16/final_output.csv",
             "/home/hechunjiang/gradio/Siamese-pytorch/final_output_tanh-resnet50/final_output.csv",
             "/home/hechunjiang/gradio/Siamese-pytorch/final_output_tanh-vit/final_output.csv"]

index_list = ["清晰度 - 1", "清晰度 - 2", "清晰度 - 3",
              "白平衡 - 1", "白平衡 - 2", "白平衡 - 3",
              "灰阶 - 1", "灰阶 - 2", "灰阶 - 3",
              "彩色饱和度 - 1", "彩色饱和度 - 2", "彩色饱和度 - 3",
              "彩色准确性 - 1", "彩色准确性 - 2", "彩色准确性 - 3",
              "对比度 - 1", "对比度 - 2", "对比度 - 3",
              "区域控光 - 1", "区域控光 - 2", "区域控光 - 3"]

demo_list = ["LG", "SONY", "AMAZON"]

def init_data():
    # file_path里面的数据是分别是三个模型的输出结果
    # 每个文件里面是三个样品的输出结果
    # 现在要把这些数据读出来，按照样品进行分组，存入all_scores中
    for path in file_path:
        data = pd.read_csv(path)
        for head in data.columns:
            all_scores[head] = pd.concat(
                [all_scores[head], data[head]], axis=1)
    for key in all_scores.keys():
        all_scores[key].columns = ['VGG16', 'ResNet50', 'VIT']
        # 插入一列opinion到第一列
        all_scores[key].insert(0, 'opinion', pd.read_csv(
            f"/home/hechunjiang/gradio/upload_data/opinion_scores/{key}.csv"))
        # 四舍五入保留两位小数
        all_scores[key] = all_scores[key].round(2)
        all_scores[key].insert(0, '指标', index_list)
    return all_scores

def init_match_res():
    for demo in demo_list:
        img_list = os.listdir(f"/home/hechunjiang/gradio/src/result/match_res/{demo}")
        for img in img_list:
            all_match_res[demo].append(f"/home/hechunjiang/gradio/src/result/match_res/{demo}/{img}")

init_data()
init_match_res()


def load_data(demo_name):
    return all_scores[demo_name]


def load_match_res(demo_name):
    return all_match_res[demo_name]


# 加载数据
lg_data = load_data('LG')
sony_data = load_data('SONY')
amazon_data = load_data('AMAZON')

LG_dataframe = gr.DataFrame(value=lg_data, interactive=False)
SONY_dataframe = gr.DataFrame(value=sony_data, interactive=False)
AMAZON_dataframe = gr.DataFrame(value=amazon_data, interactive=False)

table_interface_scores = gr.TabbedInterface(
    [LG_dataframe, SONY_dataframe, AMAZON_dataframe],
    tab_names=["样品1: LG", "样品2: SONY", "样品3: AMAZON"]
)

lg_match_res = load_match_res('LG')
sony_match_res = load_match_res('SONY')
amazon_match_res = load_match_res('AMAZON')

LG_match_res = gr.Gallery(value=lg_match_res)
SONY_match_res = gr.Gallery(value=sony_match_res)
AMAZON_match_res = gr.Gallery(value=amazon_match_res)


match_res = gr.TabbedInterface(
    [LG_match_res, SONY_match_res, AMAZON_match_res], tab_names=["LG", "SONY", "AMAZON"])

table_interface_img = gr.TabbedInterface([match_res], tab_names=["匹配结果展示"])

with gr.Blocks() as display:
    with gr.Row():
        with gr.Column():
            gr.TabbedInterface(
                [table_interface_scores, table_interface_img],
                tab_names=["分数展示", "图片展示"],
                title="展示界面"
            )
