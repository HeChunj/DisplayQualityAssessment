import os
import gradio as gr
import pandas as pd


def load_demo_img_list(demo_name):
    img_dir = f"../upload_data/img-TV/{demo_name}"
    img_list = os.listdir(img_dir)
    img_list.sort(key=lambda x: int(x.split(".")[0]))
    img_list = [f"{img_dir}/{img}" for img in img_list]
    return img_list


monitor_img_list = [
    "../upload_data/img-monitor/1.jpg",
    "../upload_data/img-monitor/2.jpg",
    "../upload_data/img-monitor/3.jpg",
    "../upload_data/img-monitor/4.jpg",
    "../upload_data/img-monitor/5.jpg",
    "../upload_data/img-monitor/6.jpg",
    "../upload_data/img-monitor/7.jpg",
    "../upload_data/img-monitor/8.jpg",
    "../upload_data/img-monitor/9.jpg",
    "../upload_data/img-monitor/10.jpg",
    "../upload_data/img-monitor/11.jpg",
    "../upload_data/img-monitor/12.jpg",
    "../upload_data/img-monitor/13.jpg",
    "../upload_data/img-monitor/14.jpg",
    "../upload_data/img-monitor/15.jpg",
    "../upload_data/img-monitor/16.jpg",
    "../upload_data/img-monitor/17.jpg",
    "../upload_data/img-monitor/18.jpg",
    "../upload_data/img-monitor/19.jpg",
    "../upload_data/img-monitor/20.jpg",
    "../upload_data/img-monitor/21.jpg"
]

index_list = ["清晰度",
              "白平衡",
              "灰阶",
              "彩色饱和度",
              "彩色准确性",
              "对比度",
              "区域控光"]


index_detail_list = ["清晰度 - 1", "清晰度 - 2", "清晰度 - 3",
                     "白平衡 - 1", "白平衡 - 2", "白平衡 - 3",
                     "灰阶 - 1", "灰阶 - 2", "灰阶 - 3",
                     "彩色饱和度 - 1", "彩色饱和度 - 2", "彩色饱和度 - 3",
                     "彩色准确性 - 1", "彩色准确性 - 2", "彩色准确性 - 3",
                     "对比度 - 1", "对比度 - 2", "对比度 - 3",
                     "区域控光 - 1", "区域控光 - 2", "区域控光 - 3"]


def get_scores(score_path, demo_name):
    if os.path.exists(score_path):
        df = pd.read_csv(score_path)
        df = df.round(2)
        return df[demo_name].tolist()
    else:
        return [0] * 21


def cal_selected_image(score_state, score_state_list, selection: gr.SelectData):
    index = int(selection.value['image']['orig_name'].split('.')[0]) - 1
    score_state = score_state_list[index]
    return monitor_img_list[index], score_state


def cal_score(score_state):
    return score_state


def get_show_page(demo_name):
    with gr.Blocks() as display:
        image_list = load_demo_img_list(demo_name)
        gr.HTML(value="<h1 style='text-align: center;'>结果展示</h1>")
        score_path = f"static/final_output_{demo_name}/final_output.csv"
        score_state_list = gr.State(value=get_scores(score_path, demo_name)) 
        with gr.Row():
            gallery = gr.Gallery(value=image_list, preview=True)
            monitor_img = gr.Image(
                value=monitor_img_list[0], label="monitor图片")
            score_state = gr.State(value=0)
            gallery.select(fn=cal_selected_image,
                           inputs=[score_state, score_state_list], outputs=[monitor_img, score_state])
        with gr.Row():
            with gr.Column():
                out = gr.Textbox(label="这张图片的得分是：", interactive=False,
                                 value=score_state_list.value[0])
            with gr.Column():
                with gr.Row():
                    with gr.Accordion(label="得分详情", open=False):
                        for i in range(0, len(score_state_list.value), 3):
                            with gr.Row():
                                for j in range(3):
                                    if i + j < len(score_state_list.value):
                                        gr.Textbox(
                                            label=f"{index_detail_list[i + j]} 得分：", interactive=False, value=score_state_list.value[i + j])
                                avg_score = sum(
                                    score_state_list.value[i:i+3]) / min(3, len(score_state_list.value) - i)
                                gr.Textbox(
                                    label=f"{index_list[i // 3]} 得分：", interactive=False, value=round(avg_score, 2))
                        with gr.Row():
                            with gr.Column():
                                gr.Textbox(label="总体评分：", interactive=False, value=round(
                                    sum(score_state_list.value) / len(score_state_list.value), 2))
                            with gr.Column():
                                download_button = gr.DownloadButton(variant="primary",
                                                                    value=f"static/final_output_{demo_name}/final_output.csv", label="下载评价结果")
        score_state.change(cal_score, score_state, out)
    display.queue()
    display.startup_events()
    return display
