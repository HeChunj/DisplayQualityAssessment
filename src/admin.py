import gradio as gr
import os
import requests
import numpy as np
import shutil
from PIL import Image
import json
from gradio_image_prompter import ImagePrompter


# 本地上传文件
original_path = '../upload_data/img-orgin/'
img_list_paths = os.listdir(original_path)
img_list_paths.sort()
all_files = [original_path + file for file in img_list_paths]


def upload_file(files):
    file_paths = [file.name for file in files]
    new_file_paths = []
    for file_path in file_paths:
        new_file_path = original_path + file_path.split('/')[-1]
        if os.path.exists(new_file_path):
            os.remove(new_file_path)
        shutil.move(file_path, new_file_path)
        new_file_paths.append(new_file_path)
    return new_file_paths


def get_points(prompts):
    res = {}
    res['points'] = []
    for point in prompts['points']:
        res_point = {}
        res_point["x1"] = point[0]
        res_point["y1"] = point[1]
        res_point["x2"] = point[3]
        res_point["y2"] = point[4]
        res['points'].append(res_point)
    return res


# 保存结果到本地
def save_highlight(done, todo, file_path, prompts):
    res = {}
    res['file_path'] = file_path
    res['points'] = []
    for point in prompts['points']:
        res_point = {}
        res_point["x1"] = point[0]
        res_point["y1"] = point[1]
        res_point["x2"] = point[3]
        res_point["y2"] = point[4]
        print(res_point)
        res['points'].append(res_point)
    current_directory = os.getcwd()
    filename_without_ext = os.path.splitext(os.path.basename(file_path))[0]
    json_file_path = f'{current_directory}/result/attention_area/{filename_without_ext}.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(res, json_file, indent=4)
    done = done + 1
    todo = todo - 1
    return done, todo


def append_highlight(done, todo, file_path, prompts):
    res = {}
    res['file_path'] = file_path
    res['points'] = []
    for point in prompts['points']:
        res_point = {}
        res_point["x1"] = point[0]
        res_point["y1"] = point[1]
        res_point["x2"] = point[3]
        res_point["y2"] = point[4]
        print(res_point)
        res['points'].append(res_point)
    current_directory = os.getcwd()
    filename_without_ext = os.path.splitext(os.path.basename(file_path))[0]
    json_file_path = f'{current_directory}/result/attention_area/{filename_without_ext}.json'
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
        res['points'] = res['points'] + data['points']
    with open(json_file_path, 'w') as json_file:
        json.dump(res, json_file, indent=4)
    done = done + 1
    todo = todo - 1
    return done, todo


def adjust_param(param):
    return param


def adapt_img(path):
    return {"image": path, "points": []}


def predict(img):
    return img["composite"]


left_app1 = gr.Interface(fn=upload_file, inputs=gr.File(label="上传原图", file_count="multiple"),
                         outputs=gr.Gallery(label="上传的结果", columns=[3]))


def update_img_next(now):
    now = now + 1
    if now >= len(all_files):
        now = len(all_files) - 1
    file_path = all_files[now]
    img = adapt_img(file_path)
    return img, now, file_path, section(img, all_files[now])


def update_img_before(now):
    now = now - 1
    if now < 0:
        now = 0
    file_path = all_files[now]
    img = adapt_img(file_path)
    return img, now, file_path, section(img, all_files[now])


def section(img, json_path):
    image = img['image']
    current_directory = os.getcwd()
    filename_without_ext = os.path.splitext(os.path.basename(json_path))[0]
    json_file_path = f'{current_directory}/result/attention_area/{filename_without_ext}.json'

    sections = []
    cnt = 0
    for point in img['points']:
        sections.append(
            ((int(point[0]), int(point[1]), int(point[3]), int(point[4])), str(cnt)))
        cnt += 1

    # 判断json_file_path是否存在
    if not os.path.exists(json_file_path):
        return (image, sections)
    data = json.load(open(json_file_path))
    for point in data['points']:
        sections.append(
            ((int(point["x1"]), int(point["y1"]), int(point["x2"]), int(point["y2"])), str(cnt)))
        cnt += 1

    return (image, sections)


def save_bak_param(selected_model, threshold, image_size):
    update_param = {}
    update_param['model'] = selected_model
    update_param['param'] = {}
    update_param['param']['threshold'] = threshold
    update_param['param']['image_size'] = image_size
    r = requests.post('http://localhost:5003/config/save_config',
                  json=update_param)
    if r.status_code == 200:
        gr.Info("保存成功", duration=1.5)


with gr.Blocks() as left_app2:
    with gr.Row():
        total = gr.Number(len(all_files), label='原图总数量', interactive=False)
        now = gr.Number(0, label='当前图片', interactive=False)
        done = gr.Number(0, label='已完成', interactive=False)
        todo = gr.Number(len(all_files), label='待完成', interactive=False)
    with gr.Row():
        img = ImagePrompter(
            value={"image": all_files[now.value], "points": []}, label="待处理的图片")
        points_out = gr.JSON(label="Points", visible=False)
        img_output = gr.AnnotatedImage()
        path = gr.Text(all_files[now.value], lines=1,
                       label='当前图片路径', visible=False)
        gr.Interface(
            fn=section,
            inputs=[img, path],
            outputs=img_output,
            allow_flagging="never"
        )

    with gr.Row():
        with gr.Row():
            bn_before = gr.Button("上一张")
            bn_next = gr.Button("下一张")
            bn_save = gr.Button("覆盖结果", variant="primary")
            bn_save_append = gr.Button("追加结果", variant="primary")
        gr.on(triggers=[bn_save.click], fn=save_highlight, inputs=[
            done, todo, path, img], outputs=[done, todo])
        gr.on(triggers=[bn_save_append.click], fn=append_highlight, inputs=[
            done, todo, path, img], outputs=[done, todo])
        gr.on(triggers=[bn_before.click],
              fn=update_img_before, inputs=now, outputs=[img, now, path, img_output])
        gr.on(triggers=[bn_next.click],
              fn=update_img_next, inputs=now, outputs=[img, now, path, img_output])

with gr.Blocks() as left_app3:
    with gr.Row():
        selected_model = gr.Dropdown(
            ["GeoFormer", "LoFTR", "SIFT"], label="请选择图像匹配算法", value="GeoFormer"
        )

    with gr.Row():
        threshold = gr.Slider(0, 1, 0.9, 0.05, label="阈值threshold")
    with gr.Row():
        image_size = gr.Number(640, label='图像尺寸', interactive=True)
    with gr.Row():
        save_param = gr.Button("保存参数", variant="primary")
    save_param.click(fn=save_bak_param, inputs=[selected_model,
                     threshold, image_size], outputs=None)

with gr.Blocks() as admin:
    with gr.Row():
        # 本地上传文件
        with gr.Column():
            gr.TabbedInterface(
                [left_app1, left_app2, left_app3],
                tab_names=["上传原图", "获取关注区域", "调整模型参数"],
                title="管理界面"
            )
