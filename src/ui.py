import os
import gradio as gr
from Services.MatchService import MatchService
from Services.ParamCalService import ParamCalService
from utils.util import crop_image

match_service = MatchService()
param_cal_service = ParamCalService()


def get_img_lits(img_dir):
    imgs_List = [os.path.join(img_dir, name) for name in sorted(os.listdir(
        img_dir)) if name.endswith(('.png', '.jpg', '.webp', '.tif', '.jpeg'))]
    return imgs_List


img_path_list = [0]*3


def input_img_path(path_dir, idx):
    idx = int(idx)
    img_path_list[idx] = (get_img_lits(path_dir))
    print(img_path_list[idx])
    return img_path_list[idx]


# left_tab1
left_app1 = gr.Interface(fn=input_img_path,
                         inputs=[gr.Textbox(value="/home/hechunjiang/gradio/data/img-monitors"), gr.Text(
                             visible=False, value="0")],
                         outputs=gr.Gallery(label="上传的结果", columns=[3]),
                         allow_flagging="never")
# left_tab2
left_app2 = gr.Interface(fn=input_img_path,
                         inputs=[gr.Textbox(value="/home/hechunjiang/gradio/data/img-TV"), gr.Text(
                             visible=False, value="1")],
                         outputs=gr.Gallery(label="上传的结果", columns=[3]),
                         allow_flagging="never")

# img_path_list = ['/home/hechunjiang/gradio/data/img-orgin/1.jpg',
#                  '/home/hechunjiang/gradio/data/img-TV/1-TV.jpg']


def model_handler(model_name):
    r = match_service.get_match_result(
        model_name, img_path_list[0][0], img_path_list[1][0])
    croped_image_path1 = crop_image(
        img_path_list[0][0], r['kpts1'], 300, model_name)
    croped_image_path2 = crop_image(
        img_path_list[1][0], r['kpts2'], 300, model_name)
    return croped_image_path1, croped_image_path2


def param_handler(model_name):
    r = param_cal_service.get_cal_result(
        model_name, img_path_list[0][0], img_path_list[1][0])
    return r


right_app1 = gr.Interface(
    model_handler,
    inputs=[
        gr.Dropdown(
            ["GeoFormer", "LoFTR", "SIFT"], label="请选择图像匹配算法"
        )
    ],
    outputs=[gr.Gallery(label="处理的结果", height='auto', columns=[5], rows=[5], type="image"),
             gr.Gallery(label="处理的结果", height='auto', columns=[5], rows=[5], type="image")],
    allow_flagging="never"
)

right_app2 = gr.Interface(
    param_handler,
    inputs=[
        gr.Dropdown(
            ["GeoFormer", "LoFTR", "SIFT"], label="请选择图像匹配算法"
        )
    ],
    outputs=[gr.Textbox(label="处理的结果", type="text")],
    allow_flagging="never"
)


with gr.Blocks() as index:
    with gr.Row():
        with gr.Column():
            gr.TabbedInterface(
                [left_app1, left_app2],
                tab_names=["上传monitor图像", "上传TV图像"],
                title="上传图像"
            )

        with gr.Column():
            gr.TabbedInterface(
                [right_app1, right_app2],
                tab_names=["图像预处理", "图像质量分析"],
                title="处理图像"
            )
