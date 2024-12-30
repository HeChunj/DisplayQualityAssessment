from static.generate_html import generate_html
from model.GeoFormer import GeoFormer
import random
import re
import shutil
import gradio as gr
import os
from model.siamese_inference import inference


def extract_number(filename):
    match = re.search(r'(\d+)', filename)  # 匹配文件名中的数字
    return int(match.group(1)) if match else float('inf')  # 如果没有数字，放到最后


index_list = ["1. 清晰度 - 1", "2. 清晰度 - 2", "3. 清晰度 - 3",
              "4. 白平衡 - 1", "5. 白平衡 - 2", "6. 白平衡 - 3",
              "7. 灰阶 - 1", "8. 灰阶 - 2", "9. 灰阶 - 3",
              "10. 彩色饱和度 - 1", "11. 彩色饱和度 - 2", "12. 彩色饱和度 - 3",
              "13. 彩色准确性 - 1", "14. 彩色准确性 - 2", "15. 彩色准确性 - 3",
              "16. 对比度 - 1", "17. 对比度 - 2", "18. 对比度 - 3",
              "19. 区域控光 - 1", "20. 区域控光 - 2", "21. 区域控光 - 3"]

monitor_path = '../upload_data/img-monitor/'
img_list_monitor = os.listdir(monitor_path)
img_list_monitor.sort(key=extract_number)
all_files_monitor = [monitor_path + file for file in img_list_monitor]


new_file_paths = []
save_path = "../upload_data/img-TV/"


def upload_file(files, demo_name, demo_name_state):
    if demo_name == "":
        demo_name = "temp"
    file_paths = [file.name for file in files]
    new_file_paths.clear()
    for file_path in file_paths:
        new_file_path = save_path + demo_name + "/" + file_path.split('/')[-1]
        if not os.path.exists(save_path + demo_name):
            os.makedirs(save_path + demo_name)
        shutil.move(file_path, new_file_path)
        new_file_paths.append(new_file_path)
    new_file_paths.sort(key=extract_number)
    demo_name_state = demo_name
    return new_file_paths, demo_name_state


demo_name = gr.State()
input_demo_name = gr.Textbox(label="请输入品牌名称")
upload_output = gr.Gallery(label="上传的结果", columns=[3])
upload_app1 = gr.Interface(fn=upload_file,
                           inputs=[gr.File(label="上传样品图", file_count="multiple"),
                                   input_demo_name,
                                   demo_name],
                           outputs=[upload_output, demo_name])


croped_image_path_list_dst = [[] for _ in range(22)]
croped_image_path_list_ref = [[] for _ in range(22)]

sampled_croped_image_path_list_dst = [[] for _ in range(22)]
sampled_croped_image_path_list_ref = [[] for _ in range(22)]


def model_handler(model_name, demo_name):

    input_img_idx = [file.split('/')[-1].split('.')[0]
                     for file in new_file_paths]
    input_img_idx = [int(idx) for idx in input_img_idx]
    input_img_idx.sort()

    filter_files_monitor = [all_files_monitor[idx - 1]
                            for idx in input_img_idx]

    # 获取匹配的结果
    croped_image_path_list_dst, croped_image_path_list_ref = GeoFormer.match(
        new_file_paths, filter_files_monitor, input_img_idx, demo_name)

    for j in range(1, 22):
        num_samples = min(10, len(croped_image_path_list_dst[j]))  # 选择最多10张图片
        sample_indices = random.sample(
            range(len(croped_image_path_list_dst[j])), num_samples)
        sampled_croped_image_path_list_dst[j] = [
            croped_image_path_list_dst[j][i] for i in sample_indices]
        sampled_croped_image_path_list_ref[j] = [
            croped_image_path_list_ref[j][i] for i in sample_indices]
    return sampled_croped_image_path_list_dst[1], sampled_croped_image_path_list_ref[1], demo_name


croped_out_ref = gr.Gallery(
    value=sampled_croped_image_path_list_dst[1], label="监视器匹配结果", columns=[3])
croped_out_dst = gr.Gallery(
    value=sampled_croped_image_path_list_dst[1], label="样品匹配结果", columns=[3])

right_app1 = gr.Interface(
    model_handler,
    inputs=[
        gr.Text(label="图像匹配算法", value="GeoFormer", interactive=False),
        demo_name
    ],
    outputs=[croped_out_ref, croped_out_dst, demo_name],
    allow_flagging="never"
)


def process_inference(demo_name, progress=gr.Progress()):
    progress(0, desc="开始...")
    if demo_name == "" or demo_name == None or len(new_file_paths) == 0:
        return "请先上传图像"

    for i in progress.tqdm(range(1, 8), desc="正在处理..."):
        inference(demo_name, i)
        # time.sleep(1)

    return f'''
            <button onclick="window.open('http://localhost:7861/show_result/{demo_name}', '_blank')"
            style="background-color: #4CAF50; color: white; border: none; padding: 10px 20px; font-size: 1em; border-radius: 5px; cursor: pointer; transition: background-color 0.3s;">点击跳转至结果展示页面</button>
    '''


def show_croped_image(selection: gr.SelectData):
    index = int(selection.value['image']['orig_name'].split('.')[0])
    return sampled_croped_image_path_list_dst[index], sampled_croped_image_path_list_ref[index]


def save_selected(img_display_selected, selection: gr.SelectData):
    index = int(selection.value.split('.')[0])
    img_display_selected = index
    return img_display_selected


def generate_cropped_imgs_htmls(demo_name, img_display_selected):
    generate_html(demo_name, img_display_selected)


with gr.Blocks() as index:
    with gr.Row():
        with gr.Column():
            gr.TabbedInterface(
                interface_list=[upload_app1],
                tab_names=["上传TV图像"],
                title="上传图像"
            )

        with gr.Column():
            gr.TabbedInterface(
                interface_list=[right_app1],
                tab_names=["图像预处理"],
                title="处理图像"
            )
    upload_output.select(show_croped_image, inputs=None, outputs=[
                         croped_out_ref, croped_out_dst])

    with gr.Row():
        with gr.Column():
            btn_process = gr.Button(value="进行显示质量评价", variant="primary")
            progress_bar = gr.Progress()
            result_display = gr.HTML()

        btn_process.click(process_inference, [demo_name], [result_display])

        with gr.Column():
            img_dropdown = gr.Dropdown(label="选择要查看的匹配图像", choices=index_list)
            check_button = gr.Button(value="查看")
            img_display_selected = gr.Text(value=0, visible=False)
        img_dropdown.select(
            fn=save_selected, inputs=img_display_selected, outputs=img_display_selected)
        img_display_selected.change(generate_cropped_imgs_htmls,
                                    [input_demo_name, img_display_selected], [])
        check_button.click(None, [input_demo_name, img_display_selected], [],
                           js="""(demo_name, img_selected) => window.open(`http://localhost:7861/static/image_pair_${demo_name}/${img_selected}.html`, "_blank")""")
