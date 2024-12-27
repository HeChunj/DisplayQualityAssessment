import os


def generate_html(demo_name, img_idx):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 设置目录路径
    finetune_dst_dir = f'{script_dir}/croped_result_{demo_name}/finetune_dst/{img_idx}'
    finetune_ref_dir = f'{script_dir}/croped_result_{demo_name}/finetune_ref/{img_idx}'

    relative_dst_dir = f'../croped_result_{demo_name}/finetune_dst/{img_idx}'
    relative_ref_dir = f'../croped_result_{demo_name}/finetune_ref/{img_idx}'

    # 获取所有图片文件
    dst_images = sorted([f for f in os.listdir(
        finetune_dst_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    ref_images = sorted([f for f in os.listdir(
        finetune_ref_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    # 生成HTML代码
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Pair Display</title>
        <style>
            .image-row {
                display: flex;
                justify-content: space-between;
                margin: 20px 0;
            }
            .image-pair {
                text-align: center;
            }
            .image-pair img {
                width: 300px;
                height: auto;
                border: 1px solid #ccc;
            }
            .image-title {
                margin-top: 10px;
                font-size: 14px;
                color: #555;
                word-wrap: break-word;
            }
        </style>
    </head>
    <body>
        <h1>Image Pair Display</h1>
    """

    # 假设图片数量一致并且按顺序配对
    for i in range(0, len(dst_images), 2):
        html_content += '<div class="image-row">'

        # 如果有两对图片
        for j in range(2):
            if i + j < len(dst_images):
                dst_img = dst_images[i + j]
                ref_img = ref_images[i + j]

                html_content += f"""
                <div class="image-pair">
                    <img src="{relative_dst_dir}/{dst_img}" alt="Finetune DST Image">
                    <div class="image-title">{relative_dst_dir}/{dst_img}</div>
                </div>
                <div class="image-pair">
                    <img src="{relative_ref_dir}/{ref_img}" alt="Finetune REF Image">
                    <div class="image-title">{relative_ref_dir}/{ref_img}</div>
                </div>
                """
        html_content += '</div>'  # 结束这一行

    # 结束HTML
    html_content += """
    </body>
    </html>
    """

    # 将HTML内容写入文件
    save_dir = f'{script_dir}/image_pair_{demo_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + f'/{img_idx}.html', 'w') as f:
        f.write(html_content)

    print(f"HTML file generated: {save_dir}/{img_idx}.html")
