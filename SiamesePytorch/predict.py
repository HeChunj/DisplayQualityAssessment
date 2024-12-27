import numpy as np
from PIL import Image

from siamese import Siamese

if __name__ == "__main__":
    model = Siamese()
    image_1 = Image.open("/home/hechunjiang/gradio/样品1 LG 65UF8580/华为P50手机采集图像/样品1采集图像/4.jpg")
    image_2 = Image.open("/home/hechunjiang/gradio/样品1 LG 65UF8580/华为P50手机采集图像/监视器采集图像/4.jpg")
    # image_1 = Image.open("/data/hechunjiang/KADID-10k/kadid10k/images/I01_01_05.png")
    # image_2 = Image.open("/data/hechunjiang/KADID-10k/kadid10k/images/I01.png")
    # image_1 = Image.open("/home/hechunjiang/gradio/GeoFormer/croped_result/finetune_dst/2/cropped_image_28.png")
    # image_2 = Image.open("/home/hechunjiang/gradio/GeoFormer/croped_result/finetune_ref/2/cropped_image_28.png")
    probability = model.detect_image(image_1,image_2)
    print(probability)
    # while True:
    #     image_1 = input('Input image_1 filename:')
    #     try:
    #         image_1 = Image.open(image_1)
    #     except:
    #         print('Image_1 Open Error! Try again!')
    #         continue

    #     image_2 = input('Input image_2 filename:')
    #     try:
    #         image_2 = Image.open(image_2)
    #     except:
    #         print('Image_2 Open Error! Try again!')
    #         continue
        
