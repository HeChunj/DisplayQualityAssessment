import os
import subprocess
import cv2
import psutil
import requests
import time

from utils.util import crop_image, get_homography_res


def kill_process_tree(pid):
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()
    parent.kill()


class GeoFormer():

    def match(new_file_paths=None, all_files_monitor=None, img_idx_list=None, demo_name=None):
        croped_image_path_list_ref = [[] for _ in range(22)]
        croped_image_path_list_dst = [[] for _ in range(22)]

        # 定义环境名称和脚本路径
        conda_env = "GeoFormer"
        script_path = "../GeoFormer/inference.py"

        start_time = time.time()

        try:
            # 启动GeoFormer服务
            command = f"conda run -n {conda_env} python {script_path}"
            process = subprocess.Popen(command, shell=True)
            # 检查服务是否启动
            while True:
                try:
                    response = requests.get('http://localhost:5001/check')
                    if response.status_code == 200:
                        print("GeoFormer service is ready")
                        break

                except requests.exceptions.RequestException as e:
                    print("GeoFormer service is not ready, retrying...")

                time.sleep(1)

            # 处理图片
            cnt = 0
            for i, (img_dst, img_ref) in enumerate(zip(new_file_paths, all_files_monitor)):
                cnt += 1

                # 获取图块的保存目录 result_dir
                filename_without_ext = os.path.splitext(
                    os.path.basename(img_dst))[0]
                result_dir_dst = f"static/croped_result_{demo_name}/finetune_dst/{filename_without_ext}"
                result_dir_ref = f"static/croped_result_{demo_name}/finetune_ref/{filename_without_ext}"

                data = {"img1": img_dst, "img2": img_ref,
                        "img_idx": img_idx_list[i]}
                print(
                    f"process image: {img_idx_list[i]}, modelGeoFormer match: ", data)

                # 检查目录是否存在，如果存在，直接返回结果，不进行图块裁剪
                if os.path.exists(result_dir_dst) and os.path.isdir(result_dir_ref):
                    # 读取目录下的所有文件
                    croped_image_path_list_ref[img_idx_list[i]] = [
                        os.path.join(result_dir_dst, f) for f in os.listdir(result_dir_dst)]
                    croped_image_path_list_dst[img_idx_list[i]] = [
                        os.path.join(result_dir_ref, f) for f in os.listdir(result_dir_ref)]
                    continue

                # 调用GeoFormer服务
                try:
                    # 尝试5次
                    for _ in range(5):
                        response = requests.post(
                            'http://localhost:5001/match', json=data)
                        if response.status_code == 200:
                            break
                except requests.exceptions.RequestException as e:
                    print(f"image {img_idx_list[i]} retrying...")

                r = response.json()

                # 向monitor对齐
                aligned_img, matrix = get_homography_res(
                    img_dst, img_ref, r['kpts1'], r['kpts2'], r['matches'], is_draw=False)

                # 保存aligned_img
                save_path = f"aligned_imgs/{demo_name}/{img_idx_list[i]}.jpg"
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                cv2.imwrite(save_path, cv2.cvtColor(
                    aligned_img, cv2.COLOR_RGB2BGR))

                # 对齐后按照monitor的块坐标裁剪图像
                croped_image_path_dst = crop_image(
                    save_path, r['kpts2'], 300, demo_name, "finetune_dst")
                croped_image_path_ref = crop_image(
                    img_ref, r['kpts2'], 300, demo_name, "finetune_ref")

                croped_image_path_list_dst[img_idx_list[i]
                                           ] = croped_image_path_dst
                croped_image_path_list_ref[img_idx_list[i]
                                           ] = croped_image_path_ref

        except subprocess.CalledProcessError as e:
            # 终止进程
            kill_process_tree(process.pid)
            process.wait()
            print(f"Error: {e}")
        finally:
            # 终止进程
            kill_process_tree(process.pid)
            process.wait()
        print(f"match images: , time: {time.time() - start_time} seconds")
        return croped_image_path_list_dst, croped_image_path_list_ref
