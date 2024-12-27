import subprocess
import time

from model.GeoFormer import kill_process_tree


def inference(demo_name, i):
    # 定义环境名称和脚本路径
    conda_env = "siamese"
    script_path = "../SiamesePytorch/inference.py"

    start_time = time.time()
    try:
        command = f"conda run -n {conda_env} python {script_path} --demo_name {demo_name} --i {i}"
        process = subprocess.Popen(command, shell=True)
    except subprocess.CalledProcessError as e:
        kill_process_tree(process.pid)
        print(f"Error: {e}")
    finally:
        process.wait()

    print(f"siamese inference, time: {time.time() - start_time} seconds")
