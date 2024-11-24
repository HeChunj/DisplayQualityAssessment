'''
微调文件的入口
默认配置文件是: configs/finetune_config.yaml, 配置文件统一放在configs目录下
读取配置文件, 然后调用main函数
在main函数中, 根据配置文件的参数, 调用不同的模型进行微调
'''

import os
import time

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from tqdm import tqdm

from nets.siamesevgg import SiameseVgg
from nets.siameseresnet import SiameseResnet
from nets.siamesevit import SiameseVIT
from utils.callbacks import LossHistory
from utils.KADID10Kdataloader import dataset_collate
from utils.utils import (calculate_iou, download_weights, get_lr_scheduler, load_dataset, map_b_to_a,
                         set_optimizer_lr, show_config)
from utils.utils_fit import fit_one_epoch
from utils.KADID10Kdataloader import KADID10kDataset
import argparse
from siamese import Siamese
from PIL import Image
import json


def main(config: OmegaConf):
    # ----------------------------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # ----------------------------------------------------#
    Cuda = True if config['device'] == 'cuda' else False
    # ---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    #   DP模式：
    #       设置            distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式：
    #       设置            distributed = True
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    # ---------------------------------------------------------------------#
    distributed = config['distributed']
    # ---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    # ---------------------------------------------------------------------#
    sync_bn = False
    # ---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    # ---------------------------------------------------------------------#
    fp16 = config['fp16']
    # ----------------------------------------------------#
    #   数据集存放的路径
    # ----------------------------------------------------#
    image_folder = ""
    dmos_file_train = config['dmos_file_train']
    # dmos_file_train = "/home/hechunjiang/gradio/GeoFormer/finetune_data/40_finetune_data_new.csv"
    # dmos_file_val = "/home/hechunjiang/gradio/GeoFormer/finetune_data/all_test_finetune_data_baipingheng.csv"
    # ----------------------------------------------------#
    #   输入图像的大小，默认为224,224
    # ----------------------------------------------------#
    input_shape = config["image_size"]
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的，pretrain不影响此处的权值加载。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，pretrain = Fasle，此时从0开始训练。
    #   一般来讲，从0开始训练效果会很差，因为权值太过随机，特征提取效果不明显。
    #
    #   网络一般不从0开始训练，至少会使用主干部分的权值，有些论文提到可以不用预训练，主要原因是他们 数据集较大 且 调参能力优秀。
    #   如果一定要训练网络的主干部分，可以了解imagenet数据集，首先训练分类模型，分类模型的 主干部分 和该模型通用，基于此进行训练。
    # ----------------------------------------------------------------------------------------------------------------------------#

    if config['pretrained_model'] == 'vgg16':
        model_path = config['vgg16_model_path']
        siamese = SiameseVgg
    elif config['pretrained_model'] == 'resnet50':
        model_path = config['resnet50_model_path']
        siamese = SiameseResnet
    elif config['pretrained_model'] == 'vit':
        model_path = config['vit_model_path']
        siamese = SiameseVIT
    model = siamese(input_shape)
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #   受到BatchNorm层影响，不能为1。
    #
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从预训练权重开始训练：
    #       Adam：
    #           Init_Epoch = 0，Epoch = 100，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0。
    #       SGD：
    #           Init_Epoch = 0，Epoch = 100，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4。
    #       其中：UnFreeze_Epoch可以在100-300之间调整。
    #   （二）batch_size的设置：
    #       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
    #       受到BatchNorm层影响，batch_size最小为2，不能为1。
    #       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
    # ----------------------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------------#
    #   训练参数
    #   Init_Epoch      模型当前开始的训练世代
    #   Epoch           模型总共训练的epoch
    #   batch_size      每次输入的图片数量
    # ------------------------------------------------------#
    Init_Epoch = config['Init_Epoch']
    Epoch = config['Epoch']
    batch_size = config['batch_size']

    # ------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    Init_lr = config['Init_lr']
    Min_lr = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ------------------------------------------------------------------#
    optimizer_type = config['optimizer']
    momentum = config['momentum']
    weight_decay = config['weight_decay']
    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
    # ------------------------------------------------------------------#
    lr_decay_type = config['lr_decay_type']
    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    # ------------------------------------------------------------------#
    save_period = config['save_period']
    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    save_dir = config['save_dir']
    # ------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    # ------------------------------------------------------------------#
    num_workers = config['num_workers']

    # ------------------------------------------------------#
    #   设置用到的显卡
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(
                f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    model.requires_grad_(False)
    for name, module in model.named_modules():
        if name.endswith(tuple(config['trainable_modules'])):
            for params in module.parameters():
                params.requires_grad = True

    # 通过下面的代码可以查看模型的每一层是否参与训练
    # for name, param in model.named_parameters():
    #     print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
    # exit(0)

    if model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[
                  :500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[
                  :500], "……\nFail To Load Key num:", len(no_load_key))
            print(
                "\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    # ----------------------#
    #   获得损失函数
    # ----------------------#
    loss = nn.MSELoss()

    # ----------------------#
    #   记录Loss，按照以下方式处理保存的路径
    #   save_dir = datetime + "-" + save_dir + "-" + model_name + "-" + target_index
    #   save_dir = 训练的时间 + 保存的目录 + 模型的名字 + 评价指标
    # ----------------------#
    # 获取时间
    date = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    save_dir = "logs/" + date + "-" + config['save_dir'] + "-" + \
        config['pretrained_model'] + "-" + config['target_index']
    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # ------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    # ------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    # ----------------------------#
    #   多卡同步Bn
    # ----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            # ----------------------------#
            #   多卡平行运行
            # ----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # ----------------------------------------------------#
    #   训练集和验证集的比例。
    # ----------------------------------------------------#
    train_ratio = 0.8
    # 创建数据集
    dataset_train = KADID10kDataset(input_shape=input_shape,
                                    image_folder=image_folder, dmos_file=dmos_file_train, random=True, autoaugment_flag=True)

    num_train = int(train_ratio * len(dataset_train))
    num_val = len(dataset_train) - num_train

    dataset_train, dataset_val = random_split(
        dataset_train, [num_train, num_val])

    if local_rank == 0:
        show_config(
            model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Epoch=Epoch, batch_size=batch_size,
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )
        # ---------------------------------------------------------#
        #   总训练世代指的是遍历全部数据的总次数
        #   总训练步长指的是梯度下降的总次数
        #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
        #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
        # ----------------------------------------------------------#
        wanted_step = 3e4 if optimizer_type == "sgd" else 1e4
        total_step = num_train // batch_size * Epoch
        if total_step <= wanted_step:
            wanted_epoch = wanted_step // (num_train // batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (
                optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m" % (
                num_train, batch_size, Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m" % (
                total_step, wanted_step, wanted_epoch))

    # -------------------------------------------------------------#
    #   训练分为两个阶段，两阶段初始的学习率不同，手动调节了学习率
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    # -------------------------------------------------------------#
    if True:
        # -------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        # -------------------------------------------------------------------#
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr,
                          lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr,
                         lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # ---------------------------------------#
        #   根据optimizer_type选择优化器
        # ---------------------------------------#
        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
        }[optimizer_type]

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(
            lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)

        # ---------------------------------------#
        #   判断每一个世代的长度
        # ---------------------------------------#
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        # train_dataset   = SiameseDataset(input_shape, train_lines, train_labels, True)
        # val_dataset     = SiameseDataset(input_shape, val_lines, val_labels, False)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_train, shuffle=True,)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_val, shuffle=False,)
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(dataset_train, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, sampler=train_sampler)
        gen_val = DataLoader(dataset_val, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, sampler=val_sampler)

        for epoch in range(Init_Epoch, Epoch):
            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss, loss_history, optimizer, epoch, epoch_step,
                          epoch_step_val, gen, gen_val, Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)

        if local_rank == 0:
            loss_history.writer.close()

    if config['get_final_output'] and not config['finetune_only']:
        get_final_output(save_dir, config)


def get_final_output(logs_path: str, config: OmegaConf):
    '''
    保存最终的输出结果
    args:
        logs_path: str, 训练日志文件夹，可以拿到模型的权重
        final_output_path: str, 保存最终输出结果的文件夹
        target_index: str, 评价指标
    '''
    # 1. 准备配置
    config_list = {
        "model_path": logs_path + "/best_epoch_weights.pth",
        "model_name": config['pretrained_model'],
        # "model_path": '/home/hechunjiang/gradio/Siamese-pytorch/logs-finetune-4-qingxidu/best_epoch_weights.pth',
        # "model_path": '/home/hechunjiang/gradio/Siamese-pytorch/logs-finetune-4-qingxidu/loss_2024_11_22_00_58_38/best_epoch_weights.pth',
        "image_size": config['image_size'],
    }
    config_ = OmegaConf.create(config_list)

    # 2. 加载模型
    model = Siamese(config_)

    # 3. 计算推理的结果
    target_index = config['target_index']
    begin_idx = config['target_index_to_img_idx'][target_index][0]
    end_idx = config['target_index_to_img_idx'][target_index][1]
    demo_list = config['demo_list']

    # 按照backbone保存结果文件，判断保存结果的文件是否存在
    final_output_path = config['final_output_path'] + \
        "-" + config['pretrained_model']
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)
        res_df = pd.DataFrame()
    else:
        res_df = pd.read_csv(f"{final_output_path}/final_output.csv")

    pbar = tqdm(total=(end_idx - begin_idx + 1) * len(demo_list))
    for demo in demo_list:
        img_list_1 = []
        img_list_2 = []
        scores = [[] for _ in range(config['total_img_num'])]
        res = [[] for _ in range(config['total_img_num'])]

        for i in range(begin_idx, end_idx + 1):
            img_list_1 = os.listdir(
                f"/home/hechunjiang/gradio/GeoFormer/croped_result_{demo}/finetune_dst/{i}/")
            img_list_2 = os.listdir(
                f"/home/hechunjiang/gradio/GeoFormer/croped_result_{demo}/finetune_ref/{i}/")
            # 过滤掉.json文件
            img_list_1 = [x for x in img_list_1 if x.endswith(".png")]
            img_list_2 = [x for x in img_list_2 if x.endswith(".png")]
            img_list_1.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            img_list_2.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            for path1, path2 in zip(img_list_1, img_list_2):
                image_1 = Image.open(
                    f"/home/hechunjiang/gradio/GeoFormer/croped_result_{demo}/finetune_dst/{i}/{path1}")
                image_2 = Image.open(
                    f"/home/hechunjiang/gradio/GeoFormer/croped_result_{demo}/finetune_ref/{i}/{path2}")
                scores[i].append(model.detect_image(
                    image_1, image_2).cpu().numpy())

            json_ref = f"/home/hechunjiang/gradio/GeoFormer/croped_result_{demo}/finetune_ref/{i}/cropped_image_coordinate.json"
            coordinates_ref = []
            with open(json_ref, "r") as f:
                coordinates_ref = json.load(f)["coordinates"]

            label_json_ref = f"/home/hechunjiang/gradio/src/result/attention_area/{i}.json"
            label_ref = []

            # 如果label_json_ref不存在，则跳过
            if not os.path.exists(label_json_ref):
                continue
            with open(label_json_ref, "r") as f:
                label_ref = json.load(f)["points"]

            # 对于每一个coordinates_ref中的框，计算其和label_ref中每一个框的iou
            for idx, coordinate in enumerate(coordinates_ref):
                for label in label_ref:
                    # 计算iou
                    iou = calculate_iou(coordinate, label)
                    if iou >= 0.5:
                        res[i].append(scores[i][idx])
                        break

            avg_score = np.array(0)
            if len(scores[i]) != 0:
                avg_score = sum(res[i]) / len(res[i])

            # 将avg_score[0]添加到res_df[demo]中
            res_df.loc[i - 1, demo] = map_b_to_a(avg_score[0])

            pbar.update()

    res_df.to_csv(f"{final_output_path}/final_output.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="/home/hechunjiang/gradio/Siamese-pytorch/configs/finetune_config.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    model_list = ['vgg16', 'resnet50', 'vit']
    for model in model_list:
        config['pretrained_model'] = model
        if config['get_output_only']:
            get_final_output(config['logs_path'], config)
        else:
            main(config)
