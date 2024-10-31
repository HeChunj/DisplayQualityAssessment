import os
from copy import deepcopy

import torch
import cv2

import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure
from flask import Flask, request, jsonify
import requests
from src.loftr import LoFTR, default_cfg
from src.utils.data_io import load_gray_scale_tensor_cv

# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.
_default_cfg = deepcopy(default_cfg)
# set to False when using the old ckpt
_default_cfg['coarse']['temp_bug_fix'] = True
matcher = LoFTR(config=_default_cfg)
matcher.load_state_dict(torch.load("weights/indoor_ds_new.ckpt")['state_dict'])
matcher = matcher.eval().cuda()


def load_im(im_path, device, imsize, enhanced=False):
    return load_gray_scale_tensor_cv(
        im_path, device, imsize=imsize, dfactor=8, enhanced=enhanced, value_to_scale=min
    )


def match_pairs(img0_pth, img1_pth):
    # Load example images
    # img0_pth = "assets/scannet_sample_images/scene0711_00_frame-001680.jpg"
    # img1_pth = "assets/scannet_sample_images/scene0711_00_frame-001995.jpg"
    gray1, sc1 = load_im(im_path=img0_pth, device='cuda', imsize=640)
    gray2, sc2 = load_im(im_path=img1_pth, device='cuda', imsize=640)
    # img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
    # img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
    # img0_raw = cv2.resize(img0_raw, (640, 640))
    # img1_raw = cv2.resize(img1_raw, (640, 640))

    # img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    # img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
    batch = {'image0': gray1, 'image1': gray2}
    upscale = np.array([sc1 + sc2])

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        kpts1 = batch['mkpts0_f'].cpu().numpy()
        kpts2 = batch['mkpts1_f'].cpu().numpy()
        scores = batch['mconf'].cpu().numpy()

    matches = np.concatenate([kpts1, kpts2], axis=1)

    # Upscale matches &  kpts
    matches = upscale * matches
    kpts1 = sc1 * kpts1
    kpts2 = sc2 * kpts2

    return matches, kpts1, kpts2, scores


app = Flask(__name__)
post_data = {"model": "LoFTR"}
configs = requests.post(
    'http://localhost:5003/config/get_config', json=post_data).json()


@app.route('/match', methods=['POST'])
def predict():
    data = request.json
    print("LoFTR, data: ", data['img1'], data['img2'])
    matches, kpts1, kpts2, scores = match_pairs(data['img1'], data['img2'])
    result = {"kpts1": kpts1.tolist(), "kpts2": kpts2.tolist(),
              "scores": scores.tolist()}
    print("LoFTR, result: ", result)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host=configs['host'], port=configs['port'])
