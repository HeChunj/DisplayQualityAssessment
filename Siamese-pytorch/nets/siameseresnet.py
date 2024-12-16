import torch
import torch.nn as nn
from torchvision import models
from torch.hub import load_state_dict_from_url
from nets.vgg import VGG16


def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        stride = 2
        for i in range(5):
            input_length = (input_length + 2 *
                            padding[i] - filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width) * get_output_length(height)


class SiameseResnet(nn.Module):
    def __init__(self, input_shape, pretrained=True):
        super(SiameseResnet, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        if pretrained:
            state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth", model_dir="./model_data")
            self.resnet.load_state_dict(state_dict)
        
        # flat_shape = 512 * \
        #     get_img_output_length(input_shape[1], input_shape[0])
        flat_shape = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Identity()
        self.fully_connect1 = torch.nn.Linear(flat_shape, 1024)
        self.fully_connect2 = torch.nn.Linear(1024, 512)
        self.fully_connect3 = torch.nn.Linear(512, 1)

    def forward(self, x1, x2):
        # x1, x2 = x
        # ------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        # ------------------------------------------#
        x1 = self.resnet(x1)
        x2 = self.resnet(x2)
        # -------------------------#
        #   相减取绝对值，取l1距离
        # -------------------------#
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        # x = torch.abs(x1 - x2)
        x = x1 - x2
        # -------------------------#
        #   进行两次全连接
        # -------------------------#
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        x = self.fully_connect3(x)
        x = 4 * torch.tanh(x)
        return x
