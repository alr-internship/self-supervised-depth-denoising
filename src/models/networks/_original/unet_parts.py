import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import conv, norm, ListModule, BloorPool, ConvLSTMCell


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad, kernel_size=3):
        super(unetConv2, self).__init__()

        # print(pad)
        if norm_layer is not None:
            self.conv1 = nn.Sequential(conv(in_size, out_size, kernel_size, bias=need_bias, pad=pad),
                                       norm(out_size, norm_layer),
                                       nn.ReLU(),)
            self.conv2 = nn.Sequential(conv(out_size, out_size, kernel_size, bias=need_bias, pad=pad),
                                       norm(out_size, norm_layer),
                                       nn.ReLU(),)
        else:
            self.conv1 = nn.Sequential(conv(in_size, out_size, kernel_size, bias=need_bias, pad=pad),
                                       nn.ReLU(),)
            self.conv2 = nn.Sequential(conv(out_size, out_size, kernel_size, bias=need_bias, pad=pad),
                                       nn.ReLU(),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetDown(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        norm_layer,
        need_bias,
        pad,
        downsample_mode
    ):
        super(unetDown, self).__init__()

        self.conv = unetConv2(in_size, out_size, norm_layer, need_bias, pad)

        if downsample_mode == 'down':
            self.down = nn.MaxPool2d(2, 2)
        elif downsample_mode == 'avg':
            self.down = nn.AvgPool2d(2, 2)
        elif downsample_mode == 'bloor':
            self.down = BloorPool(channels=in_size, stride=2)
        elif downsample_mode == 'maxbloor':
            self.down = nn.Sequential(
                nn.MaxPool2d(2, stride=1),
                BloorPool(channels=in_size, stride=2)
            )

    def forward(self, inputs):
        outputs = self.down(inputs)
        outputs = self.conv(outputs)
        return outputs
