import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ConvLSTM_cell(nn.Module):
    def __init__(self, args):
        super(ConvLSTM_cell, self).__init__()
        # dims = (height, width, chanels)
        self.input_dims = args.input_dims
        self.hidden_dims = args.hidden_dims
        self.memory_dims = args.memory_dims
        self.kernel_dims = args.kernel_dims
        self.padding = self.kernel_dims // 2

        self.input_conv = nn.Conv2d()
        self.memory_out_conv = nn.Conv2d()


    def forward(self, input, hidden_in, memory_in):
        memory_out = f * memory_in + i * F.tanh()
        hidden_out = F.tanh(o * memory_out)