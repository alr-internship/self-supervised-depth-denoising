import imp
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from networks.UNet.unet_parts import DoubleConv, Down, Up, OutConv
from networks.LSTMUNet.lstm_unet_parts import ConvLSTMCell

# TODO: Check ConvLSTM cell insertions for correctness


class LSTMUNet(nn.Module):
    '''
        upsample_mode in ['deconv', 'nearest', 'bilinear']
        pad in ['zero', 'replication', 'none']
    '''

    def __init__(
        self,
        n_channels,
        n_classes,
        bilinear=True,
        name='LSTMUNet',
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.name = name                     # name for wandb

        factor = 2 if bilinear else 1

        self.InConv = DoubleConv(n_channels, 64)

        # '- n_channels' for concatenation later of skip connections
        self.Down1 = Down(64, 128 - n_channels)
        self.Down2 = Down(128, 256 - n_channels)
        self.Down3 = Down(256, 512 - n_channels)
        self.Down4 = Down(512, 1024 // factor - n_channels)

        self.Up4 = Up(1024, 512 // factor, bilinear)
        self.Up3 = Up(512, 256 // factor, bilinear)
        self.Up2 = Up(256, 128 // factor, bilinear)
        self.Up1 = Up(128, 64, bilinear)

        self.OutConv = OutConv(64, n_classes)

        self.conv_lstm_cell1 = ConvLSTMCell(128, 128, 5)
        self.conv_lstm_cell2 = ConvLSTMCell(256, 256, 5)
        self.conv_lstm_cell3 = ConvLSTMCell(512, 512, 5)

        self.conv_lstm_cell4 = ConvLSTMCell(512, 256, 5)
        self.conv_lstm_cell5 = ConvLSTMCell(256, 128, 5)

    '''
    all_inputs = shape([, seq_len, , , ])
    '''

    def forward(self, all_inputs):
        seq_len = all_inputs.shape[1]
        res = []
        for seq_idx in range(seq_len):
            inputs = all_inputs[:, seq_idx, :, :, :]

            AdaptSize = nn.AvgPool2d(2, 2)
            inputs_skip = [inputs]
            for _ in range(4):
                inputs_skip.append(AdaptSize(inputs_skip[-1]))

            in_conv = torch.cat([self.InConv(inputs), inputs_skip[0]], 1)

            down1 = torch.cat([self.Down1(in_conv), inputs_skip[1]], 1)
            if seq_idx == 0:
                c_1 = self.first_conv_lstm_cell.init_hidden(
                    batch_size=inputs.shape[0],
                    shape=(
                        down1.shape[-2],
                        down1.shape[-1]
                    )
                )
            down1, c_1 = self.first_conv_lstm_cell(down1, c_1[0], c_1[1])
            c_1 = [down1, c_1]

            down2 = torch.cat([self.Down2(down1), inputs_skip[2]], 1)
            if seq_idx == 0:
                c_2 = self.second_conv_lstm_cell.init_hidden(
                    batch_size=inputs.shape[0],
                    shape=(
                        down2.shape[-2],
                        down2.shape[-1]
                    )
                )
            down2, c_2 = self.second_conv_lstm_cell(down2, c_2[0], c_2[1])
            c_2 = [down2, c_2]

            down3 = torch.cat([self.Down3(down2), inputs_skip[3]], 1)
            if seq_idx == 0:
                c_3 = self.third_conv_lstm_cell.init_hidden(
                    batch_size=inputs.shape[0],
                    shape=(
                        down3.shape[-2],
                        down3.shape[-1]
                    )
                )
            down3, c_3 = self.third_conv_lstm_cell(down3, c_3[0], c_3[1])
            c_3 = [down3, c_3]

            down4 = torch.cat([self.Down4(down3), inputs_skip[4]], 1)

            up4 = self.Up4(down4, down3)
            up3 = self.Up3(up4, down2)

            if seq_idx == 0:
                c_4 = self.conv_lstm_cell4.init_hidden(
                    batch_size=inputs.shape[0], shape=(up3.shape[-2], up3.shape[-1]))
            up3, c_4 = self.conv_lstm_cell4(up3, c_4[0], c_4[1])
            c_4 = [up3, c_4]

            up2 = self.up2(up3, down1)
            if seq_idx == 0:
                c_5 = self.conv_lstm_cell5.init_hidden(
                    batch_size=inputs.shape[0], shape=(up2.shape[-2], up2.shape[-1]))
            up2, c_5 = self.conv_lstm_cell5(up2, c_5[0], c_5[1])
            c_5 = [up2, c_5]

            up1 = self.up1(up2, in_conv)
            res.append(self.final(up1)[:, None])
        return torch.cat(res, 1)
