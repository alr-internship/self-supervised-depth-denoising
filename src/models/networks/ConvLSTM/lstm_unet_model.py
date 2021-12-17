import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import conv, norm, ListModule, BloorPool, ConvLSTMCell


class LSTMUNet(nn.Module):
    '''
        upsample_mode in ['deconv', 'nearest', 'bilinear']
        pad in ['zero', 'replication', 'none']
    '''
    def __init__(self, num_input_channels=3, num_output_channels=3, 
                       feature_scale=4, more_layers=0, concat_x=False,
                       upsample_mode='deconv', pad='zero', norm_layer='in', last_act='sigmoid', need_bias=True, downsample_mode='max'):
        super(LSTMUNet, self).__init__()

        self.feature_scale = feature_scale
        self.more_layers = more_layers
        self.concat_x = concat_x


        filters = [64, 128, 256, 512, 1024]
        filters = [x // self.feature_scale for x in filters]

        self.start = unetConv2(num_input_channels, filters[0] if not concat_x else filters[0] - num_input_channels, norm_layer, need_bias, pad)

        self.down1 = unetDown(filters[0], filters[1] if not concat_x else filters[1] - num_input_channels, norm_layer, need_bias, pad, downsample_mode)
        self.down2 = unetDown(filters[1], filters[2] if not concat_x else filters[2] - num_input_channels, norm_layer, need_bias, pad, downsample_mode)
        self.down3 = unetDown(filters[2], filters[3] if not concat_x else filters[3] - num_input_channels, norm_layer, need_bias, pad, downsample_mode)
        self.down4 = unetDown(filters[3], filters[4] if not concat_x else filters[4] - num_input_channels, norm_layer, need_bias, pad, downsample_mode)

        # more downsampling layers
        if self.more_layers > 0:
            self.more_downs = [
                unetDown(filters[4], filters[4] if not concat_x else filters[4] - num_input_channels , norm_layer, need_bias, pad) for i in range(self.more_layers)]
            self.more_ups = [unetUp(filters[4], upsample_mode, need_bias, pad, same_num_filt =True) for i in range(self.more_layers)]

            self.more_downs = ListModule(*self.more_downs)
            self.more_ups   = ListModule(*self.more_ups)

        self.up4 = unetUp(filters[3], upsample_mode, need_bias, pad)
        self.up3 = NewunetUp(filters[2], upsample_mode, need_bias, pad)
        self.up2 = NewunetUp(filters[1], upsample_mode, need_bias, pad)
        self.up1 = unetUp(filters[0], upsample_mode, need_bias, pad)

        self.final = conv(filters[0], num_output_channels, 1, bias=need_bias, pad=pad)

        if last_act == 'sigmoid': 
            self.final = nn.Sequential(self.final, nn.Sigmoid())
        elif last_act == 'tanh':
            self.final = nn.Sequential(self.final, nn.Tanh())

        self.first_conv_lstm_cell = ConvLSTMCell(filters[1], filters[1], 5)
        self.second_conv_lstm_cell = ConvLSTMCell(filters[2], filters[2], 5)
        self.third_conv_lstm_cell = ConvLSTMCell(filters[3], filters[3], 5)

        self.fourth_conv_lstm_cell = ConvLSTMCell(filters[2]*2, filters[2], 5)
        self.fifth_conv_lstm_cell = ConvLSTMCell(filters[1]*2, filters[1], 5)



    def forward(self, all_inputs):
        seq_len = all_inputs.shape[1]
        res = []
        for seq_idx in range(seq_len):
            inputs = all_inputs[:,seq_idx, :,:,:]
            # Downsample 
            downs = [inputs]
            down = nn.AvgPool2d(2, 2)
            for i in range(4 + self.more_layers):
                downs.append(down(downs[-1]))

            in64 = self.start(inputs)
            if self.concat_x:
                in64 = torch.cat([in64, downs[0]], 1)
            
            down1 = self.down1(in64)
            if self.concat_x:
                down1 = torch.cat([down1, downs[1]], 1)

            if seq_idx == 0:
                c_1 = self.first_conv_lstm_cell.init_hidden(batch_size=inputs.shape[0], shape=(down1.shape[-2], down1.shape[-1]))
            down1, c_1 = self.first_conv_lstm_cell(down1, c_1[0], c_1[1])
            c_1 = [down1, c_1]

            down2 = self.down2(down1)
            if self.concat_x:
                down2 = torch.cat([down2, downs[2]], 1)
            
            if seq_idx == 0:
                c_2 = self.second_conv_lstm_cell.init_hidden(batch_size=inputs.shape[0], shape=(down2.shape[-2], down2.shape[-1]))
            down2, c_2 = self.second_conv_lstm_cell(down2, c_2[0], c_2[1])
            c_2 = [down2, c_2]

            down3 = self.down3(down2)
            if self.concat_x:
                down3 = torch.cat([down3, downs[3]], 1)
            
            if seq_idx == 0:
                c_3 = self.third_conv_lstm_cell.init_hidden(batch_size=inputs.shape[0], shape=(down3.shape[-2], down3.shape[-1]))
            down3, c_3 = self.third_conv_lstm_cell(down3, c_3[0], c_3[1])
            c_3 = [down3, c_3]
            
            down4 = self.down4(down3)
            if self.concat_x:
                down4 = torch.cat([down4, downs[4]], 1)

            if self.more_layers > 0:
                prevs = [down4]
                for kk, d in enumerate(self.more_downs):
                    # print(prevs[-1].size())
                    out = d(prevs[-1])
                    if self.concat_x:
                        out = torch.cat([out,  downs[kk + 5]], 1)

                    prevs.append(out)

                up_ = self.more_ups[-1](prevs[-1], prevs[-2])
                for idx in range(self.more_layers - 1):
                    l = self.more_ups[self.more - idx - 2]
                    up_= l(up_, prevs[self.more - idx - 2])
            else:
                up_= down4

            up4= self.up4(up_, down3)

            up3= self.up3(up4, down2)

            if seq_idx == 0:
                c_4 = self.fourth_conv_lstm_cell.init_hidden(batch_size=inputs.shape[0], shape=(up3.shape[-2], up3.shape[-1]))
            up3, c_4 = self.fourth_conv_lstm_cell(up3, c_4[0], c_4[1])
            c_4 = [up3, c_4]

            up2= self.up2(up3, down1)

            if seq_idx == 0:
                c_5 = self.fifth_conv_lstm_cell.init_hidden(batch_size=inputs.shape[0], shape=(up2.shape[-2], up2.shape[-1]))
            up2, c_5 = self.fifth_conv_lstm_cell(up2, c_5[0], c_5[1])
            c_5 = [up2, c_5]

            up1= self.up1(up2, in64)
            res.append(self.final(up1)[:, None])
        return torch.cat(res, 1)

class NewunetUp(nn.Module):
    def __init__(self, out_size, upsample_mode, need_bias, pad, same_num_filt=False, kernel_size=3):
        super(NewunetUp, self).__init__()

        num_filt = out_size if same_num_filt else out_size * 2
        if upsample_mode == 'deconv':
            self.up= nn.ConvTranspose2d(num_filt, out_size, 4, stride=2, padding=1)
            self.conv= unetConv2(out_size * 2, out_size, None, need_bias, pad)
        elif upsample_mode=='bilinear' or upsample_mode=='nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                    conv(num_filt, out_size, kernel_size, bias=need_bias, pad=pad))
            # self.conv= unetConv2(out_size * 2, out_size, None, need_bias, pad)
        else:
            assert False

    def forward(self, inputs1, inputs2):
        in1_up= self.up(inputs1)
        
        if (inputs2.size(2) != in1_up.size(2)) or (inputs2.size(3) != in1_up.size(3)):
            diff2 = (inputs2.size(2) - in1_up.size(2)) // 2 
            diff3 = (inputs2.size(3) - in1_up.size(3)) // 2 
            inputs2_ = inputs2[:, :, diff2 : diff2 + in1_up.size(2), diff3 : diff3 + in1_up.size(3)]
        else:
            inputs2_ = inputs2

        # output= self.conv(torch.cat([in1_up, inputs2_], 1))
        output = torch.cat([in1_up, inputs2_], 1)
        return output

class DumpyModel(nn.Module):
    '''
        upsample_mode in ['deconv', 'nearest', 'bilinear']
        pad in ['zero', 'replication', 'none']
    '''
    def __init__(self, num_input_channels=3, num_output_channels=3, 
                       feature_scale=4, more_layers=0, concat_x=False,
                       upsample_mode='deconv', pad='zero', norm_layer='in', last_act='sigmoid', need_bias=True, downsample_mode='max'):
        super(DumpyModel, self).__init__()

        filters = [64, 128, 256, 512, 1024]
        filters = [x // feature_scale for x in filters]

        self.start = unetConv2(num_input_channels, 1, norm_layer, need_bias, pad)

    def forward(self, all_inputs):
        # seq_len = all_inputs.shape[1]
        # res = []
        # for seq_idx in range(seq_len):
        #     res.append(self.start(all_inputs[:,seq_idx])[:,None])
        # res = torch.cat(res, dim=1)

        res = self.start(all_inputs)
        # print("Model Inp shape:", all_inputs.shape)
        # print("Model Out shape:", res.shape)
        return res