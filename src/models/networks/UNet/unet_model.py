from .unet_parts import *
""" Full assembly of the parts to form the complete network """
"""
Reference: https://github.com/milesial/Pytorch-UNet
"""


class UNet(nn.Module):
    def __init__(
        self,
        n_channels,
        bilinear=True,
        name='UNet'                     # name for wandb
    ):
        super().__init__()

        self.n_channels = n_channels
        self.bilinear = bilinear
        self.name = name

        factor = 2 if bilinear else 1

        self.InConv = DoubleConv(n_channels, 64)

        # '- n_channels' for concatenation later of skip connections
        self.Down1 = Down(64, 128 - n_channels)
        self.Down2 = Down(128, 256 - n_channels)
        self.Down3 = Down(256, 512 - n_channels)
        self.Down4 = Down(512, 1024 // factor - n_channels)

        self.Up1 = Up(1024, 512 // factor, bilinear)
        self.Up2 = Up(512, 256 // factor, bilinear)
        self.Up3 = Up(256, 128 // factor, bilinear)
        self.Up4 = Up(128, 64, bilinear)

        self.OutConv = OutConv(64, n_channels)

    def forward(self, inputs):
        AdaptSize = nn.AvgPool2d(2, 2)
        inputs_skip = [inputs]
        for _ in range(4):
            inputs_skip.append(AdaptSize(inputs_skip[-1]))

        in_conv = torch.cat([self.InConv(inputs), inputs_skip[0]], 1)

        down1 = torch.cat([self.Down1(in_conv), inputs_skip[1]], 1)
        down2 = torch.cat([self.Down2(down1), inputs_skip[2]], 1)
        down3 = torch.cat([self.Down3(down2), inputs_skip[3]], 1)
        down4 = torch.cat([self.Down4(down3), inputs_skip[4]], 1)

        up1 = self.Up1(down4, down3)
        up2 = self.Up2(up1, down2)
        up3 = self.Up3(up2, down1)
        up4 = self.Up4(up3, in_conv)

        logits = self.OutConv(up4)

        return logits
