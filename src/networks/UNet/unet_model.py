from .unet_parts import *
""" Full assembly of the parts to form the complete network """
"""
Reference: https://github.com/milesial/Pytorch-UNet
"""


class UNet(nn.Module):

    def __init__(
        self,
        n_input_channels: int,
        n_output_channels: int,
        initial_channels: int,
        bilinear: bool = True,
        name: str = 'UNet'                     # name for wandb
    ):
        super().__init__()

        n_channels = n_input_channels
        n_classes = n_output_channels

        self.name = name
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        ic = initial_channels

        self.inc = DoubleConv(n_channels, ic)
        self.down1 = Down(ic, ic * 2)
        self.down2 = Down(ic * 2, ic * 4)
        self.down3 = Down(ic * 4, ic * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(ic * 8, ic * 16 // factor)
        self.up1 = Up(ic * 16, ic * 8// factor, bilinear)
        self.up2 = Up(ic * 8, ic * 4 // factor, bilinear)
        self.up3 = Up(ic * 4, ic * 2 // factor, bilinear)
        self.up4 = Up(ic * 2, ic, bilinear)
        self.outc = OutConv(ic, n_classes)

    def forward(self, inputs):
        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
