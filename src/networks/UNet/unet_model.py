from .unet_parts import *
""" Full assembly of the parts to form the complete network """
"""
Reference: https://github.com/milesial/Pytorch-UNet
"""


class UNet(nn.Module):

    class Config:

        def __init__(
            self,
            n_input_channels: int,
            n_output_channels: int,
            initial_channels: int,
            bilinear: bool = True,
            name: str = 'UNet', # name for wandb
            output_activation: str = 'none',
        ):
            self.n_input_channels = n_input_channels
            self.n_output_channels = n_output_channels
            self.initial_channels = initial_channels
            self.bilinear = bilinear
            self.name = name
            self.output_activation = output_activation

        @staticmethod
        def from_config(config: dict):
            return UNet.Config(
                n_input_channels=config['n_input_channels'],
                n_output_channels=config['n_output_channels'],
                initial_channels=config['initial_channels'],
                bilinear=config['bilinear'],
                name=config['name'] if 'name' in config else 'UNet',
                output_activation=config['output_activation'] if 'output_activation' in config else 'none'
            )

        def __iter__(self):
            for attr, value in self.__dict__.items():
                yield attr, value
        
        def get_printout(self):
            return f"""
                N input channels:  {self.n_input_channels}
                N output channels: {self.n_output_channels}
                Initial Channels:  {self.initial_channels}
                Bilinear:          {self.bilinear}
                Name:              {self.name}
                Output Activation: {self.output_activation}
            """


    @staticmethod
    def __get_output_activation(output_activation: str):
        if output_activation == 'none':
            return lambda x: x
        elif output_activation == 'relu':
            return nn.ReLU()
        elif output_activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            RuntimeError(f"invalid output activation given {output_activation}")


    def __init__(
        self,
        config: Config
    ):
        super().__init__()
        self.config = config

        n_channels = config.n_input_channels
        n_classes = config.n_output_channels

        self.name = config.name
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = config.bilinear

        ic = config.initial_channels

        self.inc = DoubleConv(n_channels, ic)
        self.down1 = Down(ic, ic * 2)
        self.down2 = Down(ic * 2, ic * 4)
        self.down3 = Down(ic * 4, ic * 8)
        factor = 2 if config.bilinear else 1
        self.down4 = Down(ic * 8, ic * 16 // factor)
        self.up1 = Up(ic * 16, ic * 8// factor, config.bilinear)
        self.up2 = Up(ic * 8, ic * 4 // factor, config.bilinear)
        self.up3 = Up(ic * 4, ic * 2 // factor, config.bilinear)
        self.up4 = Up(ic * 2, ic, config.bilinear)
        self.outc = OutConv(ic, n_classes)
        self.out_activation = self.__get_output_activation(config.output_activation)

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
        out = self.out_activation(logits)
        return out
