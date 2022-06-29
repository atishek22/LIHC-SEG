# -*- coding: utf-8 *-

import torch
import torch.nn as nn

from ..utils import model_init
from ..config import WeightInitType


# ******* GENERATORS *********

class ResNetGenerator(nn.Module):
    """
    ResNet generator: Consist of Residual Blocks sandwitched between
    downsampling and upsampling blocks.

    Reference: CycleGAN, Neural Style Transfer
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 ngf: int = 64,  # number of generator filters
                 normalisation=nn.BatchNorm2d,
                 dropout: bool = False,
                 num_blocks: int = 6,
                 padding_type: str = "reflect",
                 num_sampling_blocks: int = 2,
                 weight_init_type: WeightInitType = WeightInitType.Normal,
                 **kwargs,
                 ):
        """
        parameters:
            input_channels: int     --- number of channels in input image
            output_channels: int    --- number of channels in output image
            ngf: int                --- number of generator filters
            normalisation           --- Type of normalisation
            dropout: bool           --- dropout in residual blocks
            num_blocks: int         --- number of residual blocks
            padding_type: str       --- which padding to use in residual blocks
                                            ("reflect", "zero")
            num_sampling_blocks:int --- number of upsampling and downsampling
                                            blocks
        """
        assert(num_blocks >= 0)

        super(ResNetGenerator, self).__init__()

        # use bias only in the case of instance normalisation
        use_bias = normalisation == nn.InstanceNorm2d

        # build the model

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, ngf, kernel_size=7,
                      padding=0, bias=use_bias),
            normalisation(ngf),
            nn.ReLU(True)
        ]

        # Down Sampling Block
        for i in range(num_sampling_blocks):
            in_channels = (2 ** i) * ngf
            out_channels = input_channels * 2
            model += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          stride=2, padding=1, bias=use_bias),
                normalisation(out_channels),
                nn.ReLU(True),
            ]

        # ResidualBlocks
        dims = 2 ** num_sampling_blocks * ngf
        for i in range(num_blocks):
            model += [ResidualBlock(dims,
                                    padding_type=padding_type,
                                    normailisation=normalisation,
                                    dropout=dropout,
                                    use_bias=use_bias)]

        # upsampling blocks
        for i in range(num_sampling_blocks):
            in_channels = 2 ** (num_sampling_blocks - i) * ngf
            out_channels = in_channels // 2

            model += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                                   stride=2, padding=1, output_padding=1,
                                   bias=use_bias),
                normalisation(out_channels),
                nn.ReLU(True),
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        gpus = kwargs['gpu'] if kwargs.get('gpu') is not None else []
        model_init(self.model, weight_init_type, gpus, **kwargs)

    def forward(self, input):
        return self.model(input)


class UNetGenerator(nn.Module):
    """
    UNet generator with skip connections

    Reference: CycleGAN, StainGAN, UNet
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 sampling_block: int,
                 ngf: int = 64,
                 normalisation=nn.BatchNorm2d,
                 dropout: bool = False,
                 weight_init_type: WeightInitType = WeightInitType.Normal,
                 **kwargs,
                 ):
        """
        parameters:
            input_channels: int     --- number of channels in input image
            output_channels: int    --- number of channels in output image
            sampling_blocks: int    --- number of downsampling and upsampling
                                            blocks
            ngf: int                --- number of generator filters
            normalisation           --- Type of normalisation
            dropout: bool           --- dropout in residual blocks
        """
        super(UNetGenerator, self).__init__()

        unet_block = UNetResBlock(ngf * 8, ngf * 8, input_channels=None,
                                  submodule=None, normalisation=normalisation,
                                  innermost=True)
        for i in range(sampling_block - 5):
            unet_block = UNetResBlock(ngf * 8, ngf * 8, input_channels=None,
                                      submodule=unet_block,
                                      normalisation=normalisation,
                                      dropout=dropout)

        unet_block = UNetResBlock(ngf * 4, ngf * 8,
                                  input_channels=None, submodule=unet_block,
                                  normalisation=normalisation)

        unet_block = UNetResBlock(ngf * 2, ngf * 4,
                                  input_channels=None, submodule=unet_block,
                                  normalisation=normalisation)

        unet_block = UNetResBlock(ngf, ngf * 2,
                                  input_channels=None, submodule=unet_block,
                                  normalisation=normalisation)

        self.model = UNetResBlock(output_channels, ngf,
                                  input_channels=input_channels,
                                  submodule=unet_block, outermost=True,
                                  normalisation=normalisation)

        gpus = kwargs['gpu'] if kwargs.get('gpu') is not None else []
        model_init(self.model, weight_init_type, gpus, **kwargs)

    def forward(self, input):
        return self.model(input)

# ******* HELPERS *********


class ResidualBlock(nn.Module):
    """
    Residual Conv block with skip connections
    """

    def __init__(self,
                 dim: int,
                 padding_type: str,
                 normailisation,
                 dropout: bool,
                 use_bias: bool):

        super(ResidualBlock, self).__init__()

        conv_block = []

        padding, p = self.get_padding(padding_type)

        if padding is not None:
            conv_block += padding

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            normailisation(dim),
            nn.ReLU(True),
        ]

        if(dropout):
            conv_block += [nn.Dropout(0.5)]

        padding, p = self.get_padding(padding_type)

        if padding is not None:
            conv_block += padding

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            normailisation(dim),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def get_padding(self, padding_type):
        if padding_type == 'reflect':
            return [nn.ReflectionPad2d(1)], 0
        elif padding_type == 'zero':
            return None, 1
        else:
            raise ValueError(f"Invalid padding type: {padding_type}")

    def forward(self, input):
        output = input + self.conv_block(input)  # residual connection
        return output


class UNetResBlock(nn.Module):
    """
    This block is used to build the UNet model from the inner layers
    to the outer layers
    """

    def __init__(self,
                 outer_channels: int,
                 inner_channels: int,
                 input_channels: int = None,
                 submodule=None,
                 outermost: bool = False,
                 innermost: bool = False,
                 normalisation=nn.BatchNorm2d,
                 dropout: bool = False
                 ):
        """
        parameters:
            outer_channels: int     --- number of channels in the outer conv
                                        layers
            inner_channels: int     --- number of channels in the inner conv
                                        layers
            input_channels: int     --- number of channels in the input image
            submodule: UNetResBlock --- inner submodule (previously defined)
            outermost: bool
            innermost: bool
            normalisation
            dropout: bool
        """
        super(UNetResBlock, self).__init__()
        self.outermost = outermost
        use_bias = normalisation == nn.InstanceNorm2d

        if input_channels is None:
            input_channels = outer_channels

        down_conv = nn.Conv2d(input_channels, inner_channels,
                              kernel_size=4, stride=2,
                              padding=1, bias=use_bias)
        down_relu = nn.LeakyReLU(0.2, True)
        down_norm = normalisation(inner_channels)

        up_relu = nn.ReLU(True)
        up_norm = normalisation(outer_channels)

        if outermost:
            up_conv = nn.ConvTranspose2d(
                inner_channels * 2, outer_channels,
                kernel_size=4, stride=2, padding=1)

            down = [down_conv]
            up = [up_relu, up_conv, nn.Tanh()]
            model = down + [submodule] + up

        elif innermost:
            up_conv = nn.ConvTranspose2d(inner_channels, outer_channels,
                                         kernel_size=4, stride=2, padding=1,
                                         bias=use_bias)

            down = [down_relu, down_conv]
            up = [up_relu, up_conv, up_norm]
            model = down + up

        else:
            up_conv = nn.ConvTranspose2d(inner_channels * 2, outer_channels,
                                         kernel_size=4, stride=2, padding=1,
                                         bias=use_bias)
            down = [down_relu, down_conv, down_norm]
            up = [up_relu, up_conv, up_norm]

            if dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.outermost:
            return self.model(input)
        else:
            return torch.cat([input, self.model(input)], 1)
