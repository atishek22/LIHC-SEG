# -*- coding: utf-8 -*-

import torch.nn as nn

from ..config import WeightInitType
from ..utils import model_init


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator
    """

    def __init__(self,
                 input_channels: int,
                 ndf: int = 64,
                 num_layers: int = 3,
                 normalisation=nn.BatchNorm2d,

                 weight_init_type: WeightInitType = WeightInitType.Normal,
                 **kwargs,
                 ):
        """
        parameters:
            input_channels      --- number of channels in the input image
            ndf                 --- number of channels in the discriminator
            num_layers          --- number of layers conv layers in the
                                    discriminator
            normalisation
        """
        super(PatchDiscriminator, self).__init__()
        use_bias = normalisation == nn.InstanceNorm2d

        model = [
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]

        filters = ndf
        filters_prev_layer = ndf
        for n in range(1, num_layers):
            filters_prev_layer = filters
            filters = min(2 ** n, 8) * ndf
            model += [
                nn.Conv2d(filters_prev_layer, filters, kernel_size=4,
                          stride=2, padding=1, bias=use_bias),
                normalisation(filters),
                nn.LeakyReLU(0.2, True),
            ]

        filters_prev_layer = filters
        filters = min(2 ** num_layers, 8) * ndf

        model += [
            nn.Conv2d(filters_prev_layer, filters, kernel_size=4, stride=1,
                      padding=1, bias=use_bias),
            normalisation(filters),
            nn.LeakyReLU(0.2, True)
        ]

        model += [
            nn.Conv2d(filters, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*model)
        gpus = kwargs['gpu'] if kwargs.get('gpu') is not None else []
        model_init(self.model, weight_init_type, gpus, **kwargs)

    def forward(self, input):
        return self.model(input)
