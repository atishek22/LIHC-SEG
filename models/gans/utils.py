# -*- coding: utf-8 -*-

import torch.nn as nn

from .config import NormalisationType


class Identity(nn.Module):
    def forward(self, x):
        return x


def normalisation_layer(type: NormalisationType) -> nn.Module:
    if type == NormalisationType.Batch:
        return nn.BatchNorm2d
    elif type == NormalisationType.Instance:
        return nn.InstanceNorm2d
    elif type == NormalisationType.Identity:
        return Identity
    else:
        raise ValueError("Invalid Normalisation Type")
