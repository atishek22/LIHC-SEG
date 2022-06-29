# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from .config import NormalisationType, WeightInitType


class Identity(nn.Module):
    def forward(self, x):
        return x


def normalisation_layer(type: NormalisationType) -> nn.Module:
    """
    Get the normalisation layer from the type
    """
    if type == NormalisationType.Batch:
        return nn.BatchNorm2d
    elif type == NormalisationType.Instance:
        return nn.InstanceNorm2d
    elif type == NormalisationType.Identity:
        return Identity
    else:
        raise ValueError("Invalid Normalisation Type")


def weight_init(model, type: WeightInitType, **kwargs):
    """
    Initalise the model weights based on the init function type
    """

    def init_func(layer):
        classname = layer.__class__.__name__
        if hasattr(layer, "weight") and (classname.find('Conv') != -1 or
                                         classname.find('Linear') != -1):
            if type == WeightInitType.Normal:
                mean = kwargs['mean'] if kwargs.get(
                    'mean') is not None else 0.0
                std = kwargs['std'] if kwargs.get('std') is not None else 0.02
                nn.init.normal_(layer.weight.data, mean, std)
            elif type == WeightInitType.Xavier:
                gain = kwargs['gain'] if kwargs.get(
                    'gain') is not None else 0.02
                nn.init.xavier_normal_(layer.weight.data, gain)
            elif type == WeightInitType.Kaiming:
                nn.init.kaiming_normal_(layer.weight.data, a=0, mode='fan_in')
            elif type == WeightInitType.Orthogonal:
                gain = kwargs['gain'] if kwargs.get(
                    'gain') is not None else 0.02
                nn.init.orthogonal_(layer.weight.data, gain)
            else:
                raise ValueError("Invalid WeightInitType")

        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)

        elif classname.find('BatchNorm') != -1:
            mean = kwargs['mean'] if kwargs.get(
                'mean') is not None else 0.0
            std = kwargs['std'] if kwargs.get('std') is not None else 0.02
            nn.init.normal_(layer.weight.data, mean, std)
            nn.init.constant_(layer.bias.data, 0.0)

    print(f"Initalising weights({type})...")
    model.apply(init_func)


def model_init(model,
               weight_init_type: WeightInitType = WeightInitType.Normal,
               gpus=[],
               **kwargs):
    if len(gpus) > 0:
        assert(torch.cuda.is_available())
        model.to(gpus[0])
        model = nn.DataParallel(model, gpus)  # distribute to multiple gpus
    weight_init(model, weight_init_type, **kwargs)
    return model
