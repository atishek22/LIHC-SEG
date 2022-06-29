# -*- coding: utf-8 -*-
from typing import Tuple

import torch
import torch.nn as nn


class CycleGan(nn.Module):
    def __init__(self,
                 generators: Tuple[nn.Module, nn.Module],
                 discriminators: Tuple[nn.Module, nn.Module],
                 ):
        self.generators = generators
        self.discriminators = discriminators

    def forward(self, input: torch.Tensor):
        pass

    def train(self, input: torch.Tensor, target: torch.Tensor):
        pass
