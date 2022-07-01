# -*- coding: utf-8 -*-
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import itertools

from .loss import GANLoss, LossType


class CycleGan(nn.Module):
    def __init__(self,
                 generators: Tuple[nn.Module, nn.Module],
                 discriminators: Tuple[nn.Module, nn.Module],
                 optimisers: Tuple[optim.Optimizer,
                                   optim.Optimizer] = None,
                 device=None,
                 training: bool = False
                 ):
        self.generators = generators
        self.discriminators = discriminators

        if training:
            self.optimisers = (
                optimisers[0](itertools.chain(
                    self.generators[0].parameters(),
                    self.generators[1].parameters()
                )),
                optimisers[1](itertools.chain(
                    self.discriminators[0].parameters(),
                    self.discriminators[1].parameters()
                ))
            )

            self.critGAN = GANLoss(LossType.Normal).to(device)
            self.critCycle = nn.L1Loss()
            self.critIdt = nn.L1Loss()

    def forward(self, input: List[torch.Tensor]):
        self.fake_range = self.generators[0](input[0])
        self.rec_dom = self.generators[1](self.fake_range)
        self.fake_dom = self.generators[0](input[1])
        self.rec_range = self.generators[0](self.fake_dom)

    def train(self, input: List[torch.Tensor]):
        self.forward(input)

        for disc in self.discriminators:
            for param in disc.parameters():
                param.requires_grad = False

        self.optimisers[0].zero_grad()
