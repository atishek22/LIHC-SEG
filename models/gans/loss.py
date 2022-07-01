# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from enum import Enum


class LossType(Enum):
    Normal = 1
    LSGAN = 2           # Least square error
    WGAN_GP = 3         # Wasserstein GAN + Gradient Penalty


class GradientPenaltyType(Enum):
    Real = 1
    Fake = 2
    Mixed = 3


def gradient_penalty(discriminator,
                     real_data: torch.Tensor,
                     fake_data: torch.Tensor,
                     device,
                     type: GradientPenaltyType = GradientPenaltyType.Mixed,
                     constant=1.0,
                     lambda_gp=10.0):
    """
    Calculates the gradient penalty for WGAN-GP loss

    gradient_pen =
        lambda * (norm_2( D(alpha * real + (1-alpha * fake)) ) - 1) ** 2

    See: Lipschitz continuity, 1-Lipschitz continuous functions
         WGAN loss, weight clipping (https://arxiv.org/pdf/1701.07875.pdf)
         Improved WGAN Training(https://arxiv.org/pdf/1704.00028.pdf)
    """
    if lambda_gp > 0.0:
        if type == GradientPenaltyType.Real:
            interpolatesv = real_data
        elif type == GradientPenaltyType.Fake:
            interpolatesv = fake_data
        elif type == GradientPenaltyType.Mixed:
            # generate a random value for each instance to interpolate fake
            # and real data
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            # expand the random value to be applied all elements of that
            # instance
            alpha = alpha.expand(
                real_data.shape[0],
                real_data.nelement() // real_data.shape[0]
            ).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + (1 - alpha) * fake_data

        else:
            raise ValueError("Invalid value for GradientPenaltyType")

        interpolatesv.requires_grad_(True)
        discriminator_interpolates = discriminator(interpolatesv)
        gradients = torch.autograd.grad(
            outputs=discriminator_interpolates,
            inputs=interpolatesv,
            grad_outputs=torch.ones(
                discriminator_interpolates.shape).to(device),
            create_graph=True,
            retain_graph=True,
        )
        # flatten
        gradients = gradients[0].view(real_data.shape[0], -1)
        gradient_pen = (((gradients + 1e-16).norm(2, dim=1) -
                        constant) ** 2).mean() * lambda_gp
        return gradient_pen, gradients
    else:
        return 0.0, None


class GANLoss(nn.Module):
    """
    Wrapper around the GAN loss function

    Supports Simple GAN, LSGAN and WGAN-GP Losses
    """

    def __init__(self,
                 loss_type: LossType,
                 target_real_label=1.0,
                 target_fake_label=0.0):

        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.type = type
        if type == LossType.LSGAN:
            self.loss = nn.MSELoss
        elif type == LossType.Normal:
            self.loss = nn.BCEWithLogitsLoss        # Sigmoid  + BCELoss
        elif type == LossType.WGAN_GP:
            self.loss = None
        else:
            raise ValueError("Invalid Loss Type")

    def get_target_tensor(self,
                          prediction: torch.Tensor,
                          is_real: bool) -> torch.Tensor:

        target_tensor: torch.Tensor
        if is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label

        return target_tensor.expand_as(prediction)

    def __call__(self, prediction: torch.Tensor, is_real: bool):
        if self.type in [LossType.LSGAN, LossType.Normal]:
            target_tensor = self.get_target_tensor(prediction, is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.type == LossType.WGAN_GP:
            if is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
