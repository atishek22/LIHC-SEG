from enum import Enum


class NormalisationType(Enum):
    Batch = 1
    Instance = 2
    Identity = 3


class LossType(Enum):
    Normal = 1
    LSGAN = 2           # Least square error
    WGAN_GP = 3         # Wasserstein GAN + Gradient Penalty


class GradientPenaltyType(Enum):
    Real = 1
    Fake = 2
    Mixed = 3


class WeightInitType(Enum):
    Normal = 1
    Xavier = 2
    Kaiming = 3
    Orthogonal = 4


class LRPolicy(Enum):
    Linear = 1
