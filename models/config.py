from enum import Enum


class NormalisationType(Enum):
    Batch = 1
    Instance = 2
    Identity = 3


class WeightInitType(Enum):
    Normal = 1
    Xavier = 2
    Kaiming = 3
    Orthogonal = 4


class LRSchedulerType(Enum):
    Linear = 1
    Step = 2
    Plateau = 3
    Cosine = 4
