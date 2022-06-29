# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Tuple

import torchvision.io as io
import torch.utils.data as data


class UnalignedDataset(data.Dataset):
    def __init__(self,
                 data_dirs: Tuple[Path, Path],
                 transforms=None):
        self.data_dirs = data_dirs
        self.image_paths = (
            sorted(list(data_dirs[0].glob("*.jpeg"))),
            sorted(list(data_dirs[0].glob("*.jpeg")))
        )
        self.lens = (
            len(self.image_paths[0]),
            len(self.image_paths[1])
        )

    def __len__(self):
        return max(self.lens)

    def __getitem__(self, idx):
        images = [
            io.read_image(
                str(self.image_paths[i][idx % self.lens[i]]),
                io.ImageReadMode.RGB
            )
            for i in range(2)
        ]
        return tuple(images)
