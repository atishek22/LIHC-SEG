# -*- coding: utf-8 -*-

import pyvips
from pathlib import Path
from typing import Tuple
import fast
import cv2 as cv
import openslide
import numpy as np

from .tiles import generate_tiles


def generate_patch_fast(source_file: Path, size: Tuple[int, int], level: int):
    """
    pyFAST:
    Generate patches for a whole slide image
    """
    # Read in the
    importer = fast.WholeSlideImageImporter.create(str(source_file))

    tissue_segmentation = fast.TissueSegmentation.create().connect(importer)

    patch_generator = fast.PatchGenerator.create(
        *size, level=level
    ).connect(0, importer).connect(1, tissue_segmentation)

    patches = []
    for patch in fast.DataStream(patch_generator):
        patches.append(patch)
    return patches


def generate_patch_tiles(source_dir: Path,
                         dest_dir: Path,
                         size: Tuple[int, int] = (512, 512),
                         threshold: float = 50.0
                         ):
    """
    Generate traning patches from DeepZoom files using greylevel
    binary thresholding
    """
    dest_dir.mkdir(exist_ok=True)
    s = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    for f in source_dir.glob("*.jpeg"):
        try:
            s[:, :, :] = np.array(cv.imread(str(f)))[:, :, :3]
        except ValueError:
            continue
        img = cv.cvtColor(s, cv.COLOR_RGB2GRAY)
        blur = cv.GaussianBlur(img, (5, 5), 0)
        ret3, th3 = cv.threshold(blur, 200, 255, cv.THRESH_BINARY)

        total = size[0] * size[1]
        cells = cv.countNonZero(th3)

        perc = 100 - ((cells / total) * 100)
        if perc > threshold:
            dest = Path(dest_dir, f.name)
            cv.imwrite(str(dest), s)


def generate_patch_svs_thresholding(source_file: Path,
                                    size: Tuple[int, int],
                                    level: int,
                                    threshold: float = 50.0):
    """
    Generate patches for a whole slide image using greylevel binary
    thresholding
    """
    slide = openslide.OpenSlide(source_file)
    x, y = slide.dimensions

    s = np.zeros((1024, 1024, 3), dtype=np.uint8)
    s_x = x // size[0]
    s_y = y // size[1]

    for i in range(s_x):
        for j in range(s_y):
            s[:, :, :] = np.array(slide.read_region(
                (i * size[0], j * size[1]), 0, size))[:, :, :3]
            img = cv.cvtColor(s, cv.COLOR_RGB2GRAY)
            blur = cv.GaussianBlur(img, (5, 5), 0)
            ret3, th3 = cv.threshold(blur, 200, 255, cv.THRESH_BINARY)

            total = size[0] * size[1]
            cells = cv.countNonZero(th3)

            perc = 100 - ((cells / total) * 100)
            if perc > threshold:
                yield s


def generate_patches(source_file: Path,
                     dest_dir: Path,
                     pyramid: int = 0,
                     size: Tuple[int, int] = (512, 512),
                     threshold: float = 50.0,
                     overlap: int = 0,
                     depth: int = 2,
                     seg_patches: bool = True) -> Path:
    """
        Generate Training patches from svs files
    """
    dest_base = Path(dest_dir, source_file.stem)
    dest_base.mkdir(parents=True, exist_ok=True)
    dest_tiles = Path(dest_base, "tiles")

    print("Generating tiles...")
    generate_tiles(source_file, dest_tiles, size[0], overlap, depth)
    tiles_dir = Path(dest_base, "tiles_files", str(pyramid))
    dest_files = Path(dest_base, "patches")

    print("Generating training patches...")
    generate_patch_tiles(
        tiles_dir,
        dest_files,
        size,
        threshold
    )

    seg_files = None

    if seg_patches:
        print("Generating segmentation patches")
        source = Path(source_file.parent, source_file.stem + "_whole.tif")
        dest = Path(dest_base, "seg")
        generate_tiles(
            source,
            dest,
            size[0],
            overlap,
            depth
        )
        seg_files = Path(dest_base, "seg_files", str(pyramid))

    return dest_files, seg_files
