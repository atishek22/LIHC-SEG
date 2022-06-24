# -*- coding: utf-8 -*-

from pathlib import Path
import pyvips


def generate_tiles(source_file: Path,
                   dest_folder: Path,
                   tile_size: int = 256,
                   overlap: int = 0,
                   depth: int = 2):
    """
    Generates DeepZoom file from the whole slide image
    """
    image = pyvips.Image.new_from_file(str(source_file))
    dest = Path(dest_folder, source_file.stem)
    image.dzsave(str(dest),
                 overlap=overlap, tile_size=tile_size, depth=depth)
