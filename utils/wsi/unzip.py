# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List
import zipfile
from tqdm import tqdm


def extract(source: Path, dest: Path) -> List[Path]:
    """
    Extracts all the zip files in the source dir to the dest dir
    and returns a list of the extracted files
    """
    paths: List[Path] = []
    for f in tqdm(source.glob("*.zip")):
        with zipfile.ZipFile(f, 'r') as zip_ref:
            path = Path(dest, f.name.rstrip(".zip"))
            zip_ref.extractall(path)
            paths.append(path)

    return paths
