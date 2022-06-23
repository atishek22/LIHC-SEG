# -*- coding: utf-8 -*-

# from pathlib import Path
from utils.wsi.patches import generate_patch

if __name__ == "__main__":
    # paths = wsi.unzip.extract(Path("data"), Path("data", "wsi"))
    # print(paths)
    # svs_file = list(paths[0].glob("*.svs"))[0]
    svs_file = "/home/atishek/source/dl/lihc/data/wsi/Training_phase_1_001/01_01_0083.svs"
    patches = generate_patch(svs_file, (1024, 1024), 0)
    print(patches[0].shape)
    print(len(patches))
