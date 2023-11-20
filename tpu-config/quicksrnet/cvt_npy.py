import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

roots = [
    Path(
        '/home/ubuntu/workspace/github/openmmlab/mmagic/data/tpu_data/train/HR'
    ),
    Path(
        '/home/ubuntu/workspace/github/openmmlab/mmagic/data/tpu_data/train/LR'
    ),
    Path(
        '/home/ubuntu/workspace/github/openmmlab/mmagic/data/tpu_data/val/HR'),
    Path(
        '/home/ubuntu/workspace/github/openmmlab/mmagic/data/tpu_data/val/LR'),
]

save_root = [
    Path('/dev/shm/tpu_data/train/HR'),
    Path('/dev/shm/tpu_data/train/LR'),
    Path('/dev/shm/tpu_data/val/HR'),
    Path('/dev/shm/tpu_data/val/LR'),
]


def path_to_npy(orin_path: Path, dst_path: Path):
    img = cv2.imread(str(orin_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = dst_path / (orin_path.stem + '.npy')
    np.save(dst, img)


for root, sroot in zip(roots, save_root):
    sroot.mkdir(parents=True, exist_ok=True)
    paths = list(root.iterdir())
    bar = tqdm(paths, total=len(paths), desc=f'{root.stem} processing ... ')
    pools = mp.Pool(32)

    for i in paths:
        pools.apply_async(path_to_npy, args=(i, sroot))
        bar.update()

    pools.close()
    pools.join()
