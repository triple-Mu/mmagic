from pathlib import Path

import cv2
import numpy as np

roots = [
    Path('/home/ubuntu/workspace/github/openmmlab/mmagic/data/SIDD/train/input'
         ),
    Path(
        '/home/ubuntu/workspace/github/openmmlab/mmagic/data/SIDD/train/target'
    ),
    Path('/home/ubuntu/workspace/github/openmmlab/mmagic/data/SIDD/val/input'),
    Path(
        '/home/ubuntu/workspace/github/openmmlab/mmagic/data/SIDD/val/target'),
]

for root in roots:
    for i in root.iterdir():
        img = cv2.imread(str(i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        np.save(i.with_suffix('.npy'), img)
