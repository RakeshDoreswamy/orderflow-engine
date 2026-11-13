"""Golden-image helpers: compare charts by a coarse grayscale thumbnail (24 x 36 cell means) plus aspect ratio, so harmless
anti-aliasing differences pass and real layout/data changes fail. Regenerate with `python scripts/update_golden.py`."""
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

GOLDEN = Path(__file__).resolve().parent / "golden" / "thumbs.npz"
ROWS, COLS = 24, 36


def thumb(path) -> tuple[np.ndarray, float]:
    im = mpimg.imread(str(path)).astype(float)
    if im.ndim == 3:
        im = im[..., :3].mean(axis=2)
    cells = np.array([[blk.mean() for blk in np.array_split(r, COLS, axis=1)] for r in np.array_split(im, ROWS, axis=0)])
    return cells, im.shape[1] / im.shape[0]


def load_golden() -> dict:
    z = np.load(GOLDEN)
    return {k: z[k] for k in z.files}


def distance(a, b) -> float:
    return float(np.abs(a - b).mean())
