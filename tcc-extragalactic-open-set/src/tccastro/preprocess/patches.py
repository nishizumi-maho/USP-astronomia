from __future__ import annotations

from typing import Iterable

import numpy as np

from tccastro.preprocess.normalize import robust_normalize


def stack_patches(patches: Iterable[np.ndarray]) -> np.ndarray:
    patches_list = [robust_normalize(p) for p in patches]
    return np.stack(patches_list, axis=0).astype(np.float32)
