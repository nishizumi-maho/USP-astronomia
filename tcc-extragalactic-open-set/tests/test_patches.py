import numpy as np

from tccastro.preprocess.normalize import robust_normalize
from tccastro.preprocess.patches import stack_patches


def test_robust_normalize_shape() -> None:
    patch = np.ones((3, 4, 4), dtype=np.float32)
    out = robust_normalize(patch)
    assert out.shape == patch.shape


def test_stack_patches() -> None:
    patches = [np.random.rand(3, 4, 4).astype(np.float32) for _ in range(2)]
    stacked = stack_patches(patches)
    assert stacked.shape == (2, 3, 4, 4)
    assert stacked.dtype == np.float32
