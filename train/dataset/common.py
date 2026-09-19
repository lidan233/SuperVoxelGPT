"""Contracts every dataset in this package shares.
"""
import sys
from contextlib import contextmanager

import numpy as np


@contextmanager
def _numpy_core_compat():
    """Let pickle resolve numpy._core when running against an older numpy.

    Arrays saved by numpy >= 2.0 embed 'numpy._core.multiarray' in pickled object arrays; older
    numpy exposes it as 'numpy.core'. The alias is installed only for the duration of the load,
    because leaving it in place breaks C extensions that look the module up themselves.
    """
    needs = "numpy._core" not in sys.modules
    if needs:
        sys.modules["numpy._core"] = np.core
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
    try:
        yield
    finally:
        if needs:
            sys.modules.pop("numpy._core", None)
            sys.modules.pop("numpy._core.multiarray", None)


def load_npz(path):
    with _numpy_core_compat():
        return np.load(path, allow_pickle=True)


def load_view_features(path_stem, suffix, key="features_518"):
    """Load one object's view features, whichever container the extractor wrote.

    The extractor writes `.npz` under a named key; a bare `.npy` is accepted too because that is
    what an ad-hoc extraction produces. Reading an npz with plain np.load returns the archive
    rather than the array, and the shape error that follows names neither the file nor the key,
    so the two cases are separated here instead of at the call site.
    """
    path = path_stem + suffix
    if suffix.endswith(".npz"):
        with _numpy_core_compat():
            with np.load(path, allow_pickle=True) as z:
                if key not in z:
                    raise KeyError(f"{path} has no '{key}'; it holds {list(z.files)}")
                return z[key]
    return np.load(path)


def _part1by2(x):
    """Spread the low 10 bits of x so two zero bits sit between consecutive bits."""
    x = x & 0x3FF
    x = (x | (x << 16)) & 0x30000FF
    x = (x | (x << 8)) & 0x300F00F
    x = (x | (x << 4)) & 0x30C30C3
    x = (x | (x << 2)) & 0x9249249
    return x


def morton_order(coords, grid=64):
    """Return the permutation that puts centers in Morton order.

    coords are in [-0.5, 0.5]. They are quantized to the grid before interleaving so that a
    sub-voxel difference cannot change the order, and ties inside a cell are broken by the
    full-precision coordinate so the result is deterministic rather than dependent on input order.
    """
    q = np.clip(((coords.astype(np.float64) + 0.5) * (grid - 1)).round().astype(np.int64), 0, grid - 1)
    key = (_part1by2(q[:, 2]) << 2) | (_part1by2(q[:, 1]) << 1) | _part1by2(q[:, 0])
    fine = coords[:, 2].astype(np.float64) * 4 + coords[:, 1] * 2 + coords[:, 0]
    return np.lexsort((fine, key))


def jitter_coords(coords, jitter_vox, grid=64, rng=None):
    """Perturb centers by a fraction of a voxel. Training only — see the module docstring."""
    if jitter_vox <= 0:
        return coords
    rng = rng if rng is not None else np.random
    noise = rng.uniform(-jitter_vox, jitter_vox, coords.shape).astype(np.float32) / grid
    return coords + noise


# Padding coordinates carry a sentinel far outside the valid range so the coordinate encoder can
# recognize them; zero would be a legal position at the center of the object.
PAD_COORD = -100.0
