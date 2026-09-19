"""Annealed soft-CVT supervoxel placement, self-contained.

Turns a 64^3 saliency field into supervoxel centers -- the coordinates the generator writes
against. The same helpers run on both sides, so the corpus and inference place centers alike.

    from pycvt import pycvt_from_field, load_kernel
    cvt = pycvt_from_field(idx, val)      # idx[N,3] int, val[N] float -> [M,3] float32
"""
import os
import numpy as np
import torch
from scipy.spatial import cKDTree

GRID = 64          # resolution of the saliency field
CVT_RES = 256      # density grid the annealing iterates on
CVT_T = 0.05       # saliency floor; below it a cell takes the coarsest spacing
CVT_K = 3          # spacing ratio between the coarsest and finest cells
TAUS = np.geomspace(32.0, 4.0, 120)   # annealing temperature schedule
LAM = 0.02         # per-step pull back toward the initial seeds

_EXT = None


def load_kernel(path=None):
    """Load the soft-CVT kernel, preferring a prebuilt copy over compiling one.

    `-fmad=false` is what makes the result identical across machines, so it belongs to the
    build rather than to the caller. A prebuilt `soft_cvt_clean.so` sitting next to this file
    was compiled with it and is used as-is; that is what lets a machine without a working nvcc
    run inference at all. Falling back to a JIT build keeps a new GPU architecture working.
    """
    global _EXT
    if _EXT is not None:
        return _EXT
    here = os.path.dirname(os.path.abspath(__file__))
    prebuilt = os.path.join(here, "soft_cvt_clean.so")
    if os.path.exists(prebuilt):
        import importlib.util
        spec = importlib.util.spec_from_file_location("soft_cvt_clean", prebuilt)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            _EXT = mod
            return _EXT
        except Exception as e:
            # Built for another architecture or another torch ABI: say so, then build.
            print(f"[pycvt] prebuilt kernel unusable ({type(e).__name__}), compiling: {e}",
                  flush=True)
    from torch.utils.cpp_extension import load
    path = path or os.environ.get("T2S_CVT_KERNEL", os.path.join(here, "soft_cvt_kernel.cu"))
    _EXT = load(name="soft_cvt_clean", sources=[path],
                extra_cuda_cflags=["-fmad=false", "-O3"], verbose=False)
    return _EXT


# ---------------------------------------------------------------- size field and budget
def size_field(x, t=CVT_T, k=CVT_K):
    """saliency -> inverse fifth power of the target spacing, for the density field."""
    xc = np.clip(np.asarray(x, np.float64), t, 1.0)
    f = (xc - t) / (1.0 - t) * (1.0 - k) + k
    return 1.0 / np.power(f, 5)


def token_budget(x, t=CVT_T, k=CVT_K) -> float:
    """N = sum 1/f^3 -- how many supervoxels the saliency field asks for.

    Sensitive to the absolute scale of the saliency, not just its shape: lifting the whole field
    inflates N. So do not min-max stretch the saliency after projecting it onto voxels.
    """
    xc = np.clip(x, t, 1.0)
    f = ((xc - t) / (1.0 - t)) * (1.0 - k) + k
    return float(np.sum(1.0 / np.power(f, 3)))


def upsample_density(vol64: np.ndarray, res: int = CVT_RES) -> np.ndarray:
    """Nearest-neighbour lift of the 64^3 field onto the density grid."""
    t = torch.from_numpy(vol64).float().cuda()
    nz = torch.nonzero(t > 0, as_tuple=False)
    if len(nz) == 0:
        return np.zeros((res,) * 3, np.float32)
    vals = t[t > 0].cpu().numpy()
    low = (nz.float().cpu().numpy() + 0.5) / GRID
    mask = torch.nn.functional.interpolate((t > 0).float()[None, None], size=(res,) * 3,
                                           mode="nearest")[0, 0].bool()
    hi = torch.nonzero(mask, as_tuple=False).cpu().numpy()
    _, nn = cKDTree(low).query((hi + 0.5) / res, k=1, workers=-1)
    out = np.zeros((res,) * 3, np.float32)
    out[hi[:, 0], hi[:, 1], hi[:, 2]] = vals[nn]
    return out


# ---------------------------------------------------------------- seed selection
def pca_anchor(coords: np.ndarray) -> int:
    """Start point for farthest-point sampling: the extreme along the principal axis, with the
    sign fixed by the third moment so it does not flip between the two ends."""
    c = coords.astype(np.float64)
    w = np.full(len(c), 1.0 / len(c))
    mu = (c * w[:, None]).sum(0)
    cc = c - mu
    _, vec = np.linalg.eigh((cc * w[:, None]).T @ cc)
    p1 = cc @ vec[:, -1]
    if (w * (p1 ** 3)).sum() < 0:
        p1 = -p1
    return int(np.argmin(p1))


def farthest_point_sample(coords: np.ndarray, start: int, n: int) -> np.ndarray:
    """CPU float64 with smallest-index tie-breaks."""
    pts = coords.astype(np.float64)
    m = len(pts)
    if m <= n:
        sel = list(range(m)) + [i % m for i in range(n - m)]
        return np.array(sel[:n], np.int64)
    sel = np.empty(n, np.int64)
    d2 = np.full(m, np.inf)
    cur = int(start)
    for i in range(n):
        sel[i] = cur
        d2 = np.minimum(d2, ((pts - pts[cur]) ** 2).sum(1))
        cur = int(np.flatnonzero(d2 == d2.max())[0])
    return sel


# ---------------------------------------------------------------- ordering
def _p12(x):
    x = x & 0x3ff
    x = (x | (x << 16)) & 0x30000FF
    x = (x | (x << 8)) & 0x300F00F
    x = (x | (x << 4)) & 0x30C30C3
    x = (x | (x << 2)) & 0x9249249
    return x


def morton_order(coords: np.ndarray) -> np.ndarray:
    """z-order over one-voxel cells, ties broken by full-precision lexicographic order.

    The generator was trained on this order. Sort once and carry that one array through
    generation and decoding: re-sorting or undoing it pairs tokens with the wrong coordinates.
    """
    q = np.clip(((coords.astype(np.float64) + 0.5) * (GRID - 1)).round().astype(np.int64), 0, GRID - 1)
    key = (_p12(q[:, 2]) << 2) | (_p12(q[:, 1]) << 1) | _p12(q[:, 0])
    fine = coords[:, 2].astype(np.float64) * 4 + coords[:, 1] * 2 + coords[:, 0]
    return np.lexsort((fine, key))


# ---------------------------------------------------------------- main entry point
def pycvt_from_field(idx: np.ndarray, val: np.ndarray, kernel_path=None,
                       device="cuda", raw=False):
    """64^3 saliency -> supervoxel centers.

    Args
      idx : [N,3] integer coordinates of the occupied voxels (0..63)
      val : [N]   their saliency, at its native scale (do not min-max stretch it; see token_budget)
      raw : True  -> raw float32 centers in the 0-256 frame, the way the corpus PLYs store them
            False -> [-0.5, 0.5], rounded through float16, the way the generator consumes them

    The two are equivalent: reading a PLY back applies /256-0.5 and float16, and f16(f16(x)) is
    f16(x). Write files with raw=True; feed the model with the default.
    """
    ext = load_kernel(kernel_path)
    idx = np.asarray(idx).astype(np.int64)
    field = np.zeros((GRID,) * 3, np.float32)
    field[idx[:, 0], idx[:, 1], idx[:, 2]] = np.asarray(val, np.float16).astype(np.float32)

    occ = np.ascontiguousarray(np.argwhere(field > 0)).astype(np.int64)
    n_seed = max(1, int(round(token_budget(field[field > 0]))))

    sf = field.copy()
    nzm = np.where(sf > 0)
    sf[nzm] = size_field(sf[nzm])
    dens = upsample_density(sf)

    si = farthest_point_sample(occ, pca_anchor(occ), min(n_seed, len(occ)))
    s01 = np.stack([(occ[si][:, 2] + 0.5) / GRID,
                    (occ[si][:, 1] + 0.5) / GRID,
                    (occ[si][:, 0] + 0.5) / GRID], 1).astype(np.float32)

    nz = np.ascontiguousarray(np.argwhere(dens > 0))
    cells = torch.tensor(nz + 0.5, dtype=torch.float32, device=device)
    cell2 = torch.tensor(2 * nz + 1, dtype=torch.int64, device=device)
    rho = torch.tensor(dens[nz[:, 0], nz[:, 1], nz[:, 2]], dtype=torch.float64, device=device)

    s = torch.tensor(s01[:, [2, 1, 0]], dtype=torch.float32, device=device) * 256.0
    s0 = s
    for tau in TAUS:
        s = ext.soft_cvt(cells, cell2, rho, s, float(tau), 1, 8.0)
        s = ((1.0 - LAM) * s.double() + LAM * s0.double()).float()
    torch.cuda.synchronize()

    cent = s.cpu().numpy().astype(np.float32)
    seed_pts = occ[si].astype(np.float32) * 4.0 + 2.0          # integer seeds, centered in the 256 grid
    order = morton_order((seed_pts / CVT_RES - 0.5).astype(np.float32))
    pts = cent[order]
    if raw:
        return pts                                              # 0-256 frame, not yet float16
    return (pts / 256.0 - 0.5).astype(np.float16).astype(np.float32)


def load_sample(path):
    """Read sample.npz -> (idx, val)."""
    z = np.load(path)
    return z["idx"].astype(np.int64), z["val"].astype(np.float32)


def write_centers_ply(path, cvt):
    """Write the centers as a PLY in the 0-256 frame, as the corpus CVT files use.

    Passing a raw=True result reproduces the corpus PLY byte for byte; passing [-0.5, 0.5]
    converts back first, which is the same value but has been through float16 once.
    """
    a = np.asarray(cvt, np.float32)
    pts = a if np.abs(a).max() > 1.0 else ((a.astype(np.float64) + 0.5) * 256.0).astype(np.float32)
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n"
           "property float x\nproperty float y\nproperty float z\nend_header\n" % len(pts))
    with open(path, "wb") as f:
        f.write(hdr.encode())
        f.write(np.ascontiguousarray(pts).tobytes())


if __name__ == "__main__":
    import sys
    idx, val = load_sample(sys.argv[1])
    cvt = pycvt_from_field(idx, val, raw=True)
    print(f"occupied {len(idx)} -> supervoxels {len(cvt)}")
    if len(sys.argv) > 2:
        write_centers_ply(sys.argv[2], cvt)
        print(f"wrote {sys.argv[2]}")
