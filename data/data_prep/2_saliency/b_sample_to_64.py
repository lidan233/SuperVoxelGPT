#!/usr/bin/env python3
"""Carry the mesh saliency onto the occupancy grid the model actually sees.

Purpose
    Saliency is computed per vertex on the sharp mesh; the budget downstream is decided per voxel on
    the coarse occupancy. Something has to move the field from one to the other, and the move is
    where a whole class of silent corruption lives — a saliency field that is spatially right but
    normalized wrong, or normalized right but sampled on a grid that does not line up with the
    occupancy, both look plausible and both destroy the budget.

Input
    The voxelization of the object, and the per-vertex saliency of the same object.

Output
    One sparse field per object on the coarse grid: occupied cell indices and their pooled saliency,
    still in the raw units the saliency step produced.

    Deliberately not normalized. Normalization is a global percentile map computed over the whole
    corpus, so it cannot be known while a single object is being processed; the next step computes
    it and the step after that applies it.

Key idea
    The occupancy is not a separate voxelization that has to be aligned afterwards — it *is* the
    grid the saliency is sampled on. The voxel coordinates are read straight out of the object's
    voxelization, each voxel center takes the value of its nearest mesh vertex, and only then is the
    result pooled down to the coarse grid. So the sampled field and the occupancy agree cell for
    cell by construction rather than by correction.

    The alternative — voxelizing the mesh independently and aligning the two grids afterwards —
    does not hold: two voxelizers disagree, and on a mesh whose parts are separate they disagree
    badly enough that the two grids share almost no cells.

    Pooling keeps the maximum rather than the mean. A cell is interesting if anything inside it is
    interesting, and averaging is what dilutes a small sharp feature into its smooth surroundings —
    exactly the signal the budget exists to spend on. The sum is also emitted because it answers a
    different question (how much detail is in this cell, not how sharp the sharpest part is) and the
    two need separate normalization, their magnitudes being incomparable.
"""
import argparse, importlib.util, os, sys, types
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree


def load_read_vxz(site):
    """Import o_voxel's vxz reader.

    o_voxel is an external compiled package. If it is importable from the active environment this
    is a plain import; otherwise point --o-voxel-site at its installed directory and the modules are
    loaded from there by path.
    """
    if not site:
        from o_voxel.io.vxz import read_vxz
        return read_vxz
    site = Path(site)
    if "o_voxel" not in sys.modules:
        p = types.ModuleType("o_voxel"); p.__path__ = [str(site)]; sys.modules["o_voxel"] = p
    if "o_voxel.io" not in sys.modules:
        p = types.ModuleType("o_voxel.io"); p.__path__ = [str(site / "io")]; sys.modules["o_voxel.io"] = p
    for nm, path in [("o_voxel._C", next(iter(site.glob("_C*.so")), site / "_C.so")),
                     ("o_voxel.serialize", site / "serialize.py"),
                     ("o_voxel.io.vxz", site / "io" / "vxz.py")]:
        if nm in sys.modules: continue
        spec = importlib.util.spec_from_file_location(nm, path)
        m = importlib.util.module_from_spec(spec); sys.modules[nm] = m; spec.loader.exec_module(m)
    from o_voxel.io.vxz import read_vxz
    return read_vxz


def normalize_vertices(v, scale):
    """Center on the bounding box and scale into [-0.5, 0.5].

    Must reproduce the normalization the voxelizer used, or the vertices and the voxel coordinates
    describe different objects. The scale is marginally below 1 so that the extreme vertices land
    inside the last cell rather than on its boundary.
    """
    mn, mx = v.min(0), v.max(0)
    c = (mn + mx) / 2.0
    s = scale / max(float((mx - mn).max()), 1e-12)
    return np.clip((v - c) * s, -0.5, 0.5)


def pool_down(idx, val, target, how):
    """Group voxels by their coarse cell and reduce, keeping max or sum."""
    code = (idx[:, 0] * target + idx[:, 1]) * target + idx[:, 2]
    order = np.argsort(code, kind="stable")
    cs, vs = code[order], val[order]
    uniq, start = np.unique(cs, return_index=True)
    red = np.maximum.reduceat(vs, start) if how == "max" else np.add.reduceat(vs, start)
    out_idx = np.stack([uniq // (target * target), (uniq // target) % target, uniq % target], 1)
    return out_idx.astype(np.int16), red.astype(np.float32)


def process(stem, read_vxz, args):
    outs = {p: Path(args.out_dir) / p / f"{stem}.npz" for p in args.pools}
    if all(p.exists() and p.stat().st_size > 0 for p in outs.values()):
        return "skip"
    vxz = Path(args.vxz_dir) / f"{stem}.vxz"
    sal = Path(args.saliency_dir) / f"{stem}{args.saliency_suffix}"
    if not vxz.exists() or not sal.exists():
        return "nosrc"

    coords, _ = read_vxz(str(vxz), num_threads=args.threads)
    c = coords.cpu().numpy().astype(np.int64) if hasattr(coords, "cpu") else np.asarray(coords, np.int64)
    if len(c) == 0:
        return "empty"

    z = np.load(sal)
    V = np.asarray(z["vertices"], np.float64)
    S = np.asarray(z[args.saliency_key], np.float32)
    if len(V) == 0 or len(S) != len(V):
        return "empty"

    # Voxel centers in the same normalized frame as the mesh, then nearest vertex wins.
    Vn = normalize_vertices(V, args.norm_scale).astype(np.float32)
    centers = (c.astype(np.float32) + 0.5) / args.grid - 0.5
    _, nn = cKDTree(Vn).query(centers, k=1, workers=args.threads)
    vsal = S[nn].astype(np.float32)

    idx = (c * args.target // args.grid).clip(0, args.target - 1)
    for p in args.pools:
        oi, ov = pool_down(idx, vsal, args.target, p)
        outs[p].parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(outs[p], idx64=oi, val=ov)
    return f"OK v={len(c)}"


# ---------------------------------------------------------------------------
# Production invocation (per shard):
#
#   python b_sample_to_64.py --vxz-dir <vxz dir> --saliency-dir <saliency npz dir> \
#       --out-dir <dense64 dir> --shard-id $i --shard-count 8
#
# Defaults are the released recipe: source grid 1024, target 64, both pools,
# normalization scale 0.99999, and the saliency read from `S_log_gpu`.
#
# `S_log_gpu` is the field to use, and this is not a preference. The saliency
# step also stores `S_final_gpu`, which is stretched per shape; feeding that
# downstream flattens the difference between a plain object and an intricate one,
# which is the entire signal the budget is derived from.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Sample per-vertex saliency onto the occupancy grid.")
    ap.add_argument("--vxz-dir", required=True, help="directory of per-object voxelizations")
    ap.add_argument("--saliency-dir", required=True, help="directory of per-vertex saliency npz")
    ap.add_argument("--out-dir", required=True, help="output root; one subdirectory per pool")
    ap.add_argument("--names-file", help="restrict to these ids, one per line (default: every vxz)")
    ap.add_argument("--shard-id", type=int, default=int(os.environ.get("SHARD_ID", "0")))
    ap.add_argument("--shard-count", type=int, default=int(os.environ.get("SHARD_COUNT", "1")))
    ap.add_argument("--saliency-suffix", default="_mesh_saliency_energy_512.npz",
                    help="suffix appended to an id to find its saliency file")
    ap.add_argument("--saliency-key", default="S_log_gpu",
                    help="npz key holding the per-vertex field; must be a raw one, never a per-shape "
                         "normalized one")
    ap.add_argument("--o-voxel-site", default=os.environ.get("O_VOXEL_SITE", ""),
                    help="o_voxel install directory, if it is not importable from the environment")
    ap.add_argument("--threads", type=int, default=4, help="worker threads for the reader and the KD-tree")
    # Recipe constants: part of the contract with the released assets.
    ap.add_argument("--grid", type=int, default=1024, help="resolution the voxelization was written at")
    ap.add_argument("--target", type=int, default=64, help="resolution of the emitted coarse field")
    ap.add_argument("--norm-scale", type=float, default=0.99999,
                    help="must match the scale the voxelizer normalized with")
    ap.add_argument("--pools", default="max,sum", help="comma-separated: max, sum")
    args = ap.parse_args()
    args.pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    for p in args.pools:
        if p not in ("max", "sum"):
            ap.error(f"unknown pool {p!r}")

    read_vxz = load_read_vxz(args.o_voxel_site)
    if args.names_file:
        ids = [l.strip() for l in open(args.names_file) if l.strip()]
    else:
        ids = sorted(p.stem for p in Path(args.vxz_dir).glob("*.vxz"))
    mine = ids[args.shard_id::args.shard_count]
    print(f"[s{args.shard_id}] {len(mine)}/{len(ids)} ids, pools={args.pools}", flush=True)

    tally = {}
    for i, stem in enumerate(mine, 1):
        try:
            r = process(stem, read_vxz, args)
        except Exception as e:
            r = "EXC:" + repr(e)[:70]
        tally[r.split()[0]] = tally.get(r.split()[0], 0) + 1
        if i % 100 == 0 or not r.startswith(("OK", "skip")):
            print(f"[s{args.shard_id}] {i}/{len(mine)} {stem[:24]} {r}", flush=True)
    print(f"[s{args.shard_id}] DONE {tally}", flush=True)


if __name__ == "__main__":
    main()
