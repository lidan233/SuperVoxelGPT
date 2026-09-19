#!/usr/bin/env python3
"""Interactive viewer (polyscope): orbit, measure, toggle layers."""
import os
import sys
import numpy as np


def _norm(v):
    v = np.asarray(v, np.float64)
    c = (v.max(0) + v.min(0)) / 2
    s = float(np.abs(v - c).max()) or 1.0
    return (v - c) / s * 0.9


def show_workdir(W):
    import polyscope as ps
    import trimesh
    ps.init()
    ps.set_up_dir("y_up")

    sp = os.path.join(W, "sample.npz")
    if os.path.exists(sp):
        z = np.load(sp)
        idx = z["idx"].astype(np.float64); val = z["val"].astype(np.float64)
        pc = ps.register_point_cloud("saliency", _norm(idx + 0.5), radius=0.004, enabled=False)
        pc.add_scalar_quantity("saliency", val, cmap="viridis", enabled=True, vminmax=(0, 1))

    cp = os.path.join(W, "sample_supervoxel_center.ply")
    if os.path.exists(cp):
        c = np.asarray(trimesh.load(cp, process=False).vertices, np.float64)
        if np.abs(c).max() > 1.0:
            c = c / 256.0 - 0.5
        ps.register_point_cloud("centers", _norm(c), radius=0.003, enabled=False)

    ps.show()


def show_files(paths):
    import polyscope as ps
    import trimesh
    ps.init(); ps.set_up_dir("y_up")
    for i, p in enumerate(paths):
        g = trimesh.load(p, process=False)
        name = os.path.basename(p)
        if hasattr(g, "faces") and len(g.faces):
            ps.register_surface_mesh(name, _norm(g.vertices), np.asarray(g.faces), smooth_shade=False)
        else:
            ps.register_point_cloud(name, _norm(np.asarray(g.vertices)), radius=0.004)
    ps.show()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    if os.path.isdir(args[0]):
        show_workdir(args[0])
    else:
        show_files(args)
