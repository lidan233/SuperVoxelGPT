#!/usr/bin/env python3
"""Look at the intermediate representations: the 64^3 saliency field and the centers.
"""
import os
import sys
import numpy as np


def _load_centers(ply):
    """Read the centers, in whichever convention they were written, as [-0.5, 0.5]."""
    import trimesh
    p = np.asarray(trimesh.load(ply, process=False).vertices, np.float64)
    return (p / 256.0 - 0.5) if np.abs(p).max() > 1.0 else p


from matplotlib.colors import LinearSegmentedColormap as _LSC

# The ramp and the knee the paper figures use for saliency. The remap lifts the low end:
# most cells sit well below the 99th percentile, and on a plain linear ramp the picture is
# one flat colour with a few bright specks.
_COOLWARM = _LSC.from_list("cw", [
    (0.00, (0.229, 0.299, 0.754)), (0.25, (0.522, 0.647, 0.914)),
    (0.50, (0.865, 0.865, 0.865)), (0.75, (0.937, 0.533, 0.471)),
    (1.00, (0.706, 0.016, 0.150))])


def _remap(s, midpoint=0.25):
    s = np.clip(s, 0, 1)
    return np.where(s < midpoint, s / midpoint * 0.5,
                    0.5 + (s - midpoint) / (1 - midpoint) * 0.5)


def plot_fields(sample_npz, centers_ply=None, out_png="fields.png", title=None, elev=15, azim=30):
    """Draw the 64^3 saliency beside the supervoxel centers. Returns out_png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(sample_npz)
    idx = z["idx"].astype(float)
    val = z["val"].astype(float)

    n = 2 if centers_ply else 1
    fig = plt.figure(figsize=(8.2 * n, 6.4))

    ax = fig.add_subplot(1, n, 1, projection="3d")
    o = np.argsort(idx[:, 1])                      # back to front, so near points draw last
    p = ax.scatter(idx[o, 0], idx[o, 2], idx[o, 1], c=_remap(val[o]), cmap=_COOLWARM,
                   s=6, marker="s", linewidths=0, depthshade=True, vmin=0, vmax=1)
    # ASCII labels: a machine running this may have no CJK font installed.
    ax.set_title(f"64^3 saliency  ({len(idx)} occupied, median {np.median(val):.2f})", fontsize=12)
    fig.colorbar(p, ax=ax, shrink=0.55, pad=0.02)
    _fmt(ax, idx, elev, azim)

    if centers_ply:
        c = _load_centers(centers_ply)
        cg = (c + 0.5) * 63.0                       # into the same voxel frame as the field
        ax2 = fig.add_subplot(1, n, 2, projection="3d")
        o2 = np.argsort(cg[:, 1])
        ax2.scatter(cg[o2, 0], cg[o2, 2], cg[o2, 1], s=4, c="C0",
                    linewidths=0, depthshade=True)
        ax2.set_title(f"supervoxel centers  (N={len(cg)})", fontsize=12)
        _fmt(ax2, idx, elev, azim)

    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _fmt(ax, ref, elev, azim):
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    lo, hi = float(ref.min()), float(ref.max())
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def paint_saliency_on_mesh(sample_npz, mesh_ply, out_ply, cmap="inferno"):
    """Paint the 64^3 saliency onto the mesh: each vertex takes its nearest voxel's value.

    Easier to read than the scatter plot, because it shows which geometric features the saliency
    actually landed on — edges, spokes, the rims of holes. Writes a vertex-colored mesh; open it
    in whatever viewer you prefer, `vis/show.py` among them.
    """
    import matplotlib.cm as cm
    import trimesh
    from scipy.spatial import cKDTree

    z = np.load(sample_npz)
    idx = z["idx"].astype(np.float64); val = z["val"].astype(np.float64)
    m = trimesh.load(mesh_ply, process=False)
    V = np.asarray(m.vertices, np.float64); F = np.asarray(m.faces)

    # The mesh is in [-0.5, 0.5] and the field is on the voxel grid; match them first.
    vv = (V - (V.max(0) + V.min(0)) / 2)
    vv = vv / (np.abs(vv).max() or 1.0) * 0.5
    vq = (vv + 0.5) * 63.0
    _, nn = cKDTree(idx + 0.5).query(vq, k=1)
    sal = val[nn]
    rgb = cm.get_cmap(cmap)(np.clip(sal, 0, 1))[:, :3]

    mm = trimesh.Trimesh(V, F, process=False)
    mm.visual.vertex_colors = (np.c_[rgb, np.ones(len(rgb))] * 255).astype(np.uint8)
    mm.export(out_ply)
    return out_ply


def montage(pngs, out_png, cols=None):
    """Tile several PNGs into one image, in a row or wrapped at `cols`."""
    from PIL import Image
    ims = [Image.open(p) for p in pngs]
    cols = cols or len(ims)
    rows = [ims[i:i + cols] for i in range(0, len(ims), cols)]
    W = max(sum(i.width for i in r) for r in rows)
    H = sum(max(i.height for i in r) for r in rows)
    c = Image.new("RGB", (W, H), "white")
    y = 0
    for r in rows:
        x = 0
        for i in r:
            c.paste(i, (x, y)); x += i.width
        y += max(i.height for i in r)
    c.save(out_png)
    return out_png


if __name__ == "__main__":
    W = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(W, "fields.png")
    s = os.path.join(W, "sample.npz")
    c = os.path.join(W, "sample_supervoxel_center.ply")
    if not os.path.exists(s):
        sys.exit(f"{s} is missing — run stage 1 first")
    plot_fields(s, c if os.path.exists(c) else None, out,
                title=os.path.basename(os.path.abspath(W)))
    print(f"→ {out}", flush=True)
