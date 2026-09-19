#!/usr/bin/env python3
"""Render a dual-contoured mesh without the artifacts the extraction leaves behind."""
import argparse
import os

import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

_CTX = None


def _ctx():
    global _CTX
    if _CTX is None:
        _CTX = dr.RasterizeCudaContext()
    return _CTX


def _persp(fovy, n=0.05, f=10.0):
    t = 1.0 / np.tan(fovy / 2)
    return np.array([[t, 0, 0, 0], [0, t, 0, 0],
                     [0, 0, (f + n) / (n - f), 2 * f * n / (n - f)],
                     [0, 0, -1, 0]], np.float32)


def _cam(az, el, rad, ctr, fovy):
    az, el = np.radians(az), np.radians(el)
    eye = ctr + rad * np.array([np.cos(el) * np.sin(az), np.sin(el),
                                np.cos(el) * np.cos(az)], np.float32)
    fwd = ctr - eye
    fwd /= np.linalg.norm(fwd)
    up = np.array([0, 1, 0], np.float32)
    r = np.cross(fwd, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, fwd)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = r
    m[1, :3] = u
    m[2, :3] = -fwd
    m[:3, 3] = -m[:3, :3] @ eye
    return _persp(fovy) @ m, eye


def _face_normals(v, f):
    return torch.nn.functional.normalize(
        torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=1), dim=1)


def _vertex_normals(v, f):
    """Area-weighted vertex normals. Only meaningful once the winding is consistent — see the
    module docstring for why this speckles on raw dual-contoured output."""
    fn = torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=1)
    n = torch.zeros_like(v)
    n.index_add_(0, f[:, 0], fn)
    n.index_add_(0, f[:, 1], fn)
    n.index_add_(0, f[:, 2], fn)
    return torch.nn.functional.normalize(n, dim=1)


def render(v, f, az=35.0, el=12.0, ctr=None, rad=None, res=640,
           shading="gray", smooth=False, two_sided=True, fovy=34 * np.pi / 180):
    """One image. v/f are cuda tensors. The defaults are the dual-contouring-safe pair."""
    dev = v.device
    if ctr is None or rad is None:
        vv = v.detach().cpu().numpy()
        c = (vv.max(0) + vv.min(0)) / 2
        ext = float(np.linalg.norm(vv.max(0) - vv.min(0)))
        ctr = c.astype(np.float32) if ctr is None else ctr
        # At fovy 34 degrees a whole object needs dist >= (ext/2)/tan(fovy/2) ~ 1.64*ext; 1.8 leaves margin.
        rad = ext * 1.8 if rad is None else rad
    mvp, eye = _cam(az, el, rad, np.asarray(ctr, np.float32), fovy)
    vh = torch.cat([v, torch.ones_like(v[:, :1])], 1)
    vc = (vh @ torch.from_numpy(mvp).to(dev).T)[None]
    rast, _ = dr.rasterize(_ctx(), vc, f.int(), (res, res))
    wp, _ = dr.interpolate(v[None], rast, f.int())

    if smooth:
        wn, _ = dr.interpolate(_vertex_normals(v, f.long())[None], rast, f.int())
        nn = torch.nn.functional.normalize(wn[0], dim=-1)
    else:
        tid = rast[0, ..., 3].long() - 1
        fn = _face_normals(v, f.long())
        nn = torch.zeros(res, res, 3, device=dev)
        msk = tid >= 0
        nn[msk] = fn[tid[msk]]

    view = torch.nn.functional.normalize(torch.tensor(eye, device=dev) - wp[0], dim=-1)
    if two_sided:
        # sign() is 0 exactly edge-on; treat that degenerate case as front-facing.
        s = torch.sign((nn * view).sum(-1, keepdim=True))
        nn = nn * torch.where(s == 0, torch.ones_like(s), s)

    alpha = (rast[0, ..., 3] > 0).cpu().numpy()
    if shading == "normal":
        img = (nn * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
    else:
        # Two lights from opposing sides: a single light leaves half of a faceted surface flat
        # dark, which reads as a hole in exactly the place a real hole would be.
        L1 = torch.tensor([0.4, 0.7, 0.55], device=dev)
        L1 /= L1.norm()
        L2 = torch.tensor([-0.4, 0.3, -0.5], device=dev)
        L2 /= L2.norm()
        sh = (nn @ L1).clamp(0) * 0.7 + (nn @ L2).clamp(0) * 0.25 + 0.18
        spec = ((nn * view).sum(-1, keepdim=True).clamp(0)) ** 20 * 0.25
        img = (sh.unsqueeze(-1) * torch.tensor([0.76, 0.77, 0.79], device=dev)
               + spec).clamp(0, 1).cpu().numpy()
    img[~alpha] = 1.0
    # nvdiffrast returns row 0 at the bottom (NDC +y grows with row index); PNG and imageio
    # treat row 0 as the top, so the image is flipped on the way out.
    return (img * 255).astype(np.uint8)[::-1]


def render_views(mesh_path, out_dir, views=((30, 15), (150, 15), (270, 15)),
                 res=640, shading="gray", smooth=False):
    """Render several views of one mesh. Returns the paths written."""
    import imageio
    os.makedirs(out_dir, exist_ok=True)
    m = trimesh.load(mesh_path, process=False)
    v = torch.from_numpy(np.asarray(m.vertices, np.float32)).cuda()
    f = torch.from_numpy(np.asarray(m.faces, np.int32)).cuda()
    vv = np.asarray(m.vertices)
    ctr = ((vv.max(0) + vv.min(0)) / 2).astype(np.float32)
    rad = float(np.linalg.norm(vv.max(0) - vv.min(0)) * 1.8)
    base = os.path.splitext(os.path.basename(mesh_path))[0]
    out = []
    for az, el in views:
        img = render(v, f, az, el, ctr, rad, res=res, shading=shading, smooth=smooth)
        p = os.path.join(out_dir, f"{base}_az{int(az)}.png")
        imageio.imwrite(p, img)
        out.append(p)
    return out


def contact_sheet(rows, out_png, pad=8):
    """Stack per-mesh view strips into one sheet, so meshes are compared by looking across."""
    import imageio
    imgs = [[np.asarray(imageio.imread(p)) for p in r] for r in rows if r]
    if not imgs:
        return None
    h, w = imgs[0][0].shape[:2]
    cols = max(len(r) for r in imgs)
    sheet = np.full((len(imgs) * (h + pad) - pad, cols * (w + pad) - pad, 3), 255, np.uint8)
    for i, row in enumerate(imgs):
        for j, im in enumerate(row):
            sheet[i * (h + pad):i * (h + pad) + h, j * (w + pad):j * (w + pad) + w] = im[..., :3]
    imageio.imwrite(out_png, sheet)
    return out_png


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render dual-contoured meshes.")
    ap.add_argument("target", help="a mesh file, or a directory of them")
    ap.add_argument("--out", default=None, help="output directory (default: <target>/renders)")
    ap.add_argument("--res", type=int, default=640)
    ap.add_argument("--shading", choices=("gray", "normal"), default="gray")
    ap.add_argument("--smooth", action="store_true",
                    help="average vertex normals; speckles unless the winding is consistent")
    ap.add_argument("--views", type=int, default=3, help="evenly spaced azimuths")
    ap.add_argument("--sheet", default=None, help="also write a contact sheet here")
    args = ap.parse_args(argv)

    exts = (".ply", ".obj", ".glb", ".off")
    if os.path.isdir(args.target):
        meshes = [os.path.join(args.target, p) for p in sorted(os.listdir(args.target))
                  if p.lower().endswith(exts)]
    else:
        meshes = [args.target]
    if not meshes:
        raise SystemExit(f"no mesh files under {args.target}")

    out_dir = args.out or os.path.join(
        args.target if os.path.isdir(args.target) else os.path.dirname(args.target) or ".",
        "renders")
    views = tuple((360.0 * i / args.views + 30.0, 15.0) for i in range(args.views))

    rows = []
    for m in meshes:
        paths = render_views(m, out_dir, views=views, res=args.res,
                             shading=args.shading, smooth=args.smooth)
        rows.append(paths)
        print(f"{os.path.basename(m)} -> {len(paths)} views", flush=True)
    if args.sheet:
        contact_sheet(rows, args.sheet)
        print(f"contact sheet -> {args.sheet}", flush=True)


if __name__ == "__main__":
    main()
