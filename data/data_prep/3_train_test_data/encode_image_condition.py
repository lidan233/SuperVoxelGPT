#!/usr/bin/env python3
"""Multi-view renders -> DINOv2 patch features, the image conditioning.

Purpose
    The image branch of both stages is trained on precomputed vision-tower features, not on
    images. This renders one object from several viewpoints and encodes each view, so a single
    object can stand in for a corpus entry.

Input
    A mesh (PLY/GLB) and the DINOv2 tower. Views are placed on a ring at a fixed elevation,
    which is what the released renders did; `--num-views 1` reproduces the single-view setting.

Output
    One npz with `features_518` of shape (V, P, D) -- V views, P patch tokens, D channels.
    The loaders select one view at random per sample, so more views is strictly more
    conditioning variety for the same object.

Key idea
    The features are patch tokens, not the pooled CLS vector. The generators cross-attend to
    the patch grid, so a pooled vector has the wrong rank and would be silently broadcast.
"""
import argparse, glob, os, sys
import numpy as np


def render_views(mesh_path, n_views, size, elev_deg):
    """Rasterize the mesh from `n_views` azimuths with nvdiffrast, returning uint8 RGB."""
    import torch, trimesh, nvdiffrast.torch as dr
    m = trimesh.load(mesh_path, process=False, force="mesh")
    v = np.asarray(m.vertices, np.float32)
    f = np.asarray(m.faces, np.int32)
    c = (v.min(0) + v.max(0)) / 2.0
    v = v - c
    v = v / max(float(np.abs(v).max()), 1e-9) * 0.8

    n = trimesh.Trimesh(vertices=v, faces=f, process=False).vertex_normals.astype(np.float32)
    dev = "cuda"
    vt = torch.as_tensor(v, device=dev)
    ft = torch.as_tensor(f, device=dev)
    nt = torch.as_tensor(np.ascontiguousarray(n), device=dev)
    ctx = dr.RasterizeCudaContext()

    imgs = []
    elev = np.deg2rad(elev_deg)
    for i in range(n_views):
        az = 2.0 * np.pi * i / n_views
        eye = np.array([np.cos(elev) * np.sin(az), np.sin(elev), np.cos(elev) * np.cos(az)],
                       np.float32) * 2.6
        fwd = -eye / np.linalg.norm(eye)
        up = np.array([0, 1, 0], np.float32)
        right = np.cross(fwd, up); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        R = np.stack([right, up, -fwd], 0)
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = R
        view[:3, 3] = -R @ eye
        fov, near, far = np.deg2rad(40.0), 0.1, 10.0
        t = 1.0 / np.tan(fov / 2)
        proj = np.zeros((4, 4), np.float32)
        proj[0, 0] = t; proj[1, 1] = t
        proj[2, 2] = -(far + near) / (far - near); proj[2, 3] = -2 * far * near / (far - near)
        proj[3, 2] = -1.0
        mvp = torch.as_tensor(proj @ view, device=dev)

        hom = torch.cat([vt, torch.ones_like(vt[:, :1])], 1)
        clip = (hom @ mvp.T).unsqueeze(0)
        rast, _ = dr.rasterize(ctx, clip, ft, resolution=[size, size])
        shade, _ = dr.interpolate(nt.unsqueeze(0), rast, ft)
        shade = torch.nn.functional.normalize(shade, dim=-1)
        lit = (shade[..., 2].abs().clamp(0, 1) * 0.75 + 0.25)
        img = lit.unsqueeze(-1).repeat(1, 1, 1, 3)
        img = torch.where(rast[..., 3:4] > 0, img, torch.ones_like(img))
        a = (img[0].detach().cpu().numpy() * 255).astype(np.uint8)
        # nvdiffrast puts row 0 at the bottom; PNG and PIL treat row 0 as the top.
        imgs.append(a[::-1])
    return imgs


def load_renders(path):
    """Read `view*.png` from a directory, in name order, as uint8 RGB."""
    import imageio.v2 as iio
    ps = sorted(glob.glob(os.path.join(path, "view*.png")))
    if not ps:
        raise SystemExit(f"no view*.png in {path}")
    out = []
    for p in ps:
        a = np.asarray(iio.imread(p))
        out.append(a[..., :3] if a.ndim == 3 and a.shape[-1] == 4 else a)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render views and encode them with DINOv2.")
    ap.add_argument("--mesh", default=None,
                    help="mesh to rasterize; not needed when --renders-dir is given")
    ap.add_argument("--renders-dir", default=None,
                    help="encode these view*.png instead of rasterizing. The corpus rendered the "
                         "original asset through Blender with its materials intact, which the "
                         "rasterizer here cannot do -- see vis/render_glb_views.py")
    ap.add_argument("--out", required=True, help="output npz with features_518")
    ap.add_argument("--save-renders", default=None, help="also write the PNGs here")
    ap.add_argument("--num-views", type=int, default=4)
    ap.add_argument("--size", type=int, default=518, help="render size; DINOv2 was trained at 518")
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--model", default="facebook/dinov2-large")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    if args.renders_dir:
        imgs = load_renders(args.renders_dir)
        print("[image] %d prerendered views from %s" % (len(imgs), args.renders_dir), flush=True)
    elif args.mesh:
        imgs = render_views(args.mesh, args.num_views, args.size, args.elev)
        print("[image] rendered %d views at %d px" % (len(imgs), args.size), flush=True)
    else:
        ap.error("one of --mesh or --renders-dir is required")
    if args.save_renders:
        os.makedirs(args.save_renders, exist_ok=True)
        import imageio.v2 as iio
        for i, a in enumerate(imgs):
            iio.imwrite(os.path.join(args.save_renders, "view%02d.png" % i), a)
        print("[image] renders -> %s" % args.save_renders)

    import torch
    # TF32 off before the vision tower is built, matching inference.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(args.model)
    mdl = AutoModel.from_pretrained(args.model).to(args.device).eval()

    # The processor's own default resizes to 256 and crops to 224, which yields a 16x16 patch
    # grid. The conditioning spec is a 37x37 grid at 518px, and the loaders read the grid edge
    # from the feature count -- so a 224px encode is not a lower-quality variant of the same
    # thing, it is a differently-shaped tensor the model never saw.
    sz = {"height": args.size, "width": args.size}

    feats = []
    with torch.no_grad():
        for a in imgs:
            inp = proc(images=a, return_tensors="pt", size=sz, crop_size=sz).to(args.device)
            h = mdl(**inp).last_hidden_state[0, 1:]     # drop CLS, keep the patch grid
            feats.append(h.cpu().numpy().astype(np.float32))
    F = np.stack(feats)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, features_518=F)
    g = int(round(F.shape[1] ** 0.5))
    print("[image] %s -> features_518 %s (grid %dx%d, dim %d)" % (args.model, F.shape, g, g, F.shape[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
