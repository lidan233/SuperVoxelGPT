#!/usr/bin/env python3
"""Voxelization + supervoxel centers -> the shape tokens the generator is trained on.

Purpose
    The shape generator does not see geometry. It predicts a sequence of codebook indices, one
    per supervoxel, which the autoencoder's decoder turns back into a mesh. This runs the
    encoder half to produce those indices for one object, so a single asset can be used as a
    training corpus of its own.

Input
    A `.vxz` dual-grid voxelization, the matching `*_supervoxel_center.ply`, and the released
    supervoxel autoencoder (`stage2SupervoxelVaeEncoder.pt` + its config).

Output
    One npz with `indices` (N,) int64 and `cvt_points` (N,3) float32, the two keys the token
    loaders read. N is the supervoxel count, which varies per object by an order of magnitude --
    that variation is the compression.

Key idea
    Centers and voxelization must come from the same lineage. Both are derived from one mesh
    through one normalization; pairing a voxelization with centers computed from a different
    variant shifts them relative to each other, and nothing downstream reports it.
"""
import argparse, json, os, sys, warnings
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description="Encode one object into supervoxel tokens.")
    ap.add_argument("--vxz", required=True)
    ap.add_argument("--centers", required=True, help="*_supervoxel_center.ply")
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runtime", default=os.environ.get("T2S_RUNTIME"),
                    help="directory holding o_voxel/flex_gemm (prebuilt extensions)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--center-scale", type=float, default=256.0,
                    help="frame the centers PLY is written in; d_cvt_pycvt emits 0-256")
    args = ap.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    runtime = args.runtime or os.path.join(repo, "train", "vendor", "runtime")
    sys.path.insert(0, runtime)
    sys.path.insert(0, os.path.dirname(repo))
    pkg = os.path.basename(repo)

    import torch, trimesh
    # TF32 off before any model is built, the same contract inference applies. Measured on the
    # bundled example this encoder does not actually need it -- 2,394 tokens, byte-identical
    # either way, because the codebook's quantization step is far wider than the drift TF32
    # introduces. It is set anyway: the stage 1 encoder next door *is* affected, and a stage
    # whose reproducibility depends on which caller happened to set a global is not reproducible.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import o_voxel
    from importlib import import_module
    sp = import_module(f"{pkg}.train.vendor.trellis2.modules.sparse")
    shape_vae = import_module(f"{pkg}.train.stage2TextSaliency2Shape.vae")

    cfg = json.load(open(os.path.join(args.ckpts, "stage2SupervoxelVaeConfig.json")))
    spec = dict(cfg["models"]["encoder"])
    margs = dict(spec.get("args", {}))
    margs["use_fp16"] = False       # fp32 weights; autocast picks the compute dtype per block
    model = getattr(shape_vae, spec["name"])(**margs)
    sd = torch.load(os.path.join(args.ckpts, "stage2SupervoxelVaeEncoder.pt"), map_location="cpu")
    sd = sd.get("state_dict", sd)
    result = model.load_state_dict(sd, strict=False)
    if result.missing_keys or result.unexpected_keys:
        warnings.warn(
            "shape encoder checkpoint loaded with strict=False; "
            f"{len(result.missing_keys)} missing keys {result.missing_keys[:5]}, "
            f"{len(result.unexpected_keys)} unexpected keys {result.unexpected_keys[:5]}. "
            "Encoded tokens may be invalid.", RuntimeWarning)
    if hasattr(model, "set_resolution"):
        model.set_resolution(args.resolution)
    model = model.to(args.device).eval()

    coords, attrs = o_voxel.io.read_vxz(args.vxz, num_threads=4)
    coords = coords if torch.is_tensor(coords) else torch.as_tensor(coords)
    verts = attrs["vertices"] if torch.is_tensor(attrs["vertices"]) else torch.as_tensor(attrs["vertices"])
    inter = attrs["intersected"] if torch.is_tensor(attrs["intersected"]) else torch.as_tensor(attrs["intersected"])

    raw = np.asarray(trimesh.load(args.centers, process=False).vertices, np.float32)
    # The CVT step writes centers in the 0-256 density-grid frame (see pycvt.write_centers_ply).
    # The model works in [-0.5, 0.5], and inference rounds through
    # float16 on the way -- so the same rounding is applied here, or the tokens differ from the ones
    # generation produces for the identical object.
    centers = (raw / float(args.center_scale) - 0.5).astype(np.float16).astype(np.float32)
    print("[tokens] %d active voxels, %d centers  (raw [%.1f, %.1f] -> [%.3f, %.3f])"
          % (coords.shape[0], len(centers), raw.min(), raw.max(),
             centers.min(), centers.max()), flush=True)

    # One object is batch 0 throughout: the sparse tensors carry an explicit batch column, and
    # the VarLenTensor's default layout is the single slice covering every center.
    b = torch.zeros((coords.shape[0], 1), dtype=torch.int32)
    c4 = torch.cat([b, coords.int()], 1).to(args.device)
    vert_t = sp.SparseTensor(verts.to(torch.float32).to(args.device) / 255.0, c4)
    # `intersected` is a 3-bit mask packed into one byte, not a scalar: the loader unpacks it into
    # one channel per axis. Passing the byte through would hand the encoder a 1-channel feature
    # where it expects 3, and the concatenation in forward() would be the wrong width.
    it = inter.to(args.device)
    inter_t = vert_t.replace(torch.cat([it % 2, it // 2 % 2, it // 4 % 2], dim=-1).bool())
    cvt_t = sp.VarLenTensor(torch.as_tensor(centers, device=args.device))

    with torch.no_grad():
        out = model(vert_t, inter_t, cvt_t)
    idx = out["indices"].flatten().cpu().numpy().astype(np.int64)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, indices=idx, cvt_points=centers.astype(np.float32))
    print("[tokens] %d tokens -> %s" % (len(idx), args.out))
    print("         index range [%d, %d], unique %d" % (idx.min(), idx.max(), len(np.unique(idx))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
