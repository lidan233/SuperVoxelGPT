#!/usr/bin/env python3
"""64-cube saliency field -> the discrete codes stage 1's generator predicts.

Purpose
    Stage 1 does not predict a saliency volume directly. It predicts a sequence of 512 codebook
    indices on an 8-cube lattice, which the saliency autoencoder's decoders turn back into an
    occupancy grid and a saliency field. This runs the encoder half, so one object can serve as
    stage 1 training data the same way `encode_tokens.py` does for stage 2.

Input
    The coarse field written by `2_saliency/b_sample_to_64.py` (`idx64` + `val`), the
    percentile file from `c_global_stats.py`, and the released saliency autoencoder.

Output
    One npz with `indices_flat` (512,) int64 -- the key the stage 1 loaders read.

Key idea
    The values arrive in log units and the cut points are what map them into [0, 1]. On the
    released corpus those are one pass over every object, which is the point: an intricate
    object should ask for more supervoxels than a plain one, and per-shape normalization maps
    every object's most salient region to the top of the range, erasing exactly that
    difference. See data_prep/README.md, "Global statistics".

    A single-object stats file -- which is what the bundled example produces, since the corpus
    one is not released -- is therefore NOT the corpus normalization, and the codes it yields
    are not comparable with the released corpus. Note also that a token count taken from the
    predicted saliency field comes from a field already in [0, 1], which never passes through
    these cut points at all.
"""
import argparse, json, os, sys, warnings
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description="Encode a 64-cube saliency field into stage 1 codes.")
    ap.add_argument("--field", required=True, help="npz with idx64 [M,3] and val [M]")
    ap.add_argument("--stats", required=True, help="percentile npz from c_global_stats.py")
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grid", type=int, default=64)
    args = ap.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    sys.path.insert(0, os.path.dirname(repo))
    pkg = os.path.basename(repo)

    import torch
    # TF32 off before the model is built. This one is not optional: measured on the bundled
    # example, leaving TF32 on changes 2 of the 512 codes. TF32 costs a matmul thirteen bits of
    # mantissa, and the saliency encoder has no quantization wide enough to absorb that -- the
    # same drift is what moves the supervoxel budget by one in `inference/text2shape.py`.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from importlib import import_module
    sal_vae = import_module(f"{pkg}.train.stage1Text2Saliency.vae")

    cfg = json.load(open(os.path.join(args.ckpts, "stage1SaliencyVaeConfig.json")))
    spec = cfg["models"]["encoder"]
    model = getattr(sal_vae, spec["name"])(**spec.get("args", {}))
    sd = torch.load(os.path.join(args.ckpts, "stage1SaliencyVaeEncoder.pt"), map_location="cpu")
    sd = sd.get("state_dict", sd)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    result = model.load_state_dict(sd, strict=False)
    if result.missing_keys or result.unexpected_keys:
        warnings.warn(
            "saliency encoder checkpoint loaded with strict=False; "
            f"{len(result.missing_keys)} missing keys {result.missing_keys[:5]}, "
            f"{len(result.unexpected_keys)} unexpected keys {result.unexpected_keys[:5]}. "
            "Encoded saliency codes may be invalid.", RuntimeWarning)
    model = model.to(args.device).eval()

    z = np.load(args.field)
    idx = z["idx64"].astype(np.int64)
    raw = z["val"].astype(np.float32)
    st = np.load(args.stats)
    p01, p99 = float(st["p01"]), float(st["p99"])
    val = np.clip((raw - p01) / max(p99 - p01, 1e-9), 0.0, 1.0)

    G = args.grid
    vol = np.zeros((G, G, G), np.float32)
    vol[idx[:, 0], idx[:, 1], idx[:, 2]] = val
    x = torch.from_numpy(vol)[None, None].to(args.device)
    print("[codes] %d occupied of %d cells, percentiles p01=%.4f p99=%.4f"
          % (len(idx), G ** 3, p01, p99), flush=True)

    with torch.no_grad():
        out = model(x)
    ind = out["indices"]
    flat = ind.reshape(-1).cpu().numpy().astype(np.int64)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, indices_flat=flat)
    print("[codes] indices %s -> indices_flat (%d,) -> %s"
          % (tuple(ind.shape), len(flat), args.out))
    print("        range [%d, %d], unique %d of codebook %d"
          % (flat.min(), flat.max(), len(np.unique(flat)), int(model.codebook_size)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
