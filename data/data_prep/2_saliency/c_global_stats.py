#!/usr/bin/env python3
"""Fit the saliency scale once on the training corpus.

Purpose
    The budget formula downstream reads absolute saliency values: a cell at 0.9 earns many more
    points than a cell at 0.3. That only means anything if 0.9 means the same thing on every object.
    A single pass over the training corpus is therefore not an optimization detail — it is what makes
    one budget formula valid for a whole dataset.

Input
    Per-object sparse fields emitted by the sampling step, plus the training split.

Output
    One small file per pooling mode holding the percentile cut points, and the cell count they were
    derived from.

    The cut points are an asset, not a diagnostic. Regenerating any object with different ones
    produces a different point budget, so a stored field and the statistics it was normalized with
    belong together.

Key idea
    The alternative — normalizing each object against its own range — is the mistake this step
    exists to prevent, and it is not a subtle one. Per-shape normalization maps every object's most
    salient region to 1.0, whether that object is a featureless bowl or a filigree lattice. Detail
    stops being comparable across the dataset, the budget collapses toward a constant, and nothing
    about the result looks wrong until reconstruction quality is measured.

    The percentiles are computed exactly rather than estimated. Values are gathered from every cell
    of every object and selected by partial sort, which is affordable because only the occupied
    cells exist and because it runs once. Sketching or subsampling would put the cut points a little
    off, and a little off here is a systematic shift in every budget derived afterwards.

    Each pooling mode is measured separately. Max and sum answer different questions and their
    distributions differ by orders of magnitude, so one shared scale would misplace both.
"""
import argparse, json, os
import numpy as np
from pathlib import Path


def collect(files, key):
    """Concatenate the values of every occupied cell across all objects."""
    total = 0
    for f in files:
        with np.load(f) as z:
            total += len(z[key])
    print(f"  {len(files)} files, {total} occupied cells ({total * 4 / 1e9:.2f} GB)", flush=True)
    out = np.empty(total, np.float32)
    off = 0
    for i, f in enumerate(files, 1):
        with np.load(f) as z:
            v = z[key].astype(np.float32)
        out[off:off + len(v)] = v
        off += len(v)
        if i % 500 == 0:
            print(f"  read {i}/{len(files)}", flush=True)
    return out[:off]


def exact_percentiles(values, ps):
    """Exact percentile cut points by partial sort — no estimation, no subsampling."""
    n = len(values)
    if n == 0:
        raise ValueError("no values collected; check the input directory and --value-key")
    ranks = sorted({int(p / 100.0 * (n - 1)) for p in ps})
    values.partition(ranks)
    return {p: float(values[int(p / 100.0 * (n - 1))]) for p in ps}


def load_ids(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(row["id"]) for row in data]
    if isinstance(data, dict):
        if "id" in data:
            return [str(data["id"])]
        return [str(k) for k in data]
    raise ValueError(f"{path}: expected caption records or an id-keyed object")


def run(src_dir, out_path, value_key, ps, train_split, validation_size):
    files = sorted(Path(src_dir).glob("*.npz"))
    split_ids = load_ids(train_split)
    train_ids = set(split_ids[validation_size:])
    available = {f.stem for f in files}
    missing = train_ids - available
    if missing:
        raise SystemExit(f"{len(missing)} training ids have no npz under {src_dir}; "
                         f"examples: {sorted(missing)[:5]}")
    files = [f for f in files if f.stem in train_ids]
    if not files:
        raise SystemExit(f"no npz files under {src_dir} match the training split {train_split}")
    vals = collect(files, value_key)
    p = exact_percentiles(vals, ps)
    np.savez(out_path,
             p01=np.float32(p[1]), p99=np.float32(p[99]), p50=np.float32(p[50]),
             pmin=np.float32(p[0]), pmax=np.float32(p[100]), n=np.int64(len(vals)),
             num_objects=np.int64(len(files)),
             validation_size=np.int64(validation_size),
             train_split=np.asarray(os.path.abspath(train_split)))
    print(f"  p01={p[1]:.4f} p50={p[50]:.4f} p99={p[99]:.4f} "
          f"min={p[0]:.4f} max={p[100]:.4f} -> {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Production invocation (once per pooling mode, single process — this step is
# global by definition and must not be sharded):
#
#   python c_global_stats.py --src-dir <dense64 dir>/max --out <stats_max.npz> \
#       --train-split <train19000_captions.json> --validation-size 100
#   python c_global_stats.py --src-dir <dense64 dir>/sum --out <stats_sum.npz> \
#       --train-split <train19000_captions.json> --validation-size 100
#
# Run it after sampling has finished. Only records after the reserved validation
# prefix in --train-split are used, even if other files share the source directory.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Corpus-wide percentile cut points for the saliency field.")
    ap.add_argument("--src-dir", required=True, help="directory of per-object sparse fields, one pooling mode")
    ap.add_argument("--out", required=True, help="output npz holding the cut points")
    ap.add_argument("--train-split", required=True,
                    help="ordered JSON records used to select the training objects")
    ap.add_argument("--validation-size", type=int, default=100,
                    help="reserve the first N train-split records; use 0 for the one-object example")
    ap.add_argument("--value-key", default="val", help="npz key holding the per-cell values")
    ap.add_argument("--percentiles", default="0,1,50,99,100",
                    help="percentiles to record; the normalization uses 1 and 99, the rest are context")
    args = ap.parse_args()

    ps = [float(x) for x in args.percentiles.split(",")]
    for required in (1, 50, 99, 0, 100):
        if required not in ps:
            ap.error(f"--percentiles must include {required}; the output file records it")
    print(f"scanning {args.src_dir}", flush=True)
    if not 0 <= args.validation_size < len(load_ids(args.train_split)):
        ap.error("--validation-size must leave at least one training object")
    run(args.src_dir, args.out, args.value_key, ps, args.train_split, args.validation_size)


if __name__ == "__main__":
    main()
