#!/usr/bin/env python3
"""Saliency -> supervoxel centers, through the same CVT inference uses.

Purpose
    Places the supervoxel centers a shape's tokens are indexed by. It calls
    `pycvt_from_field`, which is the function `inference/text2shape.py` calls, so the centers
    a corpus records and the centers serving computes come out of one implementation.

Input
    The coarse saliency npz written by the sampling step -- `idx64` [M,3] and `val` [M], raw
    values -- and the corpus percentile file from `c_global_stats.py`.

Output
    `<id>_supervoxel_center.ply`, binary, xyz float32 on the 0-256 density grid. That is the frame
    `encode_tokens.py` divides by, and the frame the corpus CVT files are written in.

Key idea
    The percentile map belongs here rather than in the kernel wrapper. The values this step reads
    are raw, and what makes them comparable across a corpus is p01/p99 measured over the corpus;
    the wrapper takes saliency already in [0, 1], because the field it is handed at inference time
    comes from stage 1 and is already there. Mapping here and not there is what lets one
    tessellation serve both.

    Everything downstream of that map -- budget, spacing, density, seeding, Morton order, the
    rounding -- is inside `pycvt_from_field` and is not reimplemented here, so the corpus and
    inference run one tessellation rather than two.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

# 2_saliency -> data_prep -> data -> repo
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
# The directory holding pycvt.py goes on the path itself, ahead of PYTHONPATH, so the module
# wins over the same-named directory under train/vendor/runtime.
sys.path.insert(0, os.environ.get(
    "PYCVT_SRC_DIR", os.path.join(_REPO, "train", "vendor", "runtime", "pycvt")))
from pycvt import pycvt_from_field, write_centers_ply   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Soft-CVT supervoxel sampling, as inference does it.")
    ap.add_argument("--src-dir", default=os.environ.get("SVOX_SRC_DIR"),
                    help="coarse saliency npz directory {idx64[M,3], val[M]}, raw values "
                         "[env SVOX_SRC_DIR]")
    ap.add_argument("--stats", default=os.environ.get("SVOX_STATS"),
                    help="corpus percentile file from c_global_stats.py; its p01/p99 map the raw "
                         "values into [0,1]  [env SVOX_STATS]")
    ap.add_argument("--out-dir", default=os.environ.get("SVOX_CVT_OUT"),
                    help="output directory for *_supervoxel_center.ply  [env SVOX_CVT_OUT]")
    ap.add_argument("--ref-dir", default=os.environ.get("SVOX_REF_DIR"),
                    help="batch mode: output names follow the ply files here  [env SVOX_REF_DIR]")
    ap.add_argument("--single", default=os.environ.get("GF_SINGLE"),
                    help="process one id, naming the output <id>_supervoxel_center.ply "
                         "[env GF_SINGLE]")
    ap.add_argument("--list", dest="lst", default=os.environ.get("GF_LIST"),
                    help="file of ids to restrict the run to  [env GF_LIST]")
    ap.add_argument("--shard-id", type=int, default=int(os.environ.get("SHARD_ID", "0")),
                    help="[env SHARD_ID]")
    ap.add_argument("--shard-count", type=int, default=int(os.environ.get("SHARD_COUNT", "1")),
                    help="[env SHARD_COUNT]")
    ap.add_argument("--grid", type=int, default=64, help="resolution of the source saliency field")
    a = ap.parse_args()

    if not a.src_dir or not a.out_dir:
        ap.error("--src-dir and --out-dir are required (or set SVOX_SRC_DIR / SVOX_CVT_OUT)")
    if not a.stats:
        ap.error("--stats is required (or set SVOX_STATS): the source values are raw, and the "
                 "percentile map that gives them a corpus-wide meaning is not stored with them")

    st = np.load(a.stats)
    p01, p99 = float(st["p01"]), float(st["p99"])
    den = max(p99 - p01, 1e-9)
    os.makedirs(a.out_dir, exist_ok=True)
    print("[s%d] percentile map p01=%.4f p99=%.4f from %s" % (a.shard_id, p01, p99, a.stats),
          flush=True)

    if a.lst and os.path.exists(a.lst):
        ids = [l.strip() for l in open(a.lst) if l.strip()]
    else:
        ids = sorted(os.path.splitext(f)[0] for f in os.listdir(a.src_dir) if f.endswith(".npz"))
    mine = [a.single] if a.single else ids[a.shard_id::a.shard_count]

    n_ok = n_skip = n_err = 0
    for sid in mine:
        if a.single:
            op = os.path.join(a.out_dir, sid + "_supervoxel_center.ply")
        else:
            refs = glob.glob(os.path.join(a.ref_dir, "*" + sid + "*_supervoxel_center.ply"))
            if not refs:
                n_err += 1
                continue
            op = os.path.join(a.out_dir, os.path.basename(refs[0]))
        if os.path.exists(op):
            n_skip += 1
            continue
        try:
            t0 = time.time()
            d = np.load(os.path.join(a.src_dir, sid + ".npz"))
            idx = d["idx64"].astype(np.int64)
            raw = d["val"].astype(np.float32)
            val = np.clip((raw - p01) / den, 0.0, 1.0)
            if len(idx) == 0:
                n_err += 1
                continue
            # raw=True: the 0-256 frame, which is what the centers ply stores.
            cent = pycvt_from_field(idx, val, raw=True)
            write_centers_ply(op, cent)
            n_ok += 1
            print("[s%d] %s occupied=%d -> centers=%d  %.1fs"
                  % (a.shard_id, sid[:8], len(idx), len(cent), time.time() - t0), flush=True)
        except Exception as e:
            n_err += 1
            print("[s%d] ERR %s: %s" % (a.shard_id, sid[:8], str(e)[:120]), flush=True)
    print("[s%d] DONE ok=%d skip=%d err=%d" % (a.shard_id, n_ok, n_skip, n_err), flush=True)


# ---------------------------------------------------------------------------
# Production invocation (per shard, 8 shards per node):
#
#   SVOX_SRC_DIR=<coarse saliency npz dir> \
#   SVOX_STATS=<stats npz from c_global_stats.py> \
#   SVOX_CVT_OUT=<output dir> \
#   SVOX_REF_DIR=<dir whose ply names the outputs follow> \
#   SHARD_ID=$i SHARD_COUNT=8 CUDA_VISIBLE_DEVICES=$i \
#   python d_cvt_pycvt.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
