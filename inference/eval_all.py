#!/usr/bin/env python3
"""Time the final held-out test partition.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# OPTIONAL QUALITY METRIC (DISABLED): symmetric squared Chamfer distance.
#
# This repository does not ship the 20,000 source shapes or their processed
# ground-truth meshes, so Chamfer cannot be computed from a fresh checkout. To
# enable it, first download the train/test corpus and run the complete geometry
# preparation pipeline. Use the resulting processed meshes (for example,
# `<id>_final.ply`), not the raw GLB files. Each mesh is independently centered at
# its bounding-box center, divided by its longest bounding-box side, and shifted
# by 0.5 into [0, 1]^3 before sampling. Aspect ratio is preserved: shorter axes
# are centered rather than stretched independently. This removes translation and
# global scale, but deliberately does not rotate or otherwise align the shapes;
# their axis convention must still match.
# Then uncomment this function and the call in the evaluation loop below, and set
# `processed_mesh_dir` to the directory containing the processed test meshes.
#
# def chamfer_distance(mesh_a, mesh_b, num_points=100_000, seed=0):
#     """Symmetric mean squared nearest-neighbour distance between two surfaces."""
#     import numpy as np
#     import trimesh
#     from scipy.spatial import cKDTree
#
#     def normalize(mesh):
#         mesh = mesh.copy()
#         bounds = np.asarray(mesh.bounds, dtype=np.float64)
#         center = (bounds[0] + bounds[1]) / 2.0
#         longest_side = float((bounds[1] - bounds[0]).max())
#         if not np.isfinite(longest_side) or longest_side <= 0:
#             raise ValueError("cannot normalize an empty or degenerate mesh")
#         mesh.vertices = ((np.asarray(mesh.vertices, dtype=np.float64) - center)
#                          / longest_side + 0.5)
#         return mesh
#
#     mesh_a, mesh_b = normalize(mesh_a), normalize(mesh_b)
#     # Fixed sampling seeds make checkpoint comparisons repeatable. This is a
#     # sampled surface metric, so `num_points` must be reported with the result.
#     state = np.random.get_state()
#     try:
#         np.random.seed(seed)
#         points_a, _ = trimesh.sample.sample_surface(mesh_a, num_points)
#         np.random.seed(seed + 1)
#         points_b, _ = trimesh.sample.sample_surface(mesh_b, num_points)
#     finally:
#         np.random.set_state(state)
#     a_to_b = cKDTree(points_b).query(points_a, workers=-1)[0]
#     b_to_a = cKDTree(points_a).query(points_b, workers=-1)[0]
#     return float((np.mean(a_to_b ** 2) + np.mean(b_to_a ** 2)) / 2.0)
# -----------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description="Time the full test set.")
    ap.add_argument("--ckpts", default=os.path.join(_REPO, "ckpts"))
    ap.add_argument("--out", default=os.path.join(_REPO, "assets", "eval_all.jsonl"))
    ap.add_argument("--limit", type=int, default=None, help="stop after N objects")
    ap.add_argument("--tier", type=int, default=None,
                    help="pin every object to this caption length (0 = longest, 8 = shortest). "
                         "Left unset, each object draws a tier at random under --seed, so the "
                         "run covers all nine lengths instead of reporting one of them")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds the per-object tier draw; a run is reproducible from it")
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _REPO)
    import text2shape as T

    src = os.path.join(_REPO, "data", "data_prep", "3_train_test_data",
                       "test1000_captions.json")
    recs = json.load(open(src))
    if args.limit is not None:
        recs = recs[:args.limit]

    # Each record carries the same caption at nine lengths, and a fixed tier reports that length
    # rather than the model. Drawing per object spreads all nine over the test set at the same
    # cost of one generation each.
    n_tiers = sum(1 for k in recs[0] if k.startswith("tier_")) if recs else 0
    if args.tier is not None:
        if not 0 <= args.tier < n_tiers:
            raise SystemExit(f"--tier must be in [0, {n_tiers - 1}]; got {args.tier}")
        tiers = [args.tier] * len(recs)
    else:
        # Drawn up front so --limit takes a prefix of the same assignment a full run would use.
        tiers = list(np.random.default_rng(args.seed).integers(0, n_tiers, len(recs)))
    print("captions: %s (%d objects, %d tiers available)"
          % ("tier_%d" % args.tier if args.tier is not None
             else "random per object, seed %d" % args.seed, len(recs), n_tiers), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pipe = T.Text2Shape(args.ckpts, verbose=False)
    f = open(args.out, "w")
    secs, t0 = [], time.time()

    for i, r in enumerate(recs):
        tier = int(tiers[i])
        cap = r["tier_%d" % tier].replace("\n", " ").strip()
        try:
            a = time.time()
            mesh, tm = pipe(cap)
            el = time.time() - a
            row = {"id": r["id"], "tier": tier,
                   "supervoxels": int(tm.get("supervoxels", 0)),
                   "faces": len(mesh.faces), "sec": round(el, 3)}
            # OPTIONAL CHAMFER (disabled; see the requirements above):
            # import trimesh
            # processed_mesh_dir = "/path/to/processed/test/meshes"
            # reference_path = os.path.join(processed_mesh_dir, r["id"] + "_final.ply")
            # reference = trimesh.load(reference_path, process=False, force="mesh")
            # row["chamfer_l2"] = chamfer_distance(mesh, reference)
            secs.append(el)
            del mesh
        except Exception as e:
            row = {"id": r["id"], "tier": tier,
                   "error": "%s: %s" % (type(e).__name__, str(e)[:100])}
        f.write(json.dumps(row) + "\n")
        f.flush()
        if (i + 1) % 50 == 0:
            done = time.time() - t0
            print("  [%d/%d] %.1f min elapsed, %.0f min left" %
                  (i + 1, len(recs), done / 60,
                   (len(recs) - i - 1) * done / (i + 1) / 60), flush=True)
    f.close()

    if secs:
        print("\n%d final-test objects, %d failed" % (len(recs), len(recs) - len(secs)))
        print("  median %.2fs   mean %.2fs   min %.2fs   max %.2fs" %
              (statistics.median(secs), statistics.mean(secs), min(secs), max(secs)))
        print("  total %.0f s (%.0f min), model loading excluded" %
              (sum(secs), sum(secs) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
