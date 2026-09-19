#!/bin/bash
# Seal the bundled example into a watertight mesh, end to end.
#
#   bash example/run_watertight.sh [out_dir] [res]
#
# The input is `example/sailing_ship.glb`: 12,772 vertices, 6,752 faces, and
# genuinely broken — 3,025 disconnected components and 12,770 of its 16,513 edges
# are open boundaries (77.3%). It is bundled precisely because it is a hard case,
# not a clean one.
#
# What runs, in order:
#
#   1  a_tile_extract.py       narrow-band UDF → global floodfill → global case_id
#                              → tiled FlexiCubes            → <tag>_wt.ply
#   2  a_deform_refine_sparc   analytic UDF-gradient deformation, then a multi-view
#                              depth+normal render loss through differentiable MC
#                              → <tag>_rfb.ply (analytic) / <tag>_rfa.ply (optimized)
#   3  b_blend_keepbest.py     per-vertex pick between the two, by reference normals
#                              → <tag>_blended.ply
#   4  c_final_guard.py        whole-object roughness backstop; falls back to the
#                              un-refined mesh if refinement made it rougher
#                              → <tag>_final.ply     ← the delivered sharp mesh
#   5  d_lap_vxz.py            Laplacian smoothing + dual-grid voxelization
#                              → <tag>.vxz           ← what stage 2 consumes
#
# Steps 2–4 are a unit: the optimization is allowed to fail, and the two guards
# are what make its worst case "no improvement" rather than "ruined". On this
# object that machinery actually fires — see the numbers at the end of this file.
#
# Not covered here: the saliency and CVT stages (`data/data_prep/2_saliency/`).
# They need a corpus-wide percentile file, which `c_global_stats.py` computes over
# the whole dataset and which is not shipped with this repository. A single object
# cannot produce one, and using a per-object substitute changes the token budget —
# so this script stops at the geometry half rather than emit numbers that cannot be
# compared with the released corpus.
#
# Override the interpreter with PY=..., the resolution with RES=... (default 1024).
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PY="${PY:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
GEO="$REPO/data/data_prep/1_geometry"

GLB="$HERE/sailing_ship.glb"
OUT="${1:-$HERE/out}"
RES="${2:-${RES:-1024}}"
TAG="sailing_ship"

[ -f "$GLB" ] || { echo "missing $GLB"; exit 1; }
mkdir -p "$OUT"

echo "=== input ==="
$PY - "$GLB" <<'PYEOF'
import sys, numpy as np, trimesh, warnings
warnings.filterwarnings("ignore")
m = trimesh.load(sys.argv[1], process=False, force="mesh")
e = m.edges_sorted
u, c = np.unique(e, axis=0, return_counts=True)
print(f"  {len(m.vertices)} vertices / {len(m.faces)} faces")
print(f"  {m.body_count} connected components")
print(f"  {(c == 1).sum()} boundary edges of {len(u)} ({(c == 1).mean() * 100:.1f}%)")
print(f"  {(c > 2).sum()} non-manifold edges")
print(f"  watertight: {m.is_watertight}")
PYEOF

echo
echo "=== [1/5] watertight @${RES}³ (narrow-band field -> floodfill -> tiled FlexiCubes) ==="
$PY "$GEO/step1_watertight/a_tile_extract.py" --ref "$GLB" --res "$RES" --out "$OUT"
mv "$OUT/$(basename "${GLB%.*}")_tiled_r${RES}.ply" "$OUT/${TAG}_wt.ply"

echo
echo "=== [2/5] sharpen (analytic gradient init, then multi-view render loss) ==="
$PY "$GEO/step2_sharpen/a_deform_refine_sparc.py" --ref "$GLB" --res "$RES" --iters 200 \
    --base_mesh "$OUT/${TAG}_wt.ply" \
    --opt_sdf 1 --opt_weight 0 --sdf_smooth 50 \
    --out_before "$OUT/${TAG}_rfb.ply" --out_after "$OUT/${TAG}_rfa.ply"

echo
echo "=== [3/5] blend: keep the refinement only where it beat the reference normals ==="
$PY "$GEO/step2_sharpen/b_blend_keepbest.py" \
    --before "$OUT/${TAG}_rfb.ply" --after "$OUT/${TAG}_rfa.ply" \
    --ref "$GLB" --out "$OUT/${TAG}_blended.ply"

echo
echo "=== [4/5] guard: never deliver something rougher than the plain watertight mesh ==="
$PY "$GEO/step2_sharpen/c_final_guard.py" \
    --wt "$OUT/${TAG}_rfb.ply" --blend "$OUT/${TAG}_blended.ply" \
    --out "$OUT/${TAG}_final.ply"

echo
echo "=== [5/5] Laplacian smoothing + dual-grid voxelization ==="
# d_lap_vxz.py is written for a sharded queue; for one object we hand it a
# one-line list and a queue directory of its own.
printf '%s\n' "$OUT/${TAG}_final.ply" > "$OUT/.lap_list"
mkdir -p "$OUT/.lap_q/done"
printf '0\n' > "$OUT/.lap_q/next_task.txt"
printf '0\n' > "$OUT/.lap_q/array_end.txt"
# The queue also keeps a completion counter, which the worker increments on success.
printf '0\n' > "$OUT/.lap_q/completed_count.txt"
# d_lap_vxz.py has its own --res default, so pass the one this run extracted at.
DIRECT_ARRAY_BASE="$OUT/.lap_q" MESHLIST="$OUT/.lap_list" DIRECT_ARRAY_CHUNK=1 \
  LAP_OUTDIR="$OUT" $PY "$GEO/step2_sharpen/d_lap_vxz.py" --res "$RES" || true

echo
echo "=== result ==="
$PY - "$OUT/${TAG}_final.ply" <<'PYEOF'
import sys, os, numpy as np, trimesh, warnings
warnings.filterwarnings("ignore")
p = sys.argv[1]
if not os.path.exists(p):
    print(f"  {p} was not produced"); sys.exit(1)
m = trimesh.load(p, process=False)
e = m.edges_sorted
u, c = np.unique(e, axis=0, return_counts=True)
print(f"  {len(m.vertices)} vertices / {len(m.faces)} faces")
print(f"  {(c == 1).sum()} boundary edges, {(c > 2).sum()} non-manifold edges")
print(f"  watertight: {m.is_watertight}")
PYEOF

echo
for f in "${TAG}_wt.ply" "${TAG}_rfb.ply" "${TAG}_rfa.ply" "${TAG}_blended.ply" "${TAG}_final.ply" "${TAG}.vxz"; do
    [ -e "$OUT/$f" ] && printf "  %-26s %s\n" "$f" "$(du -h "$OUT/$f" | cut -f1)"
done

echo
echo "delivered: $OUT/${TAG}_final.ply"
echo "render it: python vis/render_mesh.py $OUT/${TAG}_final.ply"
echo
# Measured at --res 1024. Printed only when that is what ran, since every figure below
# is resolution-dependent and a 512 run reporting 1024 numbers is worse than silence.
if [ "$RES" = "1024" ]; then
echo "Reference run (RTX 4090, --res 1024 — the production resolution, measured end to end):"
echo "  step 1   16.9M active cubes, 3 tiles along x -> 2,157,851v / 4,315,838f"
echo "           0 boundary edges, 1 component, watertight; 8s; staircase 29.3%"
echo "  step 2   early-stopped at iteration 60, restored best@it0 (EMA 0.0156); 62s"
echo "           -> 2,182,676v (the render loss never beat the analytic init on this object)"
echo "  step 3   7.7% of refined vertices kept as real detail, the rest reverted"
echo "  step 4   roughness 0.068 blended vs 0.065 un-refined -> blended kept"
echo "  step 5   dual-grid voxelization v=2,630,378 -> sailing_ship.vxz (4.7 MB)"
fi
