#!/bin/bash
# Geometry conditioning for one object.
#   bash run.sh <raw.glb> <out_dir> [tag]
#
#   step 1  watertight     a_tile_extract.py                    → <tag>_wt.ply
#   step 2  sharpen        a_deform_refine_sparc.py             → <tag>_rfa.ply
#           blend+guard    b_blend_keepbest.py → c_final_guard.py → <tag>_final.ply  ← the sharp mesh
#           smooth+voxel   d_lap_vxz.py: Laplacian smoothing + RES^3 voxelization → <tag>.vxz
#
# Override the interpreter with PY=..., the resolution with RES=... (default 1024).
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY="${PY:-python}"
RES="${RES:-1024}"
D="$(cd "$(dirname "$0")" && pwd)"
GT="$1"; OUT="$2"; TAG="${3:-$(basename "${GT%.*}")}"
[ -z "$GT" ] && { echo "usage: bash run.sh <raw.glb> <out_dir> [tag]"; exit 1; }
mkdir -p "$OUT"

echo "[1/2] watertight @${RES}³ (narrow-band field → floodfill → tiled FlexiCubes)"
$PY "$D/step1_watertight/a_tile_extract.py" --ref "$GT" --res "$RES" --out "$OUT"
mv "$OUT/$(basename "${GT%.*}")_tiled_r${RES}.ply" "$OUT/${TAG}_wt.ply"

echo "[2a/2] sharpen (multi-view depth+normal render loss through differentiable MC)"
$PY "$D/step2_sharpen/a_deform_refine_sparc.py" --ref "$GT" --res "$RES" --iters 200 \
    --opt_sdf 1 --opt_weight 0 --sdf_smooth 50 \
    --out_before "$OUT/${TAG}_rfb.ply" --out_after "$OUT/${TAG}_rfa.ply"

echo "[2b/2] blend + guard (keep sharpening only where it actually improved)"
$PY "$D/step2_sharpen/b_blend_keepbest.py" --before "$OUT/${TAG}_rfb.ply" --after "$OUT/${TAG}_rfa.ply" \
    --ref "$GT" --out "$OUT/${TAG}_blended.ply"
$PY "$D/step2_sharpen/c_final_guard.py" --wt "$OUT/${TAG}_rfb.ply" --blend "$OUT/${TAG}_blended.ply" \
    --out "$OUT/${TAG}_final.ply"

echo "[2d/2] Laplacian smoothing + voxelize to .vxz"
printf '%s\n' "$OUT/${TAG}_final.ply" > "$OUT/.lap_list"
mkdir -p "$OUT/.lap_q/done"
printf '0\n' > "$OUT/.lap_q/next_task.txt"; printf '0\n' > "$OUT/.lap_q/array_end.txt"
# The worker increments this counter on success.
printf '0\n' > "$OUT/.lap_q/completed_count.txt"
# d_lap_vxz.py has its own --res default, so pass the one this run extracted at.
DIRECT_ARRAY_BASE="$OUT/.lap_q" MESHLIST="$OUT/.lap_list" DIRECT_ARRAY_CHUNK=1 \
  LAP_OUTDIR="$OUT" $PY "$D/step2_sharpen/d_lap_vxz.py" --res "$RES"

echo "[geometry] $TAG → $OUT/${TAG}_final.ply"
