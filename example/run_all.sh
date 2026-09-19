#!/bin/bash
# Run the whole data-prep pipeline on the asset bundled with this repository.
#
#   bash example/run_all.sh [work_dir]
#
# Produces a complete single-object corpus under work_dir, in the same layout as the
# released corpus. See data/data_prep/README.md for what each stage does.
#
#   1  geometry      GLB -> watertight mesh -> .vxz          (run_watertight.sh)
#   2  saliency      per-vertex spectral saliency            -> sal/
#   3  sample to 64  onto the occupancy grid                 -> f64/max/
#   4  statistics    percentile cut points                   -> stats_max.npz
#   5  CVT           supervoxel centers                      -> cvt/
#   6  codes         64-cube field -> stage 1 codes          -> saliency_code/
#   7  tokens        .vxz + centers -> stage 2 tokens        -> tokens/
#   8  conditions    captions -> CLIP; renders -> DINOv2     -> cond_text/, cond_image/
#   9  visualize     saliency field and mesh renders         -> vis/
#
# Environment: PY (python), BPY (a python that can import bpy), RES (1024),
# NUM_VIEWS (4), W as the first argument.
set -e

PY="${PY:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PREP="$REPO/data/data_prep"
TAG="sailing_ship"
GLB="$HERE/sailing_ship.glb"
W="${1:-$HERE/work}"
RES="${RES:-1024}"

RUNTIME="$REPO/train/vendor/runtime"
export PYTHONPATH="$RUNTIME${PYTHONPATH:+:$PYTHONPATH}"
# Keep the autotune cache inside the work directory rather than $HOME.
export FLEX_GEMM_AUTOTUNE_CACHE_PATH="${FLEX_GEMM_AUTOTUNE_CACHE_PATH:-$W/flex_gemm/autotune_cache.json}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

mkdir -p "$W"/{geo,sal,f64,cvt,saliency_code,tokens,cond_text,cond_image,renders,vis,stage_ply,vxz_dir,flex_gemm}

echo "=== [1/9] geometry: GLB -> watertight -> vxz ==="
bash "$HERE/run_watertight.sh" "$W/geo" "$RES"

echo
echo "=== [2/9] spectral saliency ==="
# The saliency stage globs *_<res>_watertight.ply; the geometry stage writes _final.ply.
ln -sf "$W/geo/${TAG}_final.ply" "$W/stage_ply/${TAG}_${RES}_watertight.ply"
$PY "$PREP/2_saliency/a_spectral_saliency.py" \
    --input_dir "$W/stage_ply" --output_dir "$W/sal" \
    --target_resolution "$RES" --cuda_idx 0 --task_id 0 --files_per_task 200

echo
echo "=== [3/9] sample onto the 64-cube occupancy ==="
ln -sf "$W/geo/${TAG}.vxz" "$W/vxz_dir/${TAG}.vxz"
$PY "$PREP/2_saliency/b_sample_to_64.py" \
    --vxz-dir "$W/vxz_dir" --saliency-dir "$W/sal" --out-dir "$W/f64" \
    --saliency-suffix "_mesh_saliency_energy_${RES}.npz" \
    --saliency-key S_log_gpu --grid "$RES" --target 64

echo
echo "=== [4/9] percentile cut points ==="
# One object here, the whole corpus in a real run.
$PY "$PREP/2_saliency/c_global_stats.py" \
    --src-dir "$W/f64/max" --out "$W/stats_max.npz" --value-key val \
    --train-split "$HERE/captions.json" --validation-size 0

echo
echo "=== [5/9] saliency-driven CVT ==="
$PY "$PREP/2_saliency/d_cvt_pycvt.py" \
    --src-dir "$W/f64/max" --stats "$W/stats_max.npz" \
    --out-dir "$W/cvt" --single "$TAG"
# The autoencoder's loader expects this name.
ln -sf "${TAG}_supervoxel_center.ply" \
       "$W/cvt/${TAG}_saliency_volume_64_supervoxel_center.ply"

echo
echo "=== [6/9] stage 1 codes ==="
$PY "$PREP/3_train_test_data/encode_saliency_codes.py" \
    --field "$W/f64/max/${TAG}.npz" --stats "$W/stats_max.npz" \
    --ckpts "$REPO/ckpts" --out "$W/saliency_code/${TAG}.npz"

echo
echo "=== [7/9] stage 2 tokens ==="
$PY "$PREP/3_train_test_data/encode_tokens.py" \
    --vxz "$W/geo/${TAG}.vxz" --centers "$W/cvt/${TAG}_supervoxel_center.ply" \
    --ckpts "$REPO/ckpts" --out "$W/tokens/${TAG}.npz"

echo
echo "=== [8/9] conditioning: captions -> CLIP, renders -> DINOv2 ==="
$PY "$PREP/3_train_test_data/encode_text_condition.py" \
    --captions "$HERE/captions.json" --ckpts "$REPO/ckpts" \
    --out "$W/cond_text/${TAG}.npz"
# Blender renders the GLB with its materials; the rasterizer fallback produces untextured
# views, which are usable but not what the released features were computed from.
BPY="${BPY:-python}"
if "$BPY" -c "import bpy" 2>/dev/null; then
    "$BPY" "$REPO/vis/render_glb_views.py" "$GLB" "$W/renders" \
        --num-views "${NUM_VIEWS:-4}" --size 518
    $PY "$PREP/3_train_test_data/encode_image_condition.py" \
        --renders-dir "$W/renders" --out "$W/cond_image/${TAG}.npz"
else
    echo "[image] BPY cannot import bpy; falling back to the untextured rasterizer" >&2
    $PY "$PREP/3_train_test_data/encode_image_condition.py" \
        --mesh "$GLB" --out "$W/cond_image/${TAG}.npz" \
        --save-renders "$W/renders" --num-views "${NUM_VIEWS:-4}"
fi

# Both image training entry points require this file. This one-object mean is only a smoke-test
# fixture. A formal run computes global_mean.npy from train-split features and then uses that
# same frozen file for validation and test.
$PY - "$W" <<'PYEOF'
import sys, glob, numpy as np
w = sys.argv[1]
fs = sorted(glob.glob(w + "/cond_image/*.npz"))
F = np.concatenate([np.load(f)["features_518"].reshape(-1, 1024) for f in fs], 0)
np.save(w + "/global_mean.npy", F.mean(0).astype(np.float32))
print("[image] global_mean.npy from %d view vectors" % len(F))
PYEOF

echo
echo "=== [9/9] visualize ==="
# The vis scripts read idx/val in [0,1]; the sampling step writes idx64/val in log units.
$PY - "$W" "$TAG" <<'PYEOF'
import sys, os, shutil, numpy as np
w, tag = sys.argv[1], sys.argv[2]
d = np.load(os.path.join(w, "f64", "max", tag + ".npz"))
st = np.load(os.path.join(w, "stats_max.npz"))
p01, p99 = float(st["p01"]), float(st["p99"])
raw = d["val"].astype(np.float64)
val = np.clip((raw - p01) / max(p99 - p01, 1e-9), 0.0, 1.0).astype(np.float32)
np.savez(os.path.join(w, "vis", "sample.npz"),
         idx=d["idx64"].astype(np.int64), val=val)
print("[vis] saliency normalized: raw [%.3f, %.3f] -> [%.3f, %.3f] (p01=%.4f p99=%.4f)"
      % (raw.min(), raw.max(), val.min(), val.max(), p01, p99))
shutil.copy(os.path.join(w, "cvt", tag + "_supervoxel_center.ply"),
            os.path.join(w, "vis", "sample_supervoxel_center.ply"))
PYEOF
MPLBACKEND=Agg $PY "$REPO/vis/render_fields.py"   "$W/vis"
$PY "$REPO/vis/render_mesh.py" "$W/geo/${TAG}_final.ply" --out "$W/vis" --views 3 || true

echo
echo "=== produced ==="
$PY - "$W" "$TAG" <<'PYEOF'
import sys, os, numpy as np
w, tag = sys.argv[1], sys.argv[2]
def line(label, path, extra=""):
    p = os.path.join(w, path)
    ok = os.path.exists(p)
    size = ("%.1f MB" % (os.path.getsize(p) / 1e6)) if ok else "MISSING"
    print("  %-18s %-52s %10s %s" % (label, path, size, extra))
    return ok
line("watertight", "geo/%s_final.ply" % tag)
line("voxelization", "geo/%s.vxz" % tag)
line("saliency", "sal/%s_mesh_saliency_energy_1024.npz" % tag)
f = os.path.join(w, "f64", "max", tag + ".npz")
occ = len(np.load(f)["idx64"]) if os.path.exists(f) else 0
line("64-cube field", "f64/max/%s.npz" % tag, "%d occupied cells" % occ)
c = os.path.join(w, "cvt", tag + "_supervoxel_center.ply")
n = 0
if os.path.exists(c):
    import trimesh
    n = len(trimesh.load(c, process=False).vertices)
line("supervoxels", "cvt/%s_supervoxel_center.ply" % tag,
     "%d centers, %.2fx compression" % (n, occ / max(n, 1)))
line("stage 1 codes", "saliency_code/%s.npz" % tag, "512 indices on the 8-cube lattice")
t = os.path.join(w, "tokens", tag + ".npz")
if os.path.exists(t):
    z = np.load(t)
    line("stage 2 tokens", "tokens/%s.npz" % tag,
         "%d tokens, %d unique" % (len(z["indices"]), len(np.unique(z["indices"]))))
line("text condition", "cond_text/%s.npz" % tag, "9 caption granularities")
line("image condition", "cond_image/%s.npz" % tag, "DINOv2 37x37 patches per view")
PYEOF
echo
echo "training data for one object is now under $W"
echo
echo "The four training entry points read these directories directly; --repeat-factor fills"
echo "an epoch from the single object. For example:"
echo
echo "  python -m train.stage2TextSaliency2Shape.train_text \\"
echo "      --token-dir $W/tokens --condition-dir $W/cond_text \\"
echo "      --train-split $HERE/captions.json --test-split $HERE/test_split.json \\"
echo "      --validation-size 0 \\"
echo "      --repeat-factor 8 --max-steps 2 --micro-batch-size 2 --save-path runs/s2t"
echo
echo "The image branches take --condition-dir $W/cond_image and --global-mean $W/global_mean.npy."
