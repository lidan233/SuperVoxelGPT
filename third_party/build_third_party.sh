#!/usr/bin/env bash
# Build the four CUDA extensions the pipeline needs, for the card in this machine.
#
# Purpose: o_voxel, cumesh and flex_gemm are fetched from their own upstreams and installed into
#     the active environment. pycvt is different: its source is in this repository and its product
#     is loaded by path, so it is compiled in place beside that source.
#
# Input:  git, a CUDA toolkit whose nvcc knows your card, and an activated environment with torch.
# Output: the first three importable from python; pycvt as soft_cvt_clean.so under
#     train/vendor/runtime/pycvt/.
#
# Key idea: the architecture is read from the driver rather than hardcoded. A cubin built for a
#     newer architecture than the card cannot load, and one built for an older architecture is
#     JIT-translated on every import — both are avoidable by asking what is actually installed.
#
#   bash build_third_party.sh
#   PREFIX=/scratch/ext ONLY=flex_gemm bash build_third_party.sh
#   ONLY=pycvt bash build_third_party.sh
set -euo pipefail
cd "$(dirname "$0")"

PREFIX="${PREFIX:-$PWD/src}"
PY="${PY:-python}"
ONLY="${ONLY:-}"
ARCH="${TORCH_CUDA_ARCH_LIST:-}"

if [ -z "$ARCH" ]; then
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  [ -n "$cap" ] || { echo "cannot detect the GPU architecture; set TORCH_CUDA_ARCH_LIST" >&2; exit 1; }
  ARCH="$cap"
fi
export TORCH_CUDA_ARCH_LIST="$ARCH"

command -v git >/dev/null || { echo "git not found" >&2; exit 1; }
command -v nvcc >/dev/null || {
  echo "nvcc not found. Put a CUDA toolkit in the environment and export CUDA_HOME;" >&2
  echo "see env/README.md. A system nvcc older than your card fails at build time." >&2; exit 1; }

echo "arch   $TORCH_CUDA_ARCH_LIST"
echo "nvcc   $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
echo "prefix $PREFIX"
mkdir -p "$PREFIX"

want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

clone() {   # name url [extra git args]
  local name="$1" url="$2"; shift 2
  if [ -d "$PREFIX/$name/.git" ]; then
    echo "== $name: already cloned, pulling =="
    git -C "$PREFIX/$name" pull --ff-only "$@" || true
  else
    echo "== $name: cloning =="
    git clone --recursive "$url" "$PREFIX/$name" "$@"
  fi
}

if want flex_gemm; then
  clone FlexGEMM https://github.com/JeffreyXiang/FlexGEMM.git
  "$PY" -m pip install "$PREFIX/FlexGEMM" --no-build-isolation
fi

if want cumesh; then
  clone CuMesh https://github.com/JeffreyXiang/CuMesh.git
  "$PY" -m pip install "$PREFIX/CuMesh" --no-build-isolation
fi

if want o_voxel; then
  # o-voxel is a directory inside TRELLIS.2 rather than its own repository. Only that directory is
  # needed, so the clone is shallow and the rest of the tree is left alone.
  clone TRELLIS.2 https://github.com/microsoft/TRELLIS.2 --depth 1
  "$PY" -m pip install "$PREFIX/TRELLIS.2/o-voxel" --no-build-isolation
fi

if want pycvt; then
  # The CVT kernel has no upstream: its source is in this repository and its product is loaded by
  # path, not imported from site-packages. Built through torch's own cpp_extension.load with the
  # same flags pycvt.load_kernel() uses, so a binary built here and one built lazily at first
  # import are the same thing. -fmad=false is part of the determinism contract, not an option.
  echo "== pycvt: compiling soft_cvt_kernel.cu =="
  "$PY" - <<'PY'
import os, shutil, sys
from torch.utils.cpp_extension import load
here = os.path.abspath(os.path.join("..", "train", "vendor", "runtime", "pycvt"))
src = os.path.join(here, "soft_cvt_kernel.cu")
if not os.path.exists(src):
    raise SystemExit(f"missing {src}")
ext = load(name="soft_cvt_clean", sources=[src],
           extra_cuda_cflags=["-fmad=false", "-O3"], verbose=False)
built = ext.__file__
dst = os.path.join(here, "soft_cvt_clean.so")
if os.path.abspath(built) != os.path.abspath(dst):
    shutil.copy2(built, dst)
print(f"  soft_cvt_clean.so <- {built}")
PY
fi

echo
echo "== checking =="
ONLY="$ONLY" "$PY" - <<'PY'
import importlib, importlib.util, os
only = os.environ.get("ONLY", "")
want = (lambda n: not only or only == n)
ok = True

for m in ("o_voxel", "cumesh", "flex_gemm"):
    if not want(m):
        continue
    try:
        importlib.import_module(m)
        print(f"  {m:10s} OK")
    except Exception as e:
        print(f"  {m:10s} -- {type(e).__name__}: {e}")
        ok = False

if want("pycvt"):
    # Loaded by path rather than installed, so check the file the pipeline will open. torch has to
    # be imported first: the extension links against libc10, which only resolves once torch has
    # pulled it into the process — which is also why load_kernel() never hits this.
    import torch  # noqa: F401
    so = os.path.abspath(os.path.join("..", "train", "vendor", "runtime", "pycvt",
                                      "soft_cvt_clean.so"))
    try:
        spec = importlib.util.spec_from_file_location("soft_cvt_clean", so)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "soft_cvt"), "built, but exports no soft_cvt"
        print(f"  {'pycvt':10s} OK")
    except Exception as e:
        print(f"  {'pycvt':10s} -- {type(e).__name__}: {e}")
        ok = False

raise SystemExit(0 if ok else 1)
PY
