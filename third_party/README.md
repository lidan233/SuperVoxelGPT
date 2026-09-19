# third_party

Three CUDA extensions the pipeline imports but does not contain. They are built from their own
upstream repositories rather than copied in here, so that a bug fixed upstream can be picked up by
rebuilding instead of by re-vendoring, and so their licenses stay with their own code.

| extension | what needs it | upstream |
|---|---|---|
| `o_voxel` | dual-grid extraction — `inference/text2shape.py` turns the decoder's field into a mesh through it | inside [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (MIT), directory `o-voxel/` |
| `cumesh` | hole filling and small-component removal on that mesh | [JeffreyXiang/CuMesh](https://github.com/JeffreyXiang/CuMesh) |
| `flex_gemm` | the sparse convolutions the supervoxel autoencoder is built from — it cannot be constructed without this | [JeffreyXiang/FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM) (MIT) |

The CVT kernel is in this repository rather than fetched from an upstream: its source sits in
`train/vendor/runtime/pycvt/` and its product is loaded by path, not imported from site-packages.
`ONLY=pycvt bash build_third_party.sh` compiles it in place; `pycvt.load_kernel()` also compiles it
lazily if the prebuilt `soft_cvt_clean.so` cannot load on the machine at hand, so an explicit build
is optional. Both paths pass `-fmad=false -O3`, which is part of the determinism contract rather
than a tuning choice. Both `inference/text2shape.py` and the corpus step place centers with it.

## Build

```bash
bash build_third_party.sh                 # clones into ./src, builds, installs into the env
PREFIX=/somewhere bash build_third_party.sh    # clone somewhere else
```

`src/` is where those clones land, and it is not part of this repository — the sources stay with
their own upstreams. The released binaries were built from these commits, so checking one out
reproduces exactly what `train/vendor/runtime/` contains:

| upstream | commit |
|---|---|
| [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | [`75fbf01`](https://github.com/microsoft/TRELLIS.2/commit/75fbf0183001ed9876c8dbb35de6b68552ee08bd) |
| [JeffreyXiang/CuMesh](https://github.com/JeffreyXiang/CuMesh) | [`12289e1`](https://github.com/JeffreyXiang/CuMesh/commit/12289e1062f0603f2f0d0771b02e1395d247f26f) |
| [JeffreyXiang/FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM) | [`6dd94a8`](https://github.com/JeffreyXiang/FlexGEMM/commit/6dd94a859c26ee8246888502eada3dd8ad85532e) |

```bash
git clone --recursive https://github.com/microsoft/TRELLIS.2    src/TRELLIS.2 \
    && git -C src/TRELLIS.2 checkout 75fbf0183001ed9876c8dbb35de6b68552ee08bd
git clone --recursive https://github.com/JeffreyXiang/CuMesh    src/CuMesh \
    && git -C src/CuMesh   checkout 12289e1062f0603f2f0d0771b02e1395d247f26f
git clone --recursive https://github.com/JeffreyXiang/FlexGEMM  src/FlexGEMM \
    && git -C src/FlexGEMM checkout 6dd94a859c26ee8246888502eada3dd8ad85532e
```

Each is built for the architecture of the GPU in the machine. That matters more than it looks: a
cubin built for a newer architecture cannot load at all, and one built for an older architecture is
JIT-translated at every import, which is slow and silent. `nvcc` must come from a CUDA release that
knows your card — a system `nvcc` predating it fails with `Unsupported gpu architecture`, which is
the usual cause of a failed build here. `env/README.md` shows how to get one inside the conda
environment.

## Prebuilt copies

`train/vendor/runtime/` holds prebuilt copies. `flex_gemm` and `o_voxel` are compiled for sm_80,
which is what the released results were produced on; they import on later architectures through JIT
translation rather than failing, so they are usable but not fast. Rebuilding from source is the
better path on anything newer. `pycvt` is the exception: it is compiled for sm_89 and recompiles
itself from source when that does not fit. `T2S_RUNTIME` selects which set is on `sys.path`.
