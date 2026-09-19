# Environment

`environment.yml` — the environment the reported numbers were produced in.

Reference platform: python 3.10 · torch 2.6.0+cu124 · CUDA 12.4 · GPU sm_80/86/89.

## The machine the reported time/accuracy come from

| | |
|---|---|
| GPU | 1 × NVIDIA GeForce RTX 4090, 24 GB (sm_89), driver 550.90.07 |
| CPU | AMD Ryzen 9 7950X, 16 cores / 32 threads |
| RAM | 125 GB |
| OS | Ubuntu 22.04.3 LTS, kernel 6.5.0 |

Training used 8×H200 or 8×A800 rather than this machine; sm_80 is what the
prebuilt extensions in `train/vendor/runtime/` target, which is why they are
JIT-translated on a 4090.


## Install

`conda env create -f environment.yml` gets you most of the way, but not all of
it, and that is a property of the dependency set rather than a defect in the
file. Five things cannot come from conda's single-shot pip pass: `torch` and the
two PyG extensions are local-version wheels that live outside PyPI, `flash-attn`
imports torch inside its own `setup.py`, and three packages are installed from
git. They are listed in a comment block at the end of `environment.yml`, pinned
by commit.

### Before the first command

Two things about recent conda releases will stop the install before it starts,
and neither failure says what to do next in a way that survives being scrolled
past.

**Accept the channel terms.** conda 25 and later refuse to install from
`repo.anaconda.com` until its terms are accepted, and the error arrives in the
middle of a solve rather than at the start:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

**Name the channels when creating the environment by hand.** `environment.yml`
already lists them, but `conda create -p ... python=3.10` on its own does not read
that file and falls back to the default channel, which pulls in a set of packages
this project never uses — a jupyter stack among them. They shadow nothing, but
they make `pip list` unreadable exactly when something has gone wrong and that
listing is what you are reading.

**Put it on a disk with room.** The environment is about 16 GB once flash-attn
and the CUDA toolkit are in it. `conda create -p /path/to/envs/name` places it
wherever there is space, which matters on a machine whose root partition is
nearly full.

In order:

```bash
# 1. the environment
conda create -p <somewhere>/envs/supervoxelgpt python=3.10 \
    -c conda-forge --override-channels
conda activate <somewhere>/envs/supervoxelgpt
# then the rest of environment.yml:
conda env update -p <somewhere>/envs/supervoxelgpt -f environment.yml

# 2. torch, from the cu124 index
pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 3. a CUDA toolkit inside the env, so nvcc can target your card. A system nvcc
#    from an older CUDA release fails with "Unsupported gpu architecture".
conda install -c nvidia/label/cuda-12.4.1 cuda-toolkit
export CUDA_HOME="$CONDA_PREFIX" PATH="$CONDA_PREFIX/bin:$PATH"

# 4. build-time deps, since everything below runs without build isolation
pip install -U setuptools wheel ninja packaging typing_extensions

# 5. the PyG extensions, from their own index rather than PyPI
pip install torch-scatter==2.1.2 torch-cluster==1.6.3 \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# 6. the rest of the runtime, pinned to what the reported numbers were produced
#    with. opencv and torch_geometric matter more than they look: a newer
#    torch_geometric changes MessagePassing internals the supervoxel encoder
#    calls into, and `pip install cv2` fails because the import name is not the
#    package name.
pip install transformers==4.51.3 accelerate==1.13.0 trimesh==4.12.2 \
    torch_geometric==2.7.0 \
    opencv-python==4.11.0.86 scipy easydict einops einx safetensors \
    huggingface-hub tqdm imageio pillow plyfile zstandard matplotlib \
    opencv-python-headless spconv-cu124==2.3.8 \
    libigl==2.6.3 pymeshlab==2025.7.post1 tensorboard==2.20.0

#    The last three are easy to miss because nothing in `inference/` imports them,
#    and that is the half most people run first:
#      libigl, pymeshlab  data_prep/2_saliency/a_spectral_saliency.py imports both at
#                         module level, for the cotangent Laplacian and the mesh
#                         cleanup — the saliency stage will not start without them.
#      tensorboard        both `train_vae.py` entry points import SummaryWriter
#                         through the trainer, so autoencoder training fails at import.

# 7. flash-attn: REQUIRED, not optional. The shape generator raises without it.
#    Its Jacobi decode runs one full-sequence forward per iteration, and the
#    determinism contract turns off flash and mem-efficient SDP, so the `sdpa`
#    path falls back to math attention and materializes the whole N x N score
#    matrix every iteration: 2.6x slower end to end at N around 2200, for
#    bit-identical output. `T2S_ATTN_IMPL=sdpa` accepts that cost deliberately.
#
#    85 translation units. 30-90 min at MAX_JOBS=4; a 32-core
#    machine takes about 15 min at MAX_JOBS=8. Too many parallel jobs and the
#    OOM killer ends it, leaving a bare "Killed" in the log.
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX" MAX_JOBS=8 \
    pip install flash-attn==2.6.3 --no-build-isolation

# 8. the three git packages, pinned by commit (see the end of environment.yml)
pip install --no-build-isolation \
  "git+https://github.com/ashawkey/cubvh@757b913bfbf19ed65e3a379d159391a8e29efa0f" \
  "git+https://github.com/NVlabs/nvdiffrast.git@729261dc64c4241ea36efda84fbf532cc8b425b8" \
  "git+https://github.com/Pointcept/Pointcept.git@9f37497e4f3005c90bbbe7221b86439c29d60611#subdirectory=libs/pointops"
```

`pointops` used to be a standalone `Pointcept/pointops` repository. That
repository is gone; it is now a subdirectory of the main `Pointcept` repo, which
is what the URL above points at.

### `cubvh` and `T2S_RUNTIME`

`cumesh` links its own copy of the BVH and registers the same pybind11 type the standalone
`cubvh` package registers. Import `cubvh` without `T2S_RUNTIME` on the path if you need it
directly; the text-to-shape chain does not use it.

## Data preparation only

`data/data_prep/` needs far less than the full environment: `numpy`, `scipy`,
`torch`, `trimesh`, `tqdm`, `cubvh`, `igl`, `diso`, `pymeshlab`, and the vendored
`FlexiCubes`. Installing those directly is much faster than building everything
above.

## Compiled extensions

Three CUDA extensions are built or installed per machine, and none of them comes
from the conda pass above. All three are needed to produce a mesh at all; the CVT
that places the supervoxel centers ships prebuilt in `train/vendor/runtime/pycvt/`
with its `.cu` source beside it, so nothing has to be compiled for that step.

| | what needs it | where it comes from |
|---|---|---|
| `flex_gemm` | the supervoxel autoencoder's sparse convolutions — it cannot even be constructed without this | https://github.com/JeffreyXiang/FlexGEMM (MIT) |
| `o_voxel` | dual-grid extraction, `inference/text2shape.py` | ships with TRELLIS.2 |
| `cumesh` | hole filling, same file | platform wheel |

`flex_gemm` autotunes each sparse-convolution shape the first time it sees one and
saves the result to `~/.flex_gemm/autotune_cache.json`. Two consequences worth
knowing before reading any timing:

- The first few objects of a run on a cold cache are several times slower than
  steady state. On one RTX 4090 the first two objects took 38.4s and 31.3s while
  the next two took 4.6s and 7.4s. Discard them before quoting a per-object time.
- If `~/.flex_gemm` exists but is not a usable directory — most easily a symlink
  whose target is an unmounted disk — every object fails with
  `FileExistsError(17)` raised from inside `filelock`, because
  `Path.mkdir(exist_ok=True)` does not accept a dangling symlink. The error
  surfaces mid-pipeline and is caught per object, so a batch run quietly
  completes with most objects missing. Point the cache somewhere real instead of
  deleting the link:

```bash
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/path/on/a/mounted/disk/autotune_cache.json
```

A cubin built for a newer card cannot load, and one built for an older card is
silently JIT-translated on every import, so anything compiled here should be
built against the architecture actually present.

The other three ship prebuilt in `train/vendor/runtime/`, so nothing has to be
compiled to run the pipeline:

```bash
export T2S_RUNTIME="$PWD/train/vendor/runtime"
```

Those copies are built for sm_80, the architecture the released numbers were
produced on. They import on newer cards through JIT translation, which works but
costs time at every import. To build them for the card you actually have:

```bash
bash third_party/build_third_party.sh
```

See [`third_party/README.md`](../third_party/README.md) for where each comes from.

## Checkpoints

outside git:

```bash
bash ckpts/fetch_ckpts.sh
```

It downloads and then verifies against `MD5SUMS.txt`. See
[`ckpts/README.md`](../ckpts/README.md) for what each file is and which stage
loads it.
