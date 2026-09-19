# vendor

Code this repository uses but did not write.

| subtree | what it is | who needs it |
|---|---|---|
| `trellis2/` | the sparse-tensor framework the supervoxel autoencoder is built on | `stage2TextSaliency2Shape/`, and `inference/text2shape.py` through it |
| `runtime/` | `o_voxel`, `cumesh` and `flex_gemm`, already compiled | the mesh tail of inference, and the sparse convolutions above |

The saliency stage has nothing here. Its trainer and dataset were rewritten as
`stage1Text2Saliency/trainer_vae.py` and `dataset_vae.py`, which implement the one path the
released config actually takes; the framework they came from is gone.

The supervoxel stage's own layers moved out too, into
`stage2TextSaliency2Shape/sparse_vae/` — the CVT-conditioned encoder, the residual FSQ quantizer,
the grid/Voronoi cross-attention and the subdivision decoder, about 1100 lines. What stays below is
the framework underneath them.

## What is left in `trellis2/`

Only what the two stages reach. Upstream has 92 files; 33 are here, and the rest were removed
rather than carried: the attention and transformer stacks, the serialization helpers, two of the
three sparse-convolution backends, and every model and dataset this repository does not build on.

Of those 33, **seven were written or modified here** and are listed below, because a vendored tree
whose contents are described as untouched has to actually be untouched, or the description is worse
than none.

### Files written here, not upstream

| file | lines | what it is |
|---|---|---|
| `trainers/vae/subdivision_vae.py` | 163 | the supervoxel VAE's trainer |
| `trainers/vae/subdivision_vae_full.py` | 167 | its subclass, adding the intersection and vertex losses |

These two belong with the stage that uses them, next to `sparse_vae/`. They are still here only
because the trainer rewrite is a separate change; moving them without it would split one job in two.
The dataset that used to sit beside them now lives at `train/dataset/shape_vae.py`, with the rest of
this repository's datasets.

### Upstream files modified

| file | change |
|---|---|
| `trainers/basic.py` | +139 lines: optional CUDA cache management, off by default |
| `trainers/utils.py` | +29 lines: `CosineWarmupLRScheduler`, which the released config names |
| `modules/sparse/__init__.py` | −11 lines: the lazy-import table no longer lists the attention, serialization and transformer modules, which were removed. Left in, the table turns a typo into an ImportError from deep inside importlib instead of a clear AttributeError. |
| `__init__.py`, `models/__init__.py`, `datasets/__init__.py` | emptied. Upstream's registries name modules this repository does not vendor, so importing the package would fail on the first one. |

The rest is upstream as copied. That is checked where it matters rather than asserted: the released
encoder and decoder load through this code with `strict=True`, with the original packages absent
from `sys.modules`. It is not a byte-level comparison — no upstream tree is reachable from here to
diff against.

## `runtime/`

Prebuilt copies of the three CUDA extensions, so the pipeline runs without a compiler.

`pycvt/` is the exception in two ways: it is built for **sm_89** rather than sm_80, and it carries
its `.cu` source beside the binary because `torch.utils.cpp_extension.load` compiles it on first
use. It is the tessellation both sides run: `inference/text2shape.py` places centers with it, and
`data_prep` does too, through `2_saliency/d_cvt_pycvt.py`.

That measurement is why the corpus step was moved onto `pycvt/` too: an earlier relaxation placed
the same number of centers in different positions, and the generator's parallel decode reached
token accuracy 1.0000 on `pycvt/`'s centers and 0.0000 on the other's. One tessellation, both
sides.

The rest **are built for sm_80** — A100 and A800, which is what the released numbers were produced
on. On a newer card they still import, through JIT translation at load time, which works but is
slower than a native build. `third_party/build_third_party.sh` fetches the sources and builds for whatever
card is present; `T2S_RUNTIME` chooses which set is on `sys.path`.

Licenses belong to their own projects; [`third_party/README.md`](../../third_party/README.md) names
each upstream, and [`../../NOTICE`](../../NOTICE) carries the copyright notices.
