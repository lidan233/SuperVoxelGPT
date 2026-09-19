# vendor

Upstream code, copied in so this repository has no dependency on the private framework it came
from. The directory layout mirrors the original package, which is why every relative import inside
here is untouched — the fewer edits made to vendored code, the less can silently diverge from the
version the released checkpoints were trained with.

| path | what |
|---|---|
| `modules/sparse/` | `SparseTensor` and the convolution / attention backend dispatch |
| `modules/` | attention, norm, spatial helpers |
| `models/sc_vaes/` | the autoencoder backbone the supervoxel VAE is assembled from |
| `trainers/` | `BasicTrainer` and the subdivision-VAE trainers |
| `utils/` | data, distributed, elastic-batching, gradient-clipping helpers |

`flash-attn` stays a pip dependency, selected by name in `modules/sparse/config.py`. The CUDA
extensions this tree calls into are vendored one level up, in `train/vendor/runtime/`.

## What was changed

An unused `Mesh` import was dropped from the subdivision trainers, removing a `cumesh` and
`flex_gemm` dependency the production path never takes.

The rest is upstream as copied. That claim is checked rather than asserted for the parts this
repository loads: the released encoder and decoder checkpoints load with `strict=True` through this
code, with the original package absent from `sys.modules`. It is not a byte-level comparison — no
upstream tree is reachable from here to diff against.
