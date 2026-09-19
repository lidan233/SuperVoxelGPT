# cubvh int32 patch

**You only need this if you extract above 512³.** For ordinary resolutions, install cubvh
upstream as-is:

```bash
pip install git+https://github.com/ashawkey/cubvh
```

## What breaks without it

cubvh's floodfill indexes grid cells with `int32`. Past 2³¹ cells — anything above roughly 1290³ —
the index overflows and the flood produces garbage. At 1024³ (1.07 billion cells) stock cubvh is
still within range, but the margin is thin and the intermediate arithmetic is not, so this pipeline
patches it.

## What the patch does

Switches 55 sites in `include/gpu/floodfill.cuh` from `int` to `unsigned int` / `size_t`:

- kernel signatures take `unsigned int* labels` and `size_t Ntot`
- thread and neighbour indices become `size_t`
- the "unlabeled" sentinel changes from `-1` to `UINT_MAX`
- comparisons like `labels[n] >= 0` become `labels[n] != UINT_MAX`

Labels are only ever compared for equality, so the output tensor is still `int32` (4 bytes) and is
merely reinterpreted as unsigned inside the kernel — **memory usage does not change**.

## Applying it

Either run the script against a cloned cubvh source tree (it `git checkout`s the file first, so it
is idempotent):

```bash
python patch_cubvh2.py        # rewrites include/gpu/floodfill.cuh in place
pip install -e .              # rebuild cubvh
```

…or skip the regex entirely and copy the finished file:

```bash
cp floodfill.cuh.PATCHED <cubvh>/include/gpu/floodfill.cuh
pip install -e <cubvh>
```

`floodfill.cuh.PATCHED` is the post-patch file in full (273 lines) — useful for diffing against
upstream to see exactly what changed.

## Verified behaviour

| resolution | result |
|---|---|
| R ≤ 512 | bit-identical to stock cubvh |
| R = 1024 | passes, 19.3 GB peak |
| R ≥ 1280 | **still fails** |

The remaining failure is not an indexing problem. floodfill internally `cudaMalloc`s a second copy
of its working set (3.6 GB grid + 14.5 GB labels at 1536³), which exceeds a 24 GB card on top of
the PyTorch tensors already resident. Fixing that needs an in-place floodfill — out of scope here.
