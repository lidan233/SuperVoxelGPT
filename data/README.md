# Data

Preparation pipeline for SuperVoxelGPT: raw meshes in, supervoxel centers out.
The centers are the deliverable here — turning them into tokens needs the
SuperVoxel VAE encoder, which ships with the model rather than with this code.

```
raw glb ─► 1_geometry ─► sharp watertight mesh + smoothed voxel occupancy (.vxz)
                              │                            │
                              └──► 2_saliency ─────────────┘─► N supervoxel centers (.ply)
                                                                      │
                                                       3_train_test_data ─► model datasets
```

## `data_prep/`

Three stages — geometry conditioning, saliency and adaptive sampling, then the captions and
encoders that turn a prepared object into model tensors. The reasoning for all three, and a worked
example with measured numbers at every step, is in
[`data_prep/README.md`](data_prep/README.md).

The CVT step, `2_saliency/d_cvt_pycvt.py`, places centers with the same annealed tessellation
`inference/text2shape.py` uses — one implementation, [`train/vendor/runtime/pycvt/`](../train/vendor/runtime/pycvt/),
so a corpus and serving agree by construction rather than by coincidence.

