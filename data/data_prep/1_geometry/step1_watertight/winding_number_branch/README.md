# Winding-number branch — the accurate path

Two ways to decide inside from outside. This one is the accurate one; `a_tile_extract.py` one level
up is the fast one. **If you have compute to spare, use this branch.** The winding number is the
most robust way to sign a distance field — considerably more so than ray stabbing.

```
a_floodfill.py        multi-scale floodfill → flood mask (which cells are reachable from outside)
b_sdf_winding.py      igl.signed_distance with SIGNED_DISTANCE_TYPE_WINDING_NUMBER on the band
c_extract_mesh.py     DiffDMC marching cubes over (flood mask, SDF)
d_largest_component.py   keep the largest connected component
```

## Why winding number is the accurate answer

The generalized winding number asks, for a query point, how many times the surface wraps around it —
an integral over the whole mesh. It is analytic and needs no assumption that the surface is closed,
so it stays meaningful on exactly the inputs that break topological methods: open boundaries,
self-intersections, flipped normals, interpenetrating parts. Where a shell has a hole, floodfill
leaks through it and declares the interior "outside"; the winding number simply reports a fractional
value and the surface still lands where the geometry says it should.

The cost is that it is an integral over all faces per query point. Even with a fast multipole
approximation it is orders of magnitude more expensive than a flood, which is why the production
path does not use it at 1024³ across tens of thousands of objects.

Note the two are combined here rather than opposed: floodfill first narrows the query set to the
band that matters, then the winding number assigns accurate signed values inside it.

## Choosing a branch

Distance is a BVH nearest-triangle query in every case; the methods differ only in the sign.

| | winding number (this branch) | flood labelling (`a_tile_extract.py`) | ray stabbing (available, unused) |
|---|---|---|---|
| how the sign is decided | integral over the surface | 6-connected components from the volume boundary | ray crossings, parity |
| assumes a sealed shell | no | **yes** | **yes** |
| assumes correct face orientation | no | no | no |
| cost per query | high | low | medium |
| failure mode | — | one hole flips an entire cavity | grazing hits on coplanar faces |
| tiling | not tiled — one volume at a time | tiled with globally consistent topology | — |
| used by the released dataset | no | yes | no |

(A fourth option, cubvh's `signed_distance` in `watertight` mode, takes the sign from the nearest
face's orientation. The sharpening step exposes it because the resulting field varies smoothly and
reduces staircasing, but it trusts face orientations that raw assets often get wrong.)

Use the winding-number branch when accuracy matters more than throughput: small batches, meshes
known to be messy, or when a floodfill result looks wrong (leaked interiors, inverted regions). Use
the fast path for dataset-scale work, where the band width and occupancy epsilon are what keep the
shell sealed enough for the flood to be correct.
