"""The 64-cube saliency fields the autoencoder trains on.
"""
import glob
import json
import os
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Key pairs, in the order they are tried. The first is what the 512-cube extraction wrote, with
# the view name as a prefix; the other two are what the coarse-field steps write.
_FALLBACK_KEYS = (("indices64", "saliency64"), ("idx64", "val"))


class SaliencyFieldDataset(Dataset):
    """Sparse saliency fields on a cubic grid, served dense.

    Args:
        roots: directory of `.npz` fields.
        view_key: prefix naming which extraction to read, when the file holds several.
        resolution: edge of the grid the volume is returned on. Indices from a different source
            resolution are rescaled.
        saliency_type: `percentile` or `zscore`, selecting which normalization to read. Only
            meaningful for the view-prefixed key pair.
        saliency_threshold: values below this are raised to it. 0 disables.
        value_range: kept for the trainer's sample rendering; the volume itself is always [0, 1].
        augment: random 90-degree rotations. Training only — it changes the field's orientation,
            so anything that has to reproduce a stored result must leave it off.
        caption_index: optional JSON mapping object id to captions. When given, only objects it
            lists are used; the captions themselves are not read by this model.
    """

    def __init__(self,
                 roots: str,
                 view_key: str = "",
                 resolution: int = 64,
                 saliency_type: Literal["percentile", "zscore"] = "percentile",
                 saliency_threshold: float = 0.0,
                 include_mesh: bool = False,
                 value_range: tuple = (0, 1),
                 augment: bool = False,
                 caption_index: Optional[str] = None,
                 skip_first_ids: int = 0):
        super().__init__()
        self.roots = roots
        self.view_key = view_key
        self.resolution = int(resolution)
        self.saliency_type = saliency_type
        self.saliency_threshold = float(saliency_threshold)
        self.include_mesh = bool(include_mesh)
        self.augment = bool(augment)
        self.value_range = tuple(value_range)

        files = sorted(glob.glob(os.path.join(roots, "*.npz")))
        if not files:
            raise RuntimeError(f"no .npz fields under {roots}")

        # The caption index is a filter, not an input: it exists so a run can be restricted to the
        # objects the text stage will also see. Without it every field is used.
        index = caption_index or os.environ.get("T2S_CAPTION_INDEX", "")
        ids = None
        if index:
            if not os.path.exists(index):
                raise FileNotFoundError(f"caption index not found: {index}")
            with open(index) as f:
                records = json.load(f)
            ordered = ([str(row["id"]) for row in records] if isinstance(records, list)
                       else [str(k) for k in records])
            ids = set(ordered[int(skip_first_ids):])
        self.files = [f for f in files if ids is None or self._object_id(f, ids) in ids]
        if not self.files:
            example = os.path.basename(files[0])
            raise RuntimeError(
                f"{len(files)} fields under {roots}, none of them listed in {index}. "
                f"A field file is matched by its stem ({example!r} -> "
                f"{self._object_id(files[0], ids)!r}), falling back to the part before the first "
                f"underscore when the stem carries a suffix such as "
                f"'<id>_saliency_volume_64.npz'.")
        print(f"[saliency-vae] {len(self.files)} fields at {self.resolution}^3 from {roots}"
              + (f" ({len(files) - len(self.files)} filtered out)" if ids is not None else ""),
              flush=True)

    @staticmethod
    def _object_id(path: str, ids=None) -> str:
        """Object id for a field file.

        The stem is tried whole first, because ids may themselves contain underscores
        ('sailing_ship.npz'), and splitting on the first one would cut the id in half. Only when
        the stem is not a known id is the suffix stripped, which is the
        '<id>_saliency_volume_64.npz' layout. Splitting first, on a stem that carried no suffix,
        also used to leave the '.npz' attached and match nothing.
        """
        stem = os.path.splitext(os.path.basename(path))[0]
        if ids is None or stem in ids:
            return stem
        return stem.split("_")[0]

    def __len__(self) -> int:
        return len(self.files)

    # -- reading -------------------------------------------------------------------------------
    def _read_pair(self, data, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Indices and saliency, from whichever key pair this file happens to carry."""
        if self.view_key:
            idx_key = f"{self.view_key}_voxel_indices"
            sal_key = f"{self.view_key}_voxel_saliency_{self.saliency_type}"
            if idx_key in data and sal_key in data:
                return data[idx_key], data[sal_key]
        for idx_key, sal_key in _FALLBACK_KEYS:
            if idx_key in data and sal_key in data:
                return data[idx_key], data[sal_key]
        raise KeyError(
            f"{path} has none of the expected key pairs. Looked for "
            f"'{self.view_key}_voxel_indices' + '..._voxel_saliency_{self.saliency_type}', "
            f"{' or '.join('/'.join(p) for p in _FALLBACK_KEYS)}. It holds: {list(data.files)[:8]}")

    def _threshold(self, saliency: np.ndarray) -> np.ndarray:
        s = np.asarray(saliency, np.float32).copy()
        if self.saliency_threshold > 0:
            s[s < self.saliency_threshold] = self.saliency_threshold
        return np.clip(s, 0.0, 1.0)

    def _rescale(self, indices: np.ndarray, source: int) -> np.ndarray:
        if source == self.resolution:
            return indices
        scaled = (indices * (self.resolution / source)).astype(np.int32)
        return np.clip(scaled, 0, self.resolution - 1)

    def _rotate(self, indices: np.ndarray) -> np.ndarray:
        """A random multiple of 90 degrees about a random axis, in index space."""
        axis = np.random.randint(0, 3)
        quarters = np.random.randint(0, 4)
        if quarters == 0:
            return indices
        c = self.resolution / 2.0
        p = indices.astype(np.float64) - c
        for _ in range(quarters):
            if axis == 0:
                p = np.stack([p[:, 0], -p[:, 2], p[:, 1]], 1)
            elif axis == 1:
                p = np.stack([p[:, 2], p[:, 1], -p[:, 0]], 1)
            else:
                p = np.stack([-p[:, 1], p[:, 0], p[:, 2]], 1)
        p = np.round(p + c).astype(np.int32)
        return np.clip(p, 0, self.resolution - 1)

    def _densify(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Sparse indices to a dense cube, background -1, duplicates resolved by maximum."""
        r = self.resolution
        flat = torch.full((r ** 3,), -1.0, dtype=torch.float32)
        if len(indices):
            ok = (indices >= 0).all(1) & (indices < r).all(1)
            idx, val = indices[ok], values[ok]
            if len(idx):
                lin = idx[:, 0] * r * r + idx[:, 1] * r + idx[:, 2]
                try:
                    from torch_scatter import scatter_max
                    flat, _ = scatter_max(val, lin, out=flat)
                except ImportError:
                    # PyTorch 2.x fallback with the same duplicate-cell maximum semantics.
                    flat.scatter_reduce_(0, lin, val, reduce="amax", include_self=True)
        return flat.view(r, r, r)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.files[index]
        with np.load(path, allow_pickle=True) as data:
            indices, saliency = self._read_pair(data, path)
            indices = np.asarray(indices)
            source = int(data["voxel_resolution"][0]) if "voxel_resolution" in data \
                else (64 if "indices64" in data or "idx64" in data else 512)
            mesh = None
            if self.include_mesh and "mesh_vertices" in data and "mesh_faces" in data:
                mesh = (np.asarray(data["mesh_vertices"]), np.asarray(data["mesh_faces"]))

        saliency = self._threshold(saliency)
        indices = self._rescale(indices, source)
        if self.augment:
            indices = self._rotate(indices)

        volume = self._densify(torch.from_numpy(np.ascontiguousarray(indices)).long(),
                               torch.from_numpy(saliency).float())
        volume = (volume + 1) / 2          # background -1 -> 0, saliency 1 -> 1

        out: Dict[str, Any] = {"file_path": path, "ss": volume.unsqueeze(0)}
        # Occupancy is thresholded from saliency by the trainer, so the two cannot disagree.
        out["ss_saliency"] = out["ss"]
        if mesh is not None:
            out["mesh_vertices"] = torch.from_numpy(mesh[0]).float()
            out["mesh_faces"] = torch.from_numpy(mesh[1]).long()
        return out

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}\n"
                f"  - fields: {len(self.files)}\n"
                f"  - resolution: {self.resolution}\n"
                f"  - root: {self.roots}")
