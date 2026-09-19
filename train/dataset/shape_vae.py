"""Feed the supervoxel autoencoder a voxelization together with the centers it must tokenize onto.
"""
import glob
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..vendor.trellis2.modules import sparse as sp
from ..vendor.trellis2.utils.data_utils import load_balanced_group_indices


CENTER_SCALE = 256.0


def _import_o_voxel():
    try:
        import o_voxel
        from o_voxel import _C
        return o_voxel
    except Exception:
        repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        local_path = os.path.join(repo_dir, "o-voxel")
        if os.path.isdir(local_path):
            import sys
            sys.path.insert(0, local_path)
        import o_voxel
        from o_voxel import _C
        return o_voxel


@dataclass(frozen=True)
class ShapeVaeItem:
    id: str
    vxz_path: str
    cvt_path: str
    load: int
    n_voxels: int


class ShapeVaeDataset(Dataset):
    """(voxelization, supervoxel centers) pairs for training the supervoxel autoencoder."""

    def __init__(
        self,
        root: str,
        *,
        resolution: int = 1024,
        num_threads: int = 4,
        max_active_voxels: Optional[int] = None,
        include_cvt_points: bool = True,
        include_id: bool = True,
        cvt_root: Optional[str] = None,
        cvt_suffix: str = "_saliency_volume_64_supervoxel_center.ply",
        center_scale: float = CENTER_SCALE,
        cache_file: Optional[str] = None,
        caption_index: Optional[str] = None,
        skip_first_ids: int = 0,
    ):
        self.root = os.path.abspath(root)
        self.resolution = int(resolution)
        self.num_threads = int(num_threads)
        self.max_active_voxels = None if max_active_voxels is None else int(max_active_voxels)
        self.include_cvt_points = bool(include_cvt_points)
        self.include_id = bool(include_id)
        self.cvt_root = os.path.abspath(cvt_root) if cvt_root is not None else self.root
        self.cvt_suffix = str(cvt_suffix)
        self.center_scale = float(center_scale)
        self.value_range = (0, 1)

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"dataset root not found: {self.root}")

        self.cache_file = cache_file or os.path.join(self.root, "_voxel_counts_cache.json")
        self.cvt_path_map = self._build_cvt_path_map() if self.include_cvt_points else {}
        items = self._build_items_list(self._load_or_build_cache())

        if not caption_index:
            raise ValueError("caption_index is required: training must be restricted to its split")
        with open(caption_index) as f:
            records = json.load(f)
        ordered = ([str(row["id"]) for row in records] if isinstance(records, list)
                   else [str(k) for k in records])
        allowed = set(ordered[int(skip_first_ids):])
        before = len(items)
        items = [it for it in items if it.id in allowed]
        print(f"[shape_vae] {len(items)} training-split objects "
              f"({before - len(items)} non-training objects excluded)")

        if not items:
            raise RuntimeError(f"no object has both a .vxz under {self.root} and centers "
                               f"(*{self.cvt_suffix}) under {self.cvt_root}")
        self.items = items
        self.loads = [it.load for it in items]

    def _build_cvt_path_map(self) -> dict:
        cvt_map = {}
        for cvt_path in glob.glob(os.path.join(self.cvt_root, f"*{self.cvt_suffix}")):
            base = os.path.basename(cvt_path)
            stem = base[:-len(self.cvt_suffix)] if base.endswith(self.cvt_suffix) \
                else os.path.splitext(base)[0]
            cvt_map[stem] = cvt_path
        print(f"[shape_vae] {len(cvt_map)} center files under {self.cvt_root}")
        return cvt_map

    def _build_items_list(self, voxel_counts: dict) -> List[ShapeVaeItem]:
        items: List[ShapeVaeItem] = []
        skipped_voxels = skipped_no_cvt = 0
        for name in sorted(f for f in os.listdir(self.root) if f.endswith(".vxz")):
            stem = name[:-4]
            vxz_path = os.path.join(self.root, name)

            cvt_path = ""
            if self.include_cvt_points:
                cvt_path = self.cvt_path_map.get(stem)
                if cvt_path is None:
                    skipped_no_cvt += 1
                    continue

            n_voxels = voxel_counts.get(name, 0)
            if self.max_active_voxels is not None and n_voxels > self.max_active_voxels:
                skipped_voxels += 1
                continue

            items.append(ShapeVaeItem(id=stem, vxz_path=vxz_path, cvt_path=cvt_path,
                                      load=os.path.getsize(vxz_path), n_voxels=n_voxels))

        print(f"[shape_vae] {len(items)} objects with both a voxelization and centers")
        if skipped_no_cvt:
            print(f"[shape_vae]   {skipped_no_cvt} skipped: no centers file")
        if skipped_voxels:
            print(f"[shape_vae]   {skipped_voxels} skipped: over {self.max_active_voxels} voxels")
        return items

    def _load_or_build_cache(self) -> dict:
        names = sorted(f for f in os.listdir(self.root) if f.endswith(".vxz"))
        counts = {}
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file) as f:
                    counts = json.load(f)
            except Exception as e:
                print(f"[shape_vae] ignoring unreadable voxel-count cache: {e}")

        current = set(names)
        changed = any(name not in counts for name in names) or any(name not in current for name in counts)
        counts = {name: counts[name] for name in names if name in counts}
        missing = [name for name in names if name not in counts]

        if missing:
            o_voxel = _import_o_voxel()
            print(f"[shape_vae] counting voxels in {len(missing)} new files")
            for name in missing:
                try:
                    coords, _ = o_voxel.io.read_vxz(os.path.join(self.root, name),
                                                    num_threads=self.num_threads)
                    counts[name] = int(coords.shape[0])
                except Exception as e:
                    print(f"[shape_vae] unreadable {name}: {e}")
                    counts[name] = 0
        if changed:
            with open(self.cache_file, "w") as f:
                json.dump(counts, f)
        return counts

    def __len__(self) -> int:
        return len(self.items)

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}(n={len(self)}, root={self.root}, "
                f"resolution={self.resolution}, center_scale={self.center_scale})")

    def _read_cvt_points(self, cvt_path: str) -> sp.VarLenTensor:
        if not os.path.exists(cvt_path):
            return sp.VarLenTensor(torch.empty((0, 3), dtype=torch.float32))
        import trimesh
        obj = trimesh.load(cvt_path, process=False)
        if not hasattr(obj, "vertices"):
            return sp.VarLenTensor(torch.empty((0, 3), dtype=torch.float32))
        v = np.asarray(obj.vertices, dtype=np.float32) / self.center_scale - 0.5
        return sp.VarLenTensor(torch.from_numpy(v).float().clamp(-0.5, 0.5))

    def _read_dual_grid(self, vxz_path: str) -> Tuple[sp.SparseTensor, sp.SparseTensor]:
        o_voxel = _import_o_voxel()
        coords, attr = o_voxel.io.read_vxz(vxz_path, num_threads=self.num_threads)
        coords = coords.to(torch.int32)
        vertices = sp.SparseTensor(
            attr["vertices"].to(torch.float32) / 255.0,
            torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1),
        )
        intersected = vertices.replace(
            torch.cat([attr["intersected"] % 2,
                       attr["intersected"] // 2 % 2,
                       attr["intersected"] // 4 % 2], dim=-1).bool())
        return vertices, intersected

    def __getitem__(self, idx: int):
        for offset in range(len(self.items)):
            it = self.items[(idx + offset) % len(self.items)]
            if not os.path.exists(it.vxz_path):
                continue

            vertices, intersected = self._read_dual_grid(it.vxz_path)
            out = {"vertices": vertices, "intersected": intersected}
            if self.include_id:
                out["id"] = it.id
            if self.include_cvt_points:
                out["cvt_points"] = self._read_cvt_points(it.cvt_path)
                # Too few centers is a failed CVT, not a small shape; training on it teaches the
                # encoder to project onto a set it will never see at inference.
                if out["cvt_points"].feats.shape[0] < 100:
                    continue
            return out
        raise RuntimeError("no usable shape-VAE sample remains: check voxelizations and centers")

    @staticmethod
    def collate_fn(batch, split_size: Optional[int] = None):
        """Pack objects into one sparse batch, or into `split_size` load-balanced sub-batches.

        The trainer passes `split_size=batch_split` for gradient accumulation, and the groups are
        balanced by voxel count rather than by object count: objects differ in size by orders of
        magnitude, so an even split by count leaves one sub-batch far larger than the rest.
        """
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices(
                [b["vertices"].feats.shape[0] for b in batch], split_size)

        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}
            for k in sub_batch[0].keys():
                first = sub_batch[0][k]
                if isinstance(first, torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(first, sp.SparseTensor):
                    pack[k] = sp.sparse_cat([b[k] for b in sub_batch], dim=0)
                elif isinstance(first, sp.VarLenTensor):
                    pack[k] = sp.varlen_cat([b[k] for b in sub_batch], dim=0)
                elif isinstance(first, list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]
            packs.append(pack)

        return packs[0] if split_size is None else packs
