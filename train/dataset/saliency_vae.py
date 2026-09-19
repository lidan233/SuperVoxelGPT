"""Feed the saliency autoencoder the volumes it has to compress and rebuild.
"""
import glob
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import load_npz


class SaliencyVaeDataset(Dataset):
    """Sparse saliency fields, densified into the volume the autoencoder consumes."""

    def __init__(self, field_dir: str, grid: int = 64, index_key: str = "idx64",
                 value_key: str = "val", max_samples: int = None, repeat_factor: int = 1,
                 background: float = 0.0):
        self.field_dir = field_dir
        self.grid = grid
        self.index_key = index_key
        self.value_key = value_key
        self.background = background

        files = sorted(glob.glob(os.path.join(field_dir, "*.npz")))
        if max_samples:
            files = files[:max_samples]
        self.files = files * repeat_factor
        print(f"[saliency_vae] {len(files)} fields x {repeat_factor} = {len(self.files)}, "
              f"grid {grid}^3")
        if not files:
            raise RuntimeError(f"no npz fields under {field_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx]
        with load_npz(path) as z:
            ind = z[self.index_key].astype(np.int64)
            val = z[self.value_key].astype(np.float32)

        g = self.grid
        occupancy = np.zeros((g, g, g), dtype=np.float32)
        saliency = np.full((g, g, g), self.background, dtype=np.float32)
        occupancy[ind[:, 0], ind[:, 1], ind[:, 2]] = 1.0
        saliency[ind[:, 0], ind[:, 1], ind[:, 2]] = val

        volume = np.stack([occupancy, saliency], axis=0)      # [2, g, g, g]
        return {"volume": torch.from_numpy(volume),
                "occupancy": torch.from_numpy(occupancy),
                "saliency": torch.from_numpy(saliency),
                "stem": os.path.splitext(os.path.basename(path))[0]}


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {"volume": torch.stack([b["volume"] for b in batch]),
            "occupancy": torch.stack([b["occupancy"] for b in batch]),
            "saliency": torch.stack([b["saliency"] for b in batch]),
            "stem": [b["stem"] for b in batch]}
