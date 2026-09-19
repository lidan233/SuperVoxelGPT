"""Pair a caption with the saliency code it should produce.
"""
import glob
import os
import zlib
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import load_npz


def lattice_coords(grid: int) -> np.ndarray:
    """Centered coordinates of a cubic lattice, in the code's own raster order."""
    c = np.array([[z, y, x] for z in range(grid) for y in range(grid) for x in range(grid)],
                 dtype=np.float32)
    return (c / (grid - 1)) - 0.5


class TextToSaliencyDataset(Dataset):
    """(caption features) -> saliency code."""

    def __init__(self, code_dir: str, condition_dir: str, grid: int = 8,
                 code_key: str = "indices_flat", max_samples: int = None,
                 repeat_factor: int = 1, train: bool = True, allowed_ids=None):
        self.code_dir = code_dir
        self.condition_dir = condition_dir
        self.code_key = code_key
        self.train = train
        self.coords = lattice_coords(grid)

        stems = []
        for f in sorted(glob.glob(os.path.join(code_dir, "*.npz"))):
            stem = os.path.splitext(os.path.basename(f))[0]
            if (allowed_ids is None or stem in allowed_ids) and \
                    os.path.exists(os.path.join(condition_dir, stem + ".npz")):
                stems.append(stem)
        if max_samples:
            stems = stems[:max_samples]
        self.stems = stems
        self.items = stems * repeat_factor
        print(f"[text2saliency] {len(stems)} objects with both code and captions "
              f"x {repeat_factor} = {len(self.items)}")
        if not stems:
            raise RuntimeError(f"no object has both a code in {code_dir} and caption features "
                               f"in {condition_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stem = self.items[idx]
        with load_npz(os.path.join(self.code_dir, stem + ".npz")) as z:
            ids = z[self.code_key].astype(np.int64).flatten()
        feat = load_npz(os.path.join(self.condition_dir, stem + ".npz"))
        cond = feat["text_features"].astype(np.float32)
        cmask = feat["text_attention_mask"].astype(np.int64)
        if cond.ndim == 3:
            # Training resamples caption length. Validation assigns each object one stable
            # pseudo-random tier so checkpoints see the same prompt while the 100-object set
            # still covers short and long captions.
            i = (int(np.random.randint(len(cond))) if self.train
                 else zlib.crc32(stem.encode("utf-8")) % len(cond))
            cond, cmask = cond[i], cmask[i]
        return {"cond_features": torch.from_numpy(cond),
                "cond_mask": torch.from_numpy(cmask),
                "token_ids": torch.from_numpy(ids),
                "token_coords": torch.from_numpy(self.coords.copy())}


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    ids = torch.stack([b["token_ids"] for b in batch])
    return {"cond_features": torch.stack([b["cond_features"] for b in batch]),
            "cond_mask": torch.stack([b["cond_mask"] for b in batch]).long(),
            "token_ids": ids,
            "token_coords": torch.stack([b["token_coords"] for b in batch]),
            "token_mask": torch.ones(ids.shape, dtype=torch.long)}
