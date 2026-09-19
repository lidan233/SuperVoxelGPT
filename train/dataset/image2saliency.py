"""Pair renders with the saliency code they should produce.
"""
import glob
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import load_view_features, load_npz
from .text2saliency import lattice_coords


class ImageToSaliencyDataset(Dataset):
    """(view features) -> saliency code."""

    def __init__(self, code_dir: str, condition_dir: str, global_mean_path: str,
                 num_views: int = 1, grid: int = 8, code_key: str = "indices_flat",
                 max_samples: int = None, repeat_factor: int = 1, train: bool = True,
                 feature_suffix: str = ".npz",
                 feature_key: str = "features_518", allowed_ids=None):
        self.code_dir = code_dir
        self.condition_dir = condition_dir
        self.code_key = code_key
        self.num_views = num_views
        self.train = train
        self.feature_suffix = feature_suffix
        self.feature_key = feature_key
        self.coords = lattice_coords(grid)

        if not global_mean_path or not os.path.exists(global_mean_path):
            raise FileNotFoundError(
                "the corpus feature mean is required: without it the shared component of the "
                "patch features dominates and the object-specific variation is attenuated")
        self.global_mean = np.load(global_mean_path).astype(np.float32)

        stems = []
        for f in sorted(glob.glob(os.path.join(code_dir, "*.npz"))):
            stem = os.path.splitext(os.path.basename(f))[0]
            if (allowed_ids is None or stem in allowed_ids) and \
                    os.path.exists(os.path.join(condition_dir, stem + feature_suffix)):
                stems.append(stem)
        if max_samples:
            stems = stems[:max_samples]
        self.stems = stems
        self.items = stems * repeat_factor
        print(f"[image2saliency] {len(stems)} objects with both code and views "
              f"x {repeat_factor} = {len(self.items)}, {num_views} view(s) per sample")
        if not stems:
            raise RuntimeError(f"no object has both a code in {code_dir} and view features "
                               f"in {condition_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stem = self.items[idx]
        with load_npz(os.path.join(self.code_dir, stem + ".npz")) as z:
            ids = z[self.code_key].astype(np.int64).flatten()
        feat = load_view_features(os.path.join(self.condition_dir, stem),
                                  self.feature_suffix, self.feature_key)
        total = feat.shape[0]
        k = min(self.num_views, total)
        sel = np.random.choice(total, k, replace=False) if (self.train and k < total) else np.arange(k)
        cond = feat[sel].astype(np.float32).reshape(-1, feat.shape[-1]) - self.global_mean
        return {"cond_features": torch.from_numpy(cond),
                "token_ids": torch.from_numpy(ids),
                "token_coords": torch.from_numpy(self.coords.copy())}


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    T = max(b["cond_features"].shape[0] for b in batch)
    D = batch[0]["cond_features"].shape[-1]
    cond = torch.zeros(B, T, D, dtype=torch.float32)
    cmask = torch.zeros(B, T, dtype=torch.long)
    for i, b in enumerate(batch):
        t = b["cond_features"].shape[0]
        cond[i, :t] = b["cond_features"]
        cmask[i, :t] = 1
    ids = torch.stack([b["token_ids"] for b in batch])
    return {"cond_features": cond, "cond_mask": cmask,
            "token_ids": ids,
            "token_coords": torch.stack([b["token_coords"] for b in batch]),
            "token_mask": torch.ones(ids.shape, dtype=torch.long)}
