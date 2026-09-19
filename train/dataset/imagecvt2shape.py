"""Pair renders and a supervoxel layout with the shape tokens that fill it.
"""
import glob
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import load_view_features, PAD_COORD, jitter_coords, load_npz, morton_order


class ImageCvtToShapeDataset(Dataset):
    """(view features, supervoxel centers) -> shape tokens."""

    TOKEN_KEYS = ("indices", "v9_indices", "v7_indices")
    COORD_KEYS = ("cvt_points", "indices_coords")

    def __init__(self, token_dir: str, condition_dir: str, global_mean_path: str,
                 num_views: int = 1, max_tokens: int = None, jitter_vox: float = 0.0,
                 max_samples: int = None, grid: int = 64, repeat_factor: int = 1,
                 train: bool = True, feature_suffix: str = ".npz",
                 feature_key: str = "features_518", allowed_ids=None):
        self.token_dir = token_dir
        self.condition_dir = condition_dir
        self.num_views = num_views
        self.max_tokens = max_tokens
        self.grid = grid
        self.train = train
        self.feature_suffix = feature_suffix
        self.feature_key = feature_key
        self.jitter_vox = float(jitter_vox) if train else 0.0

        if not global_mean_path or not os.path.exists(global_mean_path):
            raise FileNotFoundError(
                "the corpus feature mean is required: vision-tower patch features carry a large "
                "component identical for every object, which dominates the projection's input "
                "scale and leaves the useful variation attenuated")
        self.global_mean = np.load(global_mean_path).astype(np.float32)

        files = sorted(glob.glob(os.path.join(token_dir, "*.npz")))
        self.files = []
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            if (allowed_ids is None or stem in allowed_ids) and \
                    os.path.exists(os.path.join(condition_dir, stem + feature_suffix)):
                self.files.append(f)
        if max_samples:
            self.files = self.files[:max_samples]
        # See textcvt2shape: the unique list stays in `self.files`, only the indexed one repeats,
        # so a one-object dataset can fill an epoch without thousands of copies on disk.
        self.items = self.files * repeat_factor
        print(f"[imagecvt2shape] {len(self.files)} objects with both tokens and views "
              f"(of {len(files)} token files), {num_views} view(s) per sample "
              f"x {repeat_factor} = {len(self.items)}")
        if not self.files:
            raise RuntimeError(f"no object has both a token file in {token_dir} and view "
                               f"features in {condition_dir}")

    def __len__(self):
        return len(self.items)

    def _pick(self, data, keys, path):
        for k in keys:
            if k in data:
                return data[k]
        raise KeyError(f"none of {keys} present in {path}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.items[idx]
        stem = os.path.splitext(os.path.basename(path))[0]
        data = load_npz(path)
        tokens = self._pick(data, self.TOKEN_KEYS, path).flatten()
        coords = np.asarray(self._pick(data, self.COORD_KEYS, path), np.float32)

        feat = load_view_features(os.path.join(self.condition_dir, stem),
                                  self.feature_suffix, self.feature_key)
        total = feat.shape[0]
        k = min(self.num_views, total)
        sel = np.random.choice(total, k, replace=False) if (self.train and k < total) else np.arange(k)
        cond = feat[sel].astype(np.float32).reshape(-1, feat.shape[-1]) - self.global_mean

        # Order: FPS/seed order, as in textcvt2shape.py -- see that file for why. Both stage-2
        # datasets must use the same rule, so a change here belongs in both.
        #
        # order = morton_order(coords, self.grid)
        # tokens, coords = tokens[order], coords[order]
        #
        # Applied after the ordering decision: jitter perturbs coordinates, not sequence order.
        coords = jitter_coords(coords, self.jitter_vox, self.grid)

        if self.max_tokens and len(tokens) > self.max_tokens:
            tokens, coords = tokens[:self.max_tokens], coords[:self.max_tokens]

        return {"cond_features": torch.from_numpy(cond),
                "token_ids": torch.from_numpy(np.ascontiguousarray(tokens)).long(),
                "token_coords": torch.from_numpy(np.ascontiguousarray(coords)).float()}


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    T = max(b["cond_features"].shape[0] for b in batch)
    N = max(b["token_ids"].shape[0] for b in batch)
    D = batch[0]["cond_features"].shape[-1]

    cond = torch.zeros(B, T, D, dtype=torch.float32)
    # Every patch of every sampled view is real; the mask exists because samples may carry
    # different view counts, not because anything inside a view is padding.
    cmask = torch.zeros(B, T, dtype=torch.long)
    ids = torch.zeros(B, N, dtype=torch.long)
    coords = torch.full((B, N, 3), PAD_COORD, dtype=torch.float32)
    tmask = torch.zeros(B, N, dtype=torch.long)

    for i, b in enumerate(batch):
        t, n = b["cond_features"].shape[0], b["token_ids"].shape[0]
        cond[i, :t] = b["cond_features"]
        cmask[i, :t] = 1
        ids[i, :n] = b["token_ids"]
        coords[i, :n] = b["token_coords"]
        tmask[i, :n] = 1

    return {"cond_features": cond, "cond_mask": cmask,
            "token_ids": ids, "token_coords": coords, "token_mask": tmask}
