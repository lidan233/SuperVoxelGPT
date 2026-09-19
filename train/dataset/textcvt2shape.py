"""Pair a caption and a supervoxel layout with the shape tokens that fill it.
"""
import glob
import os
import zlib
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import PAD_COORD, jitter_coords, load_npz, morton_order


class TextCvtToShapeDataset(Dataset):
    """(caption features, supervoxel centers) -> shape tokens."""

    TOKEN_KEYS = ("indices", "v9_indices", "v7_indices")
    COORD_KEYS = ("cvt_points", "indices_coords")

    def __init__(self, token_dir: str, condition_dir: str, max_tokens: int = None,
                 jitter_vox: float = 0.0, max_samples: int = None, grid: int = 64,
                 repeat_factor: int = 1, train: bool = True, allowed_ids=None):
        self.token_dir = token_dir
        self.condition_dir = condition_dir
        self.max_tokens = max_tokens
        self.grid = grid
        self.train = train
        # Jitter is a training augmentation. Anything that has to be reproducible — extraction,
        # evaluation, inference — must leave it at zero.
        self.jitter_vox = float(jitter_vox) if train else 0.0
        if jitter_vox and not train:
            print("[textcvt2shape] jitter disabled: this is not a training split")

        files = sorted(glob.glob(os.path.join(token_dir, "*.npz")))
        self.files = [f for f in files
                      if (allowed_ids is None or
                          os.path.splitext(os.path.basename(f))[0] in allowed_ids)
                      and os.path.exists(os.path.join(condition_dir,
                                                      os.path.basename(f)))]
        if max_samples:
            self.files = self.files[:max_samples]
        # `repeat_factor` repeats the corpus within an epoch. The unique list stays in
        # `self.files`; only the indexed one repeats.
        self.items = self.files * repeat_factor
        print(f"[textcvt2shape] {len(self.files)} objects with both tokens and captions "
              f"(of {len(files)} token files) x {repeat_factor} = {len(self.items)}")
        if not self.files:
            raise RuntimeError(f"no object has both a token file in {token_dir} and a "
                               f"caption feature file in {condition_dir}")

    def __len__(self):
        return len(self.items)

    def _pick(self, data, keys, path):
        for k in keys:
            if k in data:
                return data[k]
        raise KeyError(f"none of {keys} present in {path}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.items[idx]
        stem = os.path.basename(path)
        data = load_npz(path)
        tokens = self._pick(data, self.TOKEN_KEYS, path).flatten()
        coords = np.asarray(self._pick(data, self.COORD_KEYS, path), np.float32)

        feat = load_npz(os.path.join(self.condition_dir, stem))
        cond = feat["text_features"].astype(np.float32)
        cmask = feat["text_attention_mask"].astype(np.int64)
        if cond.ndim == 3:
            # Several caption granularities were encoded; training on all of them is what makes the
            # model accept a three-word prompt as readily as a full sentence.
            # Validation uses one stable pseudo-random tier per object, so successive checkpoints
            # are compared on identical prompts while the set covers all caption lengths.
            i = (int(np.random.randint(len(cond))) if self.train
                 else zlib.crc32(os.path.splitext(stem)[0].encode("utf-8")) % len(cond))
            cond, cmask = cond[i], cmask[i]

        # order = morton_order(coords, self.grid)
        # tokens, coords = tokens[order], coords[order]
        #
        # Applied after the ordering decision: jitter perturbs coordinates, not sequence order.
        coords = jitter_coords(coords, self.jitter_vox, self.grid)

        if self.max_tokens and len(tokens) > self.max_tokens:
            tokens, coords = tokens[:self.max_tokens], coords[:self.max_tokens]

        return {"cond_features": torch.from_numpy(cond),
                "cond_mask": torch.from_numpy(cmask),
                "token_ids": torch.from_numpy(np.ascontiguousarray(tokens)).long(),
                "token_coords": torch.from_numpy(np.ascontiguousarray(coords)).float()}


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    T = max(b["cond_mask"].shape[0] for b in batch)
    N = max(b["token_ids"].shape[0] for b in batch)
    D = batch[0]["cond_features"].shape[-1]

    cond = torch.zeros(B, T, D, dtype=torch.float32)
    cmask = torch.zeros(B, T, dtype=torch.long)
    ids = torch.zeros(B, N, dtype=torch.long)
    coords = torch.full((B, N, 3), PAD_COORD, dtype=torch.float32)
    tmask = torch.zeros(B, N, dtype=torch.long)

    for i, b in enumerate(batch):
        t, n = b["cond_mask"].shape[0], b["token_ids"].shape[0]
        cond[i, :t] = b["cond_features"]
        cmask[i, :t] = b["cond_mask"]
        ids[i, :n] = b["token_ids"]
        coords[i, :n] = b["token_coords"]
        tmask[i, :n] = 1

    return {"cond_features": cond, "cond_mask": cmask,
            "token_ids": ids, "token_coords": coords, "token_mask": tmask}
