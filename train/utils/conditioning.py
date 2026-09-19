"""Describe what a generator is conditioned on, so the generator itself does not have to care.

Purpose
    text2shape and image2shape are the same model. Only the thing it is told about the shape
    differs: a caption encoded by a text tower, or renders encoded by a vision tower. Keeping that
    difference in one place is what lets both modes share a transformer — and, more importantly,
    lets an image-conditioned run inherit every architectural fix that was made while tuning the
    text-conditioned one, instead of drifting into a second lineage that has to be re-tuned.

Input
    Precomputed condition features, one file per object.

Output
    A padded batch of condition features and the mask that says which entries are real.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class ConditionSpec:
    """Everything the model needs to know about a conditioning modality."""
    name: str
    feature_dim: int
    uses_mask: bool
    suffix: str
    description: str
    feature_key: str = "text_features"
    grid: int = 0


# The towers are part of the contract: features from a different encoder are a different condition.
TEXT_CLIP = ConditionSpec(
    name="text",
    feature_dim=768,
    uses_mask=True,
    suffix=".npz",
    description="CLIP ViT-L/14 text tower, last hidden state, padded to the tokenizer's max length",
)

IMAGE_DINOV2 = ConditionSpec(
    name="image",
    feature_dim=1024,
    uses_mask=False,
    suffix=".npz",
    feature_key="features_518",
    grid=37,
    description="DINOv2 ViT-L/14 patch features at 518px: one 37x37 grid of 1024-d tokens per "
                "rendered view. 518 rather than 224 because the patch size is 14 — at 224 the "
                "grid is 16x16 and a whole ornamented panel can fall inside a single patch, "
                "which is exactly the distinction the saliency stage needs to make",
)

SPECS = {s.name: s for s in (TEXT_CLIP, IMAGE_DINOV2)}


def get_spec(name):
    if name not in SPECS:
        raise KeyError(f"unknown condition {name!r}; known: {sorted(SPECS)}")
    return SPECS[name]


class ConditionLoader:
    """Read one object's condition off disk and put it in the form the model consumes."""

    def __init__(self, spec, root, num_views=1, global_mean_path=None, rng=None):
        self.spec = spec
        self.root = Path(root)
        self.num_views = num_views
        self.rng = rng if rng is not None else np.random
        self.global_mean = None
        if global_mean_path:
            self.global_mean = np.load(global_mean_path).astype(np.float32)
            if self.global_mean.shape != (spec.feature_dim,):
                raise ValueError(f"global mean has shape {self.global_mean.shape}, "
                                 f"expected ({spec.feature_dim},)")

    def path_for(self, stem):
        return self.root / f"{stem}{self.spec.suffix}"

    def load(self, stem, train=True):
        """Return (features [L, D] float32, mask [L] uint8 or None)."""
        if self.spec.name == "text":
            return self._load_text(stem)
        return self._load_image(stem, train)

    def _load_text(self, stem):
        with np.load(self.path_for(stem)) as z:
            feat = z["text_features"].astype(np.float32)
            mask = z["text_attention_mask"].astype(np.uint8)
        if feat.ndim == 3:
            # Several caption granularities were encoded for this object; pick one. Training on all
            # lengths is what makes the model accept a three-word prompt as readily as a sentence.
            i = int(self.rng.randint(len(feat)))
            feat, mask = feat[i], mask[i]
        return feat, mask

    def _load_image(self, stem, train):
        feat = np.load(self.path_for(stem))          # [V, P, P, D]
        total_views = feat.shape[0]
        k = min(self.num_views, total_views)
        if train and k < total_views:
            # Sampling views is the augmentation: the model must not learn one canonical camera.
            idx = self.rng.choice(total_views, k, replace=False)
        else:
            idx = np.arange(k)
        sel = feat[idx].astype(np.float32)
        sel = sel.reshape(-1, sel.shape[-1])          # [V*P*P, D]
        if self.global_mean is not None:
            sel = sel - self.global_mean
        return sel, None


def collate(batch_features, batch_masks, uses_mask):
    """Pad a list of [L_i, D] conditions into [B, L, D], building the mask when there is none."""
    lengths = [f.shape[0] for f in batch_features]
    L, D = max(lengths), batch_features[0].shape[1]
    feats = torch.zeros(len(batch_features), L, D, dtype=torch.float32)
    mask = torch.zeros(len(batch_features), L, dtype=torch.uint8)
    for i, f in enumerate(batch_features):
        n = f.shape[0]
        feats[i, :n] = torch.as_tensor(f)
        if uses_mask and batch_masks[i] is not None:
            mask[i, :n] = torch.as_tensor(batch_masks[i])
        else:
            # No per-entry padding within this sample: everything present is real. The mask still
            # exists so that variable-length batches (different view counts) stay correct.
            mask[i, :n] = 1
    return feats, mask
