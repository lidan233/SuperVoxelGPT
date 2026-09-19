"""Generate the saliency code from a description.
"""
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .blocks import FourierCoordinateEncoder, TransformerBlockWithCrossAttn

SEQ_LEN = 512          # 8^3 lattice positions
CODEBOOK = 5625        # saliency VQ-VAE codebook


class BidirectionalTransformerSingleFlat(nn.Module):
    """Bidirectional transformer over the flat 512-token lattice.

    One extra embedding row past the codebook holds the mask token, and the same matrix produces
    the logits.
    """

    def __init__(
        self,
        num_tokens: int,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 24,
        hidden_dim: int = 2048,      # released checkpoint: blocks.N.mlp.0.weight is (2048, 768)
        dropout: float = 0.0,
        coord_num_freq_bands: int = 64,
        coord_max_freq: float = 32.0,
        coord_num_layers: int = 3,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.mask_token_id = num_tokens

        self.tok_emb = nn.Embedding(num_tokens + 1, embed_dim)
        self.coord_encoder = FourierCoordinateEncoder(
            embed_dim=embed_dim, num_freq_bands=coord_num_freq_bands, max_freq=coord_max_freq,
            num_layers=coord_num_layers, num_heads=num_heads, dropout=dropout,
        )
        self.ln_pre = nn.LayerNorm(embed_dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlockWithCrossAttn(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim, eps=1e-6),
        )
        self.bias = nn.Parameter(torch.zeros(num_tokens + 1))

        self._init_weights()
        self._gradient_checkpointing = False

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,                                 # [B,S] token ids, negatives treated as mask
        coords: torch.Tensor,                            # [B,S,3]
        context: torch.Tensor,                           # [B,T,C]
        context_mask: Optional[torch.Tensor] = None,     # [B,T]
        x_mask: Optional[torch.Tensor] = None,           # [B,S]
    ) -> torch.Tensor:
        h = self.tok_emb(torch.where(x < 0, self.mask_token_id, x)) + self.coord_encoder(coords, mask=x_mask)
        h = self.drop(self.ln_pre(h))
        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                h = checkpoint(block, h, context, context_mask, x_mask, use_reentrant=False)
            else:
                h = block(h, context, context_mask, x_mask)
        h = self.head(h)
        return torch.matmul(h, self.tok_emb.weight.T) + self.bias      # tied output projection


class Text3DModelMaskGITSingleCached(nn.Module):
    """Text -> saliency code, over a single 512-token stream, from precomputed CLIP features."""

    def __init__(
        self,
        text_hidden_dim: int = 768,
        num_tokens: int = CODEBOOK,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 24,
        # Defaults are the released configuration: the entry points and `inference/text2shape.py`
        # all pass these same values, and a model built with different ones cannot load
        # `stage1Text2Saliency.bin`.
        hidden_dim: int = 2048,
        dropout: float = 0.0,
        coord_num_freq_bands: int = 64,
        coord_max_freq: float = 32.0,
        coord_num_layers: int = 3,
        high_mask_prob: float = 0.0,
        verbose: bool = True,
    ):
        super().__init__()
        self._keys_to_ignore_on_save = None      # HF Trainer inspects this
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        # Target share of high-mask steps, reached at the end of training. Plain attributes, not
        # buffers: they must not enter state_dict or they would break checkpoint compatibility.
        self.high_mask_prob = high_mask_prob
        self._high_mask_ramp = 1.0

        # Identity when the widths already agree — kept as an attribute either way so the parameter
        # layout does not depend on the text encoder that happens to be in use.
        self.text_proj = nn.Linear(text_hidden_dim, embed_dim) if text_hidden_dim != embed_dim \
            else nn.Identity()

        self.transformer = BidirectionalTransformerSingleFlat(
            num_tokens=num_tokens, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
            hidden_dim=hidden_dim, dropout=dropout, coord_num_freq_bands=coord_num_freq_bands,
            coord_max_freq=coord_max_freq, coord_num_layers=coord_num_layers,
        )
        self.mask_token_id = self.transformer.mask_token_id
        self._gradient_checkpointing = False

        if verbose:
            n = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Text3DModelMaskGITSingleCached: dim={embed_dim} heads={num_heads} "
                  f"layers={num_layers} codebook={num_tokens} seq={SEQ_LEN} params={n:,}")

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self._gradient_checkpointing = True
        self.transformer._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False
        self.transformer._gradient_checkpointing = False

    def set_training_progress(self, progress: float) -> None:
        """Move the high-mask curriculum, `progress` running 0 -> 1 over the whole run.

        Call this from the training loop; nothing else depends on it, so evaluation and inference
        are unaffected and a model that is never told its progress behaves as if fully ramped.
        """
        self._high_mask_ramp = float(min(max(progress, 0.0), 1.0))

    @staticmethod
    def gamma_func(mode: str = "cosine"):
        """Mask schedule: fraction still masked after a given fraction of the steps."""
        if mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        if mode == "linear":
            return lambda r: 1 - r
        if mode == "square":
            return lambda r: 1 - r ** 2
        raise NotImplementedError(f"unknown schedule: {mode}")

    @staticmethod
    def _flatten_ids(token_3d_ids: torch.Tensor) -> torch.Tensor:
        if token_3d_ids.dim() == 3:
            assert token_3d_ids.shape[-1] == 1, f"expected [B,N,1], got {tuple(token_3d_ids.shape)}"
            return token_3d_ids[..., 0]
        return token_3d_ids

    def forward(
        self,
        text_features: torch.Tensor,                        # [B,T,768]
        text_attention_mask: torch.Tensor,                  # [B,T]
        token_3d_ids: torch.Tensor,                         # [B,S] or [B,S,1]
        token_3d_coords: torch.Tensor,                      # [B,S,3]
        token_3d_attention_mask: Optional[torch.Tensor] = None,
        mask_ratio: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        ids = self._flatten_ids(token_3d_ids)
        B, S = ids.shape
        device = ids.device
        valid = token_3d_attention_mask

        if mask_ratio is None and not self.training:
            # Validation mirrors generation: every code starts masked. This also makes eval_loss
            # deterministic enough to compare checkpoints instead of resampling a mask each time.
            mask_ratio = 1.0
        if mask_ratio is None:
            p_high = self.high_mask_prob * self._high_mask_ramp
            if p_high > 0 and np.random.uniform() < p_high:
                mask_ratio = float(np.clip(1.0 - abs(np.random.normal(0.0, 0.12)), 0.7, 1.0))
            else:
                mask_ratio = self.gamma_func("cosine")(np.random.uniform())
                if mask_ratio > 0.9:  # snap the top of the range to fully masked, as at inference
                    mask_ratio = 1.0
        num_to_mask = max(1, int(mask_ratio * S))

        # choose which positions to mask; padding is pushed to the end of the sort order
        rank = torch.rand(B, S, device=device)
        if valid is not None:
            rank = rank.masked_fill(valid == 0, 2.0)
        mask = torch.zeros(B, S, device=device, dtype=torch.bool)
        mask.scatter_(1, rank.sort(dim=1).indices[:, :num_to_mask], True)

        logits = self.transformer(
            torch.where(mask, self.mask_token_id, ids),
            token_3d_coords, self.text_proj(text_features), text_attention_mask, valid,
        )[:, :, :self.num_tokens]                          # drop the mask row from the vocabulary

        labels = ids.clone()
        labels[~mask] = -100                               # score only what was hidden
        if valid is not None:
            labels[valid == 0] = -100
        loss = F.cross_entropy(logits.reshape(-1, self.num_tokens).float(),
                               labels.reshape(-1), ignore_index=-100)

        with torch.no_grad():
            scored = mask & (valid == 1) if valid is not None else mask
            n = scored.sum().float()
            acc = ((logits.argmax(-1) == ids) & scored).sum().float() / n if n > 0 \
                else torch.zeros((), device=device)

        return {
            "loss": loss,
            "accuracy": acc,
            "logits": logits,
            "mask": mask,
            "mask_ratio": torch.tensor(mask_ratio, device=device),
        }

    @torch.no_grad()
    def generate(
        self,
        text_features: torch.Tensor,
        text_attention_mask: torch.Tensor,
        coords: torch.Tensor,
        num_steps: int = 12,
        token_3d_attention_mask: Optional[torch.Tensor] = None,
        mode: str = "cosine",
        verbose: bool = False,
    ) -> torch.Tensor:
        """Fill the lattice by confidence-ordered unmasking, in a fixed number of steps."""
        self.eval()
        B, S = text_features.shape[0], coords.shape[1]
        device = text_features.device
        valid = token_3d_attention_mask
        context = self.text_proj(text_features)
        gamma = self.gamma_func(mode)

        cur = torch.full((B, S), self.mask_token_id, device=device, dtype=torch.long)
        for step in range(num_steps):
            logits = self.transformer(cur, coords, context, text_attention_mask, valid)[:, :, :self.num_tokens]
            sampled = logits.argmax(-1)
            cur = torch.where(cur == self.mask_token_id, sampled, cur)

            if step < num_steps - 1:
                keep_masked = int(np.floor(S * gamma((step + 1) / num_steps)))
                if keep_masked > 0:
                    conf = F.softmax(logits, dim=-1).gather(-1, cur.unsqueeze(-1)).squeeze(-1)
                    cutoff = conf.sort(dim=-1).values[:, keep_masked:keep_masked + 1]
                    cur = torch.where(conf < cutoff, self.mask_token_id, cur)
                    if verbose:
                        print(f"  step {step+1}/{num_steps}: re-masked {keep_masked}")
        return cur

    @torch.no_grad()
    def evaluate(
        self,
        text_features, text_attention_mask, gt_tokens, gt_coords,
        token_3d_attention_mask=None, num_steps: int = 12, mode: str = "cosine", verbose: bool = True,
    ) -> Dict[str, torch.Tensor]:
        gt = self._flatten_ids(gt_tokens)
        pred = self.generate(text_features, text_attention_mask, gt_coords, num_steps=num_steps,
                             mode=mode, token_3d_attention_mask=token_3d_attention_mask, verbose=False)
        valid = (token_3d_attention_mask == 1) if token_3d_attention_mask is not None \
            else torch.ones_like(gt, dtype=torch.bool)
        n = int(valid.sum().item())
        acc = (((pred == gt) & valid).sum().item() / n) if n > 0 else 0.0
        if verbose:
            print(f"[eval] tokens={n} acc={acc*100:.2f}%")
        return {"pred_tokens": pred, "accuracy": torch.tensor(acc, device=gt.device)}
