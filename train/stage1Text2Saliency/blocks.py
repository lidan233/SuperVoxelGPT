"""Building blocks for the saliency generator.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class CoordSelfAttentionBlock(nn.Module):
    """Pre-norm self-attention + MLP, used inside the coordinate encoder."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout1(attn_out)
        return x + self.mlp(self.norm2(x))


class FourierCoordinateEncoder(nn.Module):
    """Coordinates -> embeddings, via Fourier features followed by self-attention.

    Padding positions arrive as -100.0 and are zeroed both before and after the attention stack,
    so they contribute nothing and receive nothing.
    """

    def __init__(
        self,
        embed_dim: int,
        num_freq_bands: int = 64,
        max_freq: float = 32.0,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_freq_bands = num_freq_bands
        self.register_buffer("freq_bands", 2.0 ** torch.linspace(0, math.log2(max_freq), num_freq_bands))

        fourier_dim = 3 * 2 * num_freq_bands + 3          # sin+cos per axis per band, plus raw xyz
        self.input_proj = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList([
            CoordSelfAttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, coords: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = coords.shape
        # padding is -100.0; sin/cos of that is meaningless, so neutralize before encoding
        safe = torch.where(coords < -1.0, torch.zeros_like(coords), coords)

        scaled = safe.unsqueeze(-1) * self.freq_bands * math.pi          # [B,N,3,K]
        fourier = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).reshape(B, N, -1)
        h = self.input_proj(torch.cat([safe, fourier], dim=-1))

        key_padding_mask = None
        if mask is not None:
            h = h * mask.unsqueeze(-1).float()
            key_padding_mask = mask == 0

        for layer in self.layers:
            if self.training and torch.is_grad_enabled():
                h = checkpoint(layer, h, key_padding_mask, use_reentrant=False)
            else:
                h = layer(h, key_padding_mask=key_padding_mask)

        h = self.output_proj(h)
        if mask is not None:
            h = h * mask.unsqueeze(-1).float()
        return h


class SelfAttention(nn.Module):
    """Multi-head self-attention over the token field, via scaled_dot_product_attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must divide by num_heads"
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_p = dropout
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        shape = (B, N, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            m = key_padding_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, N, -1)   # [B,1,N,N]
            attn_mask = torch.where(m, float("-inf"), 0.0).to(q.dtype)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0, is_causal=False,
        )
        return self.out_proj(out.transpose(1, 2).reshape(B, N, C))


class TransformerBlockWithCrossAttn(nn.Module):
    """Self-attention over positions, cross-attention into the condition, then an MLP."""

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 4
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.self_attn = SelfAttention(embed_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def _cross_mask(self, B, N, T, x_mask, context_mask, device, dtype):
        """Additive [-inf/0] mask of shape [B*heads, N, T]: padded queries attend to nothing, and
        padded condition keys are attended by nobody."""
        if x_mask is None and context_mask is None:
            return None
        blocked = torch.zeros(B, N, T, dtype=torch.bool, device=device)
        if x_mask is not None:
            blocked |= (x_mask == 0).unsqueeze(-1).expand(B, N, T)
        if context_mask is not None:
            blocked |= (context_mask == 0).unsqueeze(1).expand(B, N, T)
        blocked = blocked.unsqueeze(1).expand(B, self.num_heads, N, T).reshape(B * self.num_heads, N, T)
        return torch.zeros_like(blocked, dtype=dtype).masked_fill(blocked, float("-inf"))

    def forward(
        self,
        x: torch.Tensor,                                  # [B,N,C]
        context: torch.Tensor,                            # [B,T,C]
        context_mask: Optional[torch.Tensor] = None,      # [B,T]
        x_mask: Optional[torch.Tensor] = None,            # [B,N]
    ) -> torch.Tensor:
        B, N, _ = x.shape
        T = context.shape[1]
        ckpt_on = self.training and torch.is_grad_enabled()

        h = self.norm1(x)
        spm = (x_mask == 0) if x_mask is not None else None
        if ckpt_on:
            attn_out = checkpoint(lambda q, m: self.self_attn(q, m), h, spm, use_reentrant=False)
        else:
            attn_out = self.self_attn(h, spm)
        x = x + self.dropout1(attn_out)

        h = self.norm2(x)
        cm = self._cross_mask(B, N, T, x_mask, context_mask, x.device, x.dtype)
        if ckpt_on:
            cross_out = checkpoint(
                lambda q, k, v, m: self.cross_attn(q, k, v, attn_mask=m, need_weights=False)[0],
                h, context, context, cm, use_reentrant=False)
        else:
            cross_out, _ = self.cross_attn(h, context, context, attn_mask=cm, need_weights=False)
        x = x + self.dropout2(cross_out)

        return x + self.mlp(self.norm3(x))
