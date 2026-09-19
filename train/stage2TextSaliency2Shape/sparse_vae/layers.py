"""The layers the supervoxel autoencoder is assembled from.

Purpose
    One place for the pieces both halves of the autoencoder share: positional encoding, the KNN
    self-attention the grid is refined with, and the bidirectional cross-attention that moves
    features between the grid and the supervoxel centers.

Input / Output
    Nothing here is an entry point. `vae.py` composes these into the encoder and decoder that the
    released checkpoints load into.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, knn
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

import einops
import pointops

from ...vendor.trellis2.modules import sparse as sp


def _cvt_points_to_voronoi_points(cvt_points: sp.VarLenTensor, *, input_resolution: int,
                                   downsample_factor: int, device: torch.device) -> torch.Tensor:
    pts = cvt_points.feats.to(device=device, dtype=torch.float32)
    b = cvt_points.batch_boardcast_map.to(device=device, dtype=torch.int64).unsqueeze(-1)
    return torch.cat([b, pts], dim=-1)


# Measured over the training corpus, per latent channel. The encoder standardises its latent with
# these before quantization; they are part of the trained configuration, not a tunable.
_DEFAULT_SHAPE_SLAT_NORMALIZATION = {
    "mean": [0.781296, 0.018091, -0.495192, -0.558457, 1.060530, 0.093252, 1.518149, -0.933218,
             -0.732996, 2.604095, -0.118341, -2.143904, 0.495076, -2.179512, -2.130751, -0.996944,
             0.261421, -2.217463, 1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
             -0.059621, 2.220780, 1.621212, 0.877230, 0.567247, -3.175944, -3.186688, 1.578665],
    "std": [5.972266, 4.706852, 5.445010, 5.209927, 5.320220, 4.547237, 5.020802, 5.444004,
            5.226681, 5.683095, 4.831436, 5.286469, 5.652043, 5.367606, 5.525084, 4.730578,
            4.805265, 5.124013, 5.530808, 5.619001, 5.103930, 5.417670, 5.269677, 5.547194,
            5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207, 5.347008, 5.405691],
}


class FourierPositionalEmbedding(nn.Module):
    """Fourier positional embedding for 3D coordinates."""
    def __init__(self, embed_dim, num_freq_bands=64, max_freq=32.0, include_input=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_freq_bands = num_freq_bands
        self.max_freq = max_freq
        self.include_input = include_input

        freq_bands = 2.0 ** torch.linspace(0, math.log2(max_freq), num_freq_bands)
        self.register_buffer('freq_bands', freq_bands)

        fourier_dim = 3 * 2 * num_freq_bands
        if include_input:
            fourier_dim += 3

        self.linear = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, coords):
        original_shape = coords.shape[:-1]
        coords = coords.reshape(-1, 3)

        scaled_coords = coords.unsqueeze(-1) * self.freq_bands.unsqueeze(0).unsqueeze(0) * math.pi
        sin_coords = torch.sin(scaled_coords)
        cos_coords = torch.cos(scaled_coords)

        fourier_features = torch.cat([sin_coords, cos_coords], dim=-1)
        fourier_features = fourier_features.reshape(coords.shape[0], -1)

        if self.include_input:
            fourier_features = torch.cat([coords, fourier_features], dim=-1)

        embeddings = self.linear(fourier_features)
        embeddings = embeddings.reshape(*original_shape, self.embed_dim)
        return embeddings


class PointBatchNorm(nn.Module):
    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.LayerNorm(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.norm(input)


class FFN(nn.Module):
    """Feed-Forward Network with GELU activation."""
    def __init__(self, embed_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 4
        self.net = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, embed_dim), nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)


class PointFeatureKNNCrossAttentionFourierMaskCkpt(nn.Module):
    """
    KNN-based cross attention with Fourier positional encoding and hard distance masking.

    Args:
        embed_channels (int): Number of embedding channels.
        groups (int): Number of attention groups.
        k (int): Number of KNN neighbors. Default: 32.
        attn_drop_rate (float): Dropout rate for attention weights. Default: 0.
        qkv_bias (bool): If True, add bias to Q, K, V projections. Default: True.
        pe_multiplier (bool): If True, use multiplicative position encoding. Default: False.
        pe_bias (bool): If True, use additive position encoding bias. Default: True.
        use_skip (bool): If True, add skip connection. Default: True.
        num_freq_bands (int): Number of frequency bands for Fourier encoding. Default: 64.
        max_freq (float): Maximum frequency for Fourier encoding. Default: 32.0.
        distance_mask_threshold (float): Distance threshold for hard masking. Default: None (no masking).
    """
    def __init__(self,
                 embed_channels,
                 groups,
                 k=32,
                 attn_drop_rate=0.,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 use_skip=True,
                 num_freq_bands=64,
                 max_freq=32.0,
                 distance_mask_threshold=None,
                 ):
        super(PointFeatureKNNCrossAttentionFourierMaskCkpt, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.k = k
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias
        self.use_skip = use_skip

        # Distance masking (hard mask only)
        self.use_distance_mask = distance_mask_threshold is not None
        if self.use_distance_mask:
            self.register_buffer(
                'distance_threshold',
                torch.tensor(float(distance_mask_threshold))
            )

        # Query projection
        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        # Key projection
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        # Value projection
        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        # Fourier positional encoding
        if self.pe_multiplier:
            self.fourier_p_multiplier = FourierPositionalEmbedding(
                embed_dim=embed_channels,
                num_freq_bands=num_freq_bands,
                max_freq=max_freq,
                include_input=True
            )
        if self.pe_bias:
            self.fourier_p_bias = FourierPositionalEmbedding(
                embed_dim=embed_channels,
                num_freq_bands=num_freq_bands,
                max_freq=max_freq,
                include_input=True
            )

        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups)
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

        if self.use_skip:
            self.skip_proj = nn.Linear(embed_channels, embed_channels)
            self.norm = nn.LayerNorm(embed_channels)

    def _compute_distance_mask(self, rel_pos):
        """
        Compute hard distance-based mask for KNN neighbors.

        Args:
            rel_pos: [M, k, 3] relative positions from query to neighbors

        Returns:
            mask: [M, k] boolean mask (True for valid, False for masked)
        """
        distances = torch.norm(rel_pos, dim=-1)  # [M, k]
        # Use abs() for learnable threshold to ensure positive value
        threshold = torch.abs(self.distance_threshold)
        mask = distances <= threshold
        return mask

    def forward(self, features_A, coords_A, coords_B, features_B=None):
        """
        Args:
            features_A: [N, C] source point features
            coords_A: [N, 4] source coordinates (batch_idx, x, y, z)
            coords_B: [M, 4] query coordinates (batch_idx, x, y, z)
            features_B: [M, C] query point features, optional.
        Returns:
            features_B: [M, C] features for each query point
        """
        M = coords_B.shape[0]
        N = coords_A.shape[0]

        batch_A = coords_A[:, 0].long()
        xyz_A = coords_A[:, 1:4].contiguous().float()
        batch_B = coords_B[:, 0].long()
        xyz_B = coords_B[:, 1:4].contiguous().float()

        # Cross KNN query
        edge_index = knn(xyz_A, xyz_B, self.k, batch_A, batch_B)
        query_idx, src_idx = edge_index[0], edge_index[1]
        if N < self.k:
            reference_index = src_idx.view(M, N)
        else:
            reference_index = src_idx.view(M, self.k)

        # Generate query features
        if features_B is None:
            neighbor_feats = features_A[reference_index]
            features_B = neighbor_feats.mean(dim=1)

        features_B_input = features_B

        query = self.linear_q(features_B)
        key = self.linear_k(features_A)
        value = self.linear_v(features_A)

        # Gather key, value, and positions
        key = pointops.grouping(reference_index, key, xyz_A, xyz_B, with_xyz=True)
        value = pointops.grouping(reference_index, value, xyz_A, xyz_B, with_xyz=False)
        rel_pos, key = key[:, :, 0:3], key[:, :, 3:]

        # Compute attention relation
        relation_qk = key - query.unsqueeze(1)

        # Apply Fourier positional encoding
        if self.pe_multiplier:
            pem = self.fourier_p_multiplier(rel_pos)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.fourier_p_bias(rel_pos)
            relation_qk = relation_qk + peb
            value = value + peb

        # Compute attention weights
        weight = self.weight_encoding(relation_qk)  # [M, k, groups]

        # Apply hard distance mask if enabled
        if self.use_distance_mask:
            distance_mask = self._compute_distance_mask(rel_pos)  # [M, k]
            # Set masked positions to -inf before softmax
            mask_value = torch.finfo(weight.dtype).min
            weight = weight.masked_fill(~distance_mask.unsqueeze(-1), mask_value)

        weight = self.attn_drop(self.softmax(weight))

        # Grouped aggregation
        value = einops.rearrange(value, "m k (g i) -> m k g i", g=self.groups)
        feat = torch.einsum("m k g i, m k g -> m g i", value, weight)
        feat = einops.rearrange(feat, "m g i -> m (g i)")

        if self.use_skip:
            skip = self.skip_proj(features_B_input)
            feat = self.norm(feat + skip)

        return feat


# ---------- bidirectional KNN edge construction ----------

def compute_bidir_edges(grid_xyz: torch.Tensor, voronoi_xyz: torch.Tensor,
                        k: int,
                        grid_batch: torch.Tensor, voronoi_batch: torch.Tensor):
    """Build a unified bidirectional edge set between grid and voronoi.

    Returns:
        edges: [2, ≤ 2*k*max(N,M)]  edges[0]=grid idx, edges[1]=voronoi idx
        v2g:   [2, k*N_voronoi]     v2g[0]=voronoi idx, v2g[1]=grid idx
               (per-voronoi k nearest grid — handy for mean-pool seeding)

    Edge set contains:
      (a) for each grid, its k nearest voronoi  → guarantees grid has k voronoi sources
      (b) for each voronoi, its k nearest grid  → guarantees voronoi has k grid sources
    """
    # (a) for each grid (query), k nearest voronoi (key)
    e_a = knn(voronoi_xyz, grid_xyz, k, voronoi_batch, grid_batch)
    #   e_a[0] = grid index (query), e_a[1] = voronoi index (k nearest per query)

    # (b) for each voronoi (query), k nearest grid (key)
    v2g = knn(grid_xyz, voronoi_xyz, k, grid_batch, voronoi_batch)
    #   v2g[0] = voronoi index (query), v2g[1] = grid index (k nearest per voronoi)

    # Flip (b) to (grid, voronoi) order and union with (a)
    e_b_flip = torch.stack([v2g[1], v2g[0]], dim=0)
    edges = torch.cat([e_a, e_b_flip], dim=1)
    return edges, v2g


# ---------- Bidirectional cross-attention modules ----------

class _BidirCrossAttentionBase(MessagePassing):
    """Shared cross-attention body. Subclasses define forward direction.

    edge_index passed to forward() is in (grid, voronoi) order.
    """
    def __init__(self, embed_dim, num_heads=8, num_freq_bands=64, max_freq=32.0,
                 dropout=0.0, bias=True, pe_multiplier=False, pe_bias=True, ffn_mult=4):
        super().__init__(aggr=None, node_dim=0)
        self.embed_dim = embed_dim
        self.groups = num_heads
        self.group_dim = embed_dim // num_heads
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=bias),
            PointBatchNorm(embed_dim), nn.ReLU(True))
        self.linear_k = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=bias),
            PointBatchNorm(embed_dim), nn.ReLU(True))
        self.linear_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        if pe_multiplier:
            self.fourier_p_multiplier = FourierPositionalEmbedding(
                embed_dim, num_freq_bands, max_freq, True)
        if pe_bias:
            self.fourier_p_bias = FourierPositionalEmbedding(
                embed_dim, num_freq_bands, max_freq, True)
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_dim, self.groups), PointBatchNorm(self.groups),
            nn.ReLU(True), nn.Linear(self.groups, self.groups))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.skip_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn = FFN(embed_dim, embed_dim * ffn_mult, dropout)

    def message(self, query_i, key_j, value_j, rel_pos, index, ptr, dim_size):
        rel = key_j - query_i
        if self.pe_multiplier:
            rel = rel * self.fourier_p_multiplier(rel_pos)
        if self.pe_bias:
            peb = self.fourier_p_bias(rel_pos)
            rel = rel + peb
            value_j = value_j + peb
        weight = self.attn_drop(softmax(self.weight_encoding(rel), index, ptr, dim_size))
        self._cached_w, self._cached_v = weight, value_j
        return value_j

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        v = self._cached_v.view(-1, self.groups, self.group_dim)
        return scatter_add(v * self._cached_w.unsqueeze(-1), index, dim=0,
                           dim_size=dim_size).view(-1, self.embed_dim)


class BidirGridToVoronoiAttention(_BidirCrossAttentionBase):
    """grid → voronoi: aggregate grid features into voronoi using shared edges."""

    def forward(self, grid_features, grid_xyz, voronoi_features, voronoi_xyz, edge_index):
        N, M = grid_features.shape[0], voronoi_features.shape[0]
        # edge_index[0] = grid (source), edge_index[1] = voronoi (target)
        rel_pos = grid_xyz[edge_index[0]] - voronoi_xyz[edge_index[1]]
        out = self.propagate(
            edge_index, query=self.linear_q(voronoi_features),
            key=self.linear_k(grid_features), value=self.linear_v(grid_features),
            rel_pos=rel_pos, size=(N, M),
        )
        out = self.norm1(self.out_proj(out) + self.skip_proj(voronoi_features))
        return self.norm2(out + self.ffn(out))


class BidirVoronoiToGridAttention(_BidirCrossAttentionBase):
    """voronoi → grid: aggregate voronoi features into grid using shared edges (reversed)."""

    def forward(self, voronoi_features, voronoi_xyz, grid_xyz, edge_index, grid_skip=None):
        M, N = voronoi_features.shape[0], grid_xyz.shape[0]
        # Reverse edge direction: (voronoi, grid)
        edge_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
        rel_pos = voronoi_xyz[edge_rev[0]] - grid_xyz[edge_rev[1]]
        # Build a query for grid side. Grid features aren't easily available here
        # (they've gone through pre_quantize_attns), so we use Fourier PE as query
        # input — the same trick the superseded forward-only attention used.
        if self.pe_bias:
            grid_query = self.fourier_p_bias(grid_xyz)
        else:
            grid_query = torch.zeros(N, self.embed_dim,
                                     device=voronoi_features.device,
                                     dtype=voronoi_features.dtype)
        out = self.propagate(
            edge_rev, query=self.linear_q(grid_query),
            key=self.linear_k(voronoi_features), value=self.linear_v(voronoi_features),
            rel_pos=rel_pos, size=(M, N),
        )
        out = self.norm1(self.out_proj(out) + self.skip_proj(grid_query))
        return self.norm2(out + self.ffn(out))
