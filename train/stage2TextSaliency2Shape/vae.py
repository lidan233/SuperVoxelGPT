"""The supervoxel autoencoder: the alphabet the shape generator writes in, and its inverse.
"""
import copy
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from torch_scatter import scatter_mean

from ..vendor.trellis2.modules import sparse as sp
from ..vendor.trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
from .sparse_vae.encoder_cvt import SparseUnetVaeEncoderCVT
from .sparse_vae.layers import (
    BidirGridToVoronoiAttention,
    BidirVoronoiToGridAttention,
    PointFeatureKNNCrossAttentionFourierMaskCkpt,
    _DEFAULT_SHAPE_SLAT_NORMALIZATION,
    _cvt_points_to_voronoi_points,
    compute_bidir_edges,
)
from .sparse_vae.quantizer import ResidualFSQ


class StatefulVoronoiToGridAttention(BidirVoronoiToGridAttention):
    """Read-back whose query is the grid's current state rather than its position alone.

    Adds no parameters — it only changes what is fed to the existing projections — so it is
    interchangeable with its base as far as a checkpoint is concerned. Position enters the query
    only, never the residual stream, so repeated rounds cannot accumulate positional bias.
    """

    def forward(self, grid_features, voronoi_features, voronoi_xyz, grid_xyz, edge_index):
        M, N = voronoi_features.shape[0], grid_xyz.shape[0]
        edge_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
        rel_pos = voronoi_xyz[edge_rev[0]] - grid_xyz[edge_rev[1]]
        grid_query = grid_features
        if self.pe_bias:
            grid_query = grid_query + self.fourier_p_bias(grid_xyz)
        out = self.propagate(
            edge_rev, query=self.linear_q(grid_query),
            key=self.linear_k(voronoi_features), value=self.linear_v(voronoi_features),
            rel_pos=rel_pos, size=(M, N),
        )
        out = self.norm1(self.out_proj(out) + self.skip_proj(grid_query))
        return self.norm2(out + self.ffn(out))


class SuperVoxelEncoder(nn.Module):
    """Geometry and centers in, one codebook index per center out.

    Submodules are created in the same order and under the same names the subclass chain produced
    them, so a checkpoint written by that chain loads here with no key remapping.
    """

    def __init__(self, model_channels: List[int], latent_channels: int,
                 num_blocks: List[int], block_type: List[str],
                 down_block_type: List[str], block_args: List[Dict[str, Any]], *,
                 vq_levels: Optional[List[int]] = None, num_quantizers: int = 2,
                 quantize_dropout: bool = True,
                 apply_latent_norm: bool = True,
                 latent_norm: Optional[Dict[str, Any]] = None,
                 latent_norm_eps: float = 1e-6,
                 apply_latent_denorm_to_decoder: bool = True,
                 pre_k: int = 32, bidir_k: int = 16, attn_groups: int = 8,
                 num_freq_bands: int = 64, max_freq: float = 32.0,
                 distance_mask_threshold: Optional[float] = None,
                 num_pre_attn_layers: int = 2, num_post_attn_layers: int = 2,
                 ffn_mult: int = 4, num_readout_rounds: int = 5,
                 use_fp16: bool = False, use_checkpoint: bool = True):
        super().__init__()
        self.use_fp16 = bool(use_fp16)
        self.use_checkpoint = bool(use_checkpoint)
        self.latent_channels = int(latent_channels)
        self.downsample_levels = max(0, len(num_blocks) - 1)
        self.apply_latent_norm = bool(apply_latent_norm)
        self.latent_norm_eps = float(latent_norm_eps)
        self.apply_latent_denorm_to_decoder = bool(apply_latent_denorm_to_decoder)
        self.bidir_k = int(bidir_k)
        self.num_readout_rounds = max(1, int(num_readout_rounds))
        lc = self.latent_channels

        # Corpus statistics, non-persistent because they belong to the recipe, not the weights.
        if self.apply_latent_norm:
            ln = _DEFAULT_SHAPE_SLAT_NORMALIZATION if latent_norm is None else latent_norm
            self.register_buffer("latent_norm_mean",
                                 torch.tensor(ln["mean"], dtype=torch.float32).reshape(1, -1),
                                 persistent=False)
            self.register_buffer("latent_norm_std",
                                 torch.tensor(ln["std"], dtype=torch.float32).reshape(1, -1),
                                 persistent=False)

        self.encoder = SparseUnetVaeEncoderCVT(
            6, model_channels, latent_channels, num_blocks,
            block_type, down_block_type, block_args, use_fp16)
        self.quantizer = ResidualFSQ(
            levels=vq_levels, dim=latent_channels,
            num_quantizers=num_quantizers, quantize_dropout=quantize_dropout)
        self.pre_quantize_attns = nn.ModuleList([
            PointFeatureKNNCrossAttentionFourierMaskCkpt(
                latent_channels, attn_groups, pre_k, 0., True, False, True, True,
                num_freq_bands, max_freq, distance_mask_threshold)
            for _ in range(num_pre_attn_layers)])
        self.grid_to_voronoi_attn = BidirGridToVoronoiAttention(
            latent_channels, attn_groups, num_freq_bands, max_freq, 0., True, False, True, ffn_mult)
        self.voronoi_to_grid_attn = BidirVoronoiToGridAttention(
            latent_channels, attn_groups, num_freq_bands, max_freq, 0., True, False, True, ffn_mult)
        self.post_quantize_attns = nn.ModuleList([
            PointFeatureKNNCrossAttentionFourierMaskCkpt(
                latent_channels, attn_groups, pre_k, 0., True, False, True, True,
                num_freq_bands, max_freq, distance_mask_threshold)
            for _ in range(num_post_attn_layers)])

        # One weight-tied read-back block reused every round: fewer new parameters is what keeps a
        # warm start from being destabilized.
        self.iter_readout = StatefulVoronoiToGridAttention(
            lc, attn_groups, num_freq_bands, max_freq, 0., True, False, True, ffn_mult)
        self.readout_scale = nn.ParameterList(
            [nn.Parameter(torch.ones(lc) * 1e-4) for _ in range(self.num_readout_rounds)])

        # Cloned from the always-on post-quantization block rather than initialised fresh: a random
        # module inserted into a converged network damages what it is inserted into, and the scale
        # gating it then gets driven to zero to undo the damage.
        self.iter_selfattn = copy.deepcopy(self.post_quantize_attns[0])
        self.sa_scale = nn.ParameterList(
            [nn.Parameter(torch.ones(lc) * 1e-4) for _ in range(self.num_readout_rounds)])

    def _ckpt_attn(self, attn, f_A, c_A, c_B, f_B):
        if self.use_checkpoint and self.training:
            return ckpt.checkpoint(attn, f_A, c_A, c_B, f_B, use_reentrant=False)
        return attn(f_A, c_A, c_B, f_B)

    def _bridge(self, attn, *inp):
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return ckpt.checkpoint(attn, *inp, use_reentrant=False)
        return attn(*inp)

    def quantize_voronoi(self, voronoi_features: torch.Tensor, voronoi_batch_indices: torch.Tensor):
        """Quantize per object, padding a ragged batch to a rectangle.

        Objects carry different numbers of centers. Padding by repeating each object's last entry
        keeps the quantizer's batch dimension rectangular without inventing values that would shift
        its statistics, and the mask drops the padding again.
        """
        batch_ids, counts = torch.unique(voronoi_batch_indices, return_counts=True)
        if batch_ids.numel() == 0:
            q, idx = self.quantizer(voronoi_features.unsqueeze(0))
            return q.squeeze(0), idx.squeeze(0)
        max_len = int(counts.max().item())
        bsz = int(counts.numel())
        base = torch.cat([torch.tensor([0], device=counts.device),
                          torch.cumsum(counts, dim=0)[:-1]])
        pos = torch.arange(max_len, device=voronoi_features.device).unsqueeze(0).expand(bsz, max_len)
        gather_idx = torch.minimum(pos, (counts - 1).unsqueeze(1)) + base.unsqueeze(1)
        q_feat, q_idx = self.quantizer(voronoi_features[gather_idx])
        mask = pos < counts.unsqueeze(1)
        return q_feat[mask], q_idx[mask]

    def forward(self, vertices, intersected, cvt_points) -> Dict[str, Any]:
        dtype = torch.float16 if self.use_fp16 else torch.float32
        x = vertices.replace(torch.cat([
            vertices.feats.to(dtype=dtype) - 0.5,
            intersected.feats.to(dtype=dtype) - 0.5,
        ], dim=1))

        z_e = self.encoder(x, sample_posterior=False, return_raw=False)
        if self.apply_latent_norm:
            mean = self.latent_norm_mean.to(z_e.feats.device, z_e.feats.dtype)
            std = self.latent_norm_std.to(z_e.feats.device, z_e.feats.dtype)
            z_e = z_e.replace((z_e.feats - mean) / (std + self.latent_norm_eps))

        input_res = int(vertices.coords[:, 1:].max().item() + 1) if vertices.coords.numel() > 0 else 1
        down_factor = 2 ** self.downsample_levels
        voronoi_points = _cvt_points_to_voronoi_points(
            cvt_points, input_resolution=input_res, downsample_factor=down_factor,
            device=z_e.feats.device)

        grid_coords_scaled = z_e.coords.float().clone()
        grid_coords_scaled[:, 1:] = grid_coords_scaled[:, 1:] / (input_res / down_factor - 1) - 0.5
        voronoi_coords_scaled = voronoi_points.float()
        grid_feats = z_e.feats

        for attn in self.pre_quantize_attns:
            grid_feats = self._ckpt_attn(attn, grid_feats, grid_coords_scaled,
                                         grid_coords_scaled, grid_feats)

        grid_xyz = grid_coords_scaled[:, 1:4].float()
        voronoi_xyz = voronoi_coords_scaled[:, 1:4].float()
        edges, v2g = compute_bidir_edges(grid_xyz, voronoi_xyz, self.bidir_k,
                                         grid_coords_scaled[:, 0].long(),
                                         voronoi_coords_scaled[:, 0].long())
        seed = scatter_mean(grid_feats[v2g[1]], v2g[0], dim=0, dim_size=voronoi_xyz.shape[0])
        voronoi_features = self._bridge(self.grid_to_voronoi_attn, grid_feats, grid_xyz,
                                        seed, voronoi_xyz, edges)

        z_q_voronoi, indices = self.quantize_voronoi(voronoi_features / 3, voronoi_points[:, 0])

        for attn in self.post_quantize_attns:
            z_q_voronoi = self._ckpt_attn(attn, z_q_voronoi, voronoi_coords_scaled,
                                          voronoi_coords_scaled, z_q_voronoi)

        z_q_grid = self._bridge(self.voronoi_to_grid_attn, z_q_voronoi, voronoi_xyz, grid_xyz, edges)

        h_v = z_q_voronoi
        for r in range(self.num_readout_rounds):
            sa = self._ckpt_attn(self.iter_selfattn, h_v, voronoi_coords_scaled,
                                 voronoi_coords_scaled, h_v)
            h_v = h_v + self.sa_scale[r] * (sa - h_v)
            read = self._bridge(self.iter_readout, z_q_grid, h_v, voronoi_xyz, grid_xyz, edges)
            z_q_grid = z_q_grid + self.readout_scale[r] * read

        return {"z_q": z_e.replace(z_q_grid), "indices": indices,
                "z_q_voronoi": z_q_voronoi, "z_voronoi": voronoi_features}


class SuperVoxelDecoder(SparseUnetVaeDecoder):
    """Latent in, surface out: vertex offsets, intersection logits, and subdivision decisions.

    Three variant layers collapsed into one. Neither of the two above this class added a submodule —
    one set two scalars and forwarded, the other added a float — so the flattening cannot move a
    single parameter name. What remains as a base is the framework's sparse U-Net, which is a
    component rather than a variant.
    """

    def __init__(self, resolution: int, model_channels: List[int], latent_channels: int,
                 num_blocks: List[int], block_type: List[str], up_block_type: List[str],
                 block_args: List[Dict[str, Any]], use_fp16: bool = False,
                 use_checkpoint: bool = True, voxel_margin: float = 0.5):
        # Set before the base __init__: it builds blocks that read these.
        self.resolution = int(resolution)
        self.use_checkpoint = bool(use_checkpoint)
        # Seven output channels: three vertex offsets, three intersection logits, one unused. The
        # spare channel is kept because removing it would change the output layer's shape and make
        # every existing checkpoint unloadable, for no gain.
        super().__init__(7, model_channels, latent_channels, num_blocks,
                         block_type, up_block_type, block_args, use_fp16)
        self.voxel_margin = float(voxel_margin)

    def set_resolution(self, resolution: int) -> None:
        self.resolution = int(resolution)

    def _forward_block(self, block, h):
        return block(h)

    def forward(self, x: "sp.SparseTensor", gt_intersected: "sp.SparseTensor" = None):
        h = self.from_latent(x)
        h = h.type(self.dtype)
        subs_gt, subs = [], []
        n_levels = len(self.blocks)

        for i, res in enumerate(self.blocks):
            for j, block in enumerate(res):
                is_upsample = (i < n_levels - 1 and j == len(res) - 1)
                if is_upsample:
                    if self.pred_subdiv:
                        if self.training:
                            # The ground-truth expansion is read from the encoder's spatial cache,
                            # so training expands on the true structure while the predicted
                            # subdivision is supervised against it rather than acted upon.
                            subs_gt.append(h.get_spatial_cache("subdivision"))
                        h, sub = block(h)
                        subs.append(sub)
                    else:
                        h = block(h)
                elif self.use_checkpoint and self.training and torch.is_grad_enabled():
                    h = ckpt.checkpoint(self._forward_block, block, h, use_reentrant=False)
                else:
                    h = block(h)

        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.output_layer(h)

        # Offsets are squashed into a range slightly wider than one cell. The margin lets a vertex
        # settle just outside its own cell, which is what allows a surface to pass exactly through
        # a cell boundary instead of being pinned inside one.
        vm = self.voxel_margin
        vertices_pred = h.replace((1.0 + 2.0 * vm) * torch.sigmoid(h.feats[..., 0:3]) - vm)
        intersected_pred = h.replace(h.feats[..., 3:6])

        if self.training:
            return vertices_pred, intersected_pred, subs_gt, subs
        return vertices_pred, intersected_pred, subs
