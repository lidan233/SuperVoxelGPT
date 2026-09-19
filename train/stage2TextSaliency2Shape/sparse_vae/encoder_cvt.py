"""The encoder half of the supervoxel autoencoder's U-Net.

Purpose
    Sparse geometry in, a latent per supervoxel out. This is the branch that differs from the
    framework's own encoder: it takes CVT centers alongside the grid, so the latent it produces is
    already per-center rather than per-voxel.

Input
    A sparse tensor of geometry features, built by the blocks named in the config.

Output
    A sparse tensor of `2 * latent_channels` — mean and log-variance, split by the caller.
"""
from typing import Any, Dict, List

import torch
import torch.nn as nn
# `forward` layer-norms the features before the latent projection.
import torch.nn.functional as F

from ...vendor.trellis2.modules import sparse as sp
from ...vendor.trellis2.modules.utils import convert_module_to_f16, convert_module_to_f32
# Resolved by name out of globals() below. Imported for that reason alone.
from ...vendor.trellis2.models.sc_vaes.sparse_unet_vae import (  # noqa: F401
    SparseResBlock3d,
    SparseResBlockDownsample3d,
    SparseResBlockUpsample3d,
    SparseResBlockS2C3d,
    SparseResBlockC2S3d,
    SparseConvNeXtBlock3d,
)







class SparseUnetVaeEncoderCVT(nn.Module):
    """
    Sparse Swin Transformer Unet VAE model.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        down_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = sp.SparseLinear(in_channels, model_channels[0])
        self.to_latent = sp.SparseLinear(model_channels[-1], 2 * latent_channels)
        
        self.blocks = nn.ModuleList([])
        for i in range(len(num_blocks)):
            self.blocks.append(nn.ModuleList([]))
            for j in range(num_blocks[i]):
                self.blocks[-1].append(
                    globals()[block_type[i]](
                        model_channels[i],
                        **block_args[i],
                    )
                )
            if i < len(num_blocks) - 1:
                self.blocks[-1].append(
                    globals()[down_block_type[i]](
                        model_channels[i],
                        model_channels[i+1],
                        **block_args[i],
                    )
                )
                
        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: sp.SparseTensor, sample_posterior=False, return_raw=False):
        h = self.input_layer(x)
        h = h.type(self.dtype)
        for i, res in enumerate(self.blocks):
            for j, block in enumerate(res):
                h = block(h)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.to_latent(h)
        
        # Sample from the posterior distribution
        mean, logvar = h.feats.chunk(2, dim=-1)
        z = h.replace(mean)
        # This encoder takes the mean unconditionally: `sample_posterior` and `return_raw` are
        # accepted for signature compatibility with the framework's encoder but not acted on.
        return z
