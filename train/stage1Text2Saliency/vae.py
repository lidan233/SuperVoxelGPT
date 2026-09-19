"""The saliency autoencoder: the codebook the generator writes into, and its inverse.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from .layers import (
    ChannelLayerNorm32,
    FSQ,
    GroupNorm32,
    convert_module_to_f16,
    convert_module_to_f32,
    pixel_shuffle_3d,
    zero_module,
)

from typing import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt


# Import FSQ from existing module


def norm_layer(norm_type: str, *args, **kwargs) -> nn.Module:
    if norm_type == "group":
        return GroupNorm32(32, *args, **kwargs)
    elif norm_type == "layer":
        return ChannelLayerNorm32(*args, **kwargs)
    else:
        raise ValueError(f"Invalid norm type {norm_type}")


# ============================================
# Basic Building Blocks
# ============================================

class ResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        norm_type: Literal["group", "layer"] = "layer",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.norm1 = norm_layer(norm_type, channels)
        self.norm2 = norm_layer(norm_type, self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1))
        self.skip_connection = nn.Conv3d(channels, self.out_channels, 1) if channels != self.out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h


class DownsampleBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: Literal["conv", "avgpool"] = "conv"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels, 2, stride=2)
        elif mode == "avgpool":
            assert in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            return self.conv(x)
        else:
            return F.avg_pool3d(x, 2)


class UpsampleBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: Literal["conv", "nearest"] = "conv"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels * 8, 3, padding=1)
        elif mode == "nearest":
            assert in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            x = self.conv(x)
            return pixel_shuffle_3d(x, 2)
        else:
            return F.interpolate(x, scale_factor=2, mode="nearest")


# ============================================
# Fourier Positional Embedding for 3D
# ============================================

class FourierPositionalEmbedding3D(nn.Module):
    """Fourier positional embedding for 3D coordinates."""
    def __init__(self, embed_dim: int, num_freq_bands: int = 64, max_freq: float = 32.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_freq_bands = num_freq_bands
        freqs = torch.linspace(1.0, max_freq, num_freq_bands)
        self.register_buffer("freqs", freqs)
        # 3D coords -> sin/cos for each freq band -> 3 * 2 * num_freq_bands
        fourier_dim = 3 * 2 * num_freq_bands
        self.proj = nn.Linear(fourier_dim, embed_dim)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [..., 3] normalized coordinates in [-0.5, 0.5]
        Returns:
            [..., embed_dim] positional embeddings
        """
        shape = coords.shape[:-1]
        coords = coords.reshape(-1, 3)
        
        # [N, 3, num_freq_bands]
        scaled = coords.unsqueeze(-1) * self.freqs.view(1, 1, -1) * 2 * math.pi
        # [N, 3 * 2 * num_freq_bands]
        fourier = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).reshape(coords.shape[0], -1)
        
        out = self.proj(fourier.to(self.proj.weight.dtype))
        return out.reshape(*shape, self.embed_dim)


# ============================================
# Self-Attention for Dense 3D Grid
# ============================================

class DenseGridSelfAttention(nn.Module):
    """
    Self-attention for dense 3D grid features.
    Operates on flattened spatial dimensions.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_freq_bands: int = 64,
        max_freq: float = 32.0,
        dropout: float = 0.0,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_checkpoint = use_checkpoint
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.pos_embed = FourierPositionalEmbedding3D(embed_dim, num_freq_bands, max_freq)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def _forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W] dense features
            coords: [B, D, H, W, 3] normalized coordinates
        Returns:
            [B, C, D, H, W] attended features
        """
        B, C, D, H, W = x.shape
        N = D * H * W
        
        # Flatten spatial dims: [B, N, C]
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, N, C)
        coords_flat = coords.reshape(B, N, 3)
        
        # Add positional embedding
        pos = self.pos_embed(coords_flat)
        x_with_pos = x_flat + pos
        
        # QKV projections
        q = self.q_proj(x_with_pos).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_with_pos).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_flat).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        # Residual + norm
        out = self.norm(x_flat + out)
        
        # Reshape back to 3D
        out = out.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        return out
    
    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return ckpt.checkpoint(self._forward, x, coords, use_reentrant=False)
        return self._forward(x, coords)


# ============================================
# Saliency VQ-VAE Encoder
# ============================================

class SaliencyVQVAEEncoder(nn.Module):
    """
    VQ-VAE Encoder for Saliency Volume.
    
    Architecture:
    1. Conv encoder: 64³ → 8³ (3 downsampling layers)
    2. Pre-quantize self-attention layers
    3. FSQ quantization (8³ = 512 tokens)
    4. Post-quantize self-attention layers
    
    Args:
        in_channels: Input channels (1 for saliency volume)
        latent_channels: Latent feature channels
        num_res_blocks: Number of residual blocks per resolution
        channels: Channel progression [32, 128, 512]
        num_res_blocks_middle: Number of middle residual blocks
        vq_levels: FSQ quantization levels per dimension (e.g., [9, 5, 5, 5, 5])
        num_pre_attn_layers: Number of pre-quantize attention layers
        num_post_attn_layers: Number of post-quantize attention layers
        num_heads: Number of attention heads
        num_freq_bands: Fourier positional embedding frequency bands
        use_fp16: Whether to use FP16
        use_checkpoint: Whether to use gradient checkpointing
    """
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 8,
        num_res_blocks: int = 2,
        channels: List[int] = [32, 128, 512],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        vq_levels: List[int] = [9, 5, 5, 5, 5],
        num_pre_attn_layers: int = 2,
        num_post_attn_layers: int = 2,
        num_heads: int = 8,
        num_freq_bands: int = 64,
        max_freq: float = 32.0,
        use_fp16: bool = False,
        use_checkpoint: bool = True,
        num_quantizers: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        
        # ============================================
        # 1. Convolutional Encoder (64³ → 8³)
        # ============================================
        self.input_layer = nn.Conv3d(in_channels, channels[0], 3, padding=1)
        
        self.encoder_blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.encoder_blocks.extend([
                ResBlock3d(ch, ch, norm_type) for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.encoder_blocks.append(DownsampleBlock3d(ch, channels[i + 1]))
        
        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[-1], channels[-1], norm_type) for _ in range(num_res_blocks_middle)
        ])
        
        self.to_latent = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], latent_channels, 3, padding=1)
        )
        
        # ============================================
        # 2. Pre-quantize Self-Attention
        # ============================================
        self.pre_quantize_attns = nn.ModuleList([
            DenseGridSelfAttention(
                latent_channels, num_heads, num_freq_bands, max_freq, 
                dropout=0.0, use_checkpoint=use_checkpoint
            ) for _ in range(num_pre_attn_layers)
        ])
        
        # ============================================
        # 3. FSQ Quantization
        # ============================================
        # levels=[9,5,5,5,5] -> codebook_size = 9*5*5*5*5 = 5625
        self.num_quantizers = num_quantizers
        self.quantizer = FSQ(
            levels=vq_levels,
            dim=latent_channels,
            num_codebooks=num_quantizers,
            keep_num_codebooks_dim=(num_quantizers > 1),
            channel_first=True,  # Input is [B, C, D, H, W]
            return_indices=True,
        )
        self.codebook_size = self.quantizer.codebook_size
        
        # ============================================
        # 4. Post-quantize Self-Attention
        # ============================================
        self.post_quantize_attns = nn.ModuleList([
            DenseGridSelfAttention(
                latent_channels, num_heads, num_freq_bands, max_freq,
                dropout=0.0, use_checkpoint=use_checkpoint
            ) for _ in range(num_post_attn_layers)
        ])
        
        if use_fp16:
            self.convert_to_fp16()
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def convert_to_fp16(self):
        self.use_fp16 = True
        self.dtype = torch.float16
        self.encoder_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
    
    def convert_to_fp32(self):
        self.use_fp16 = False
        self.dtype = torch.float32
        self.encoder_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
    
    def _make_coords(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Create normalized coordinates for a 3D grid."""
        B, C, D, H, W = shape
        # Create coordinate grids normalized to [-0.5, 0.5]
        d = torch.linspace(-0.5, 0.5, D, device=device)
        h = torch.linspace(-0.5, 0.5, H, device=device)
        w = torch.linspace(-0.5, 0.5, W, device=device)
        coords = torch.stack(torch.meshgrid(d, h, w, indexing='ij'), dim=-1)  # [D, H, W, 3]
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B, D, H, W, 3]
        return coords
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 1, 64, 64, 64] saliency volume
        Returns:
            dict with:
                - z_q: [B, latent_channels, 8, 8, 8] quantized latent
                - indices: [B, 8, 8, 8] codebook indices
                - z_e: [B, latent_channels, 8, 8, 8] pre-quantization latent
        """
        # 1. Encode
        h = self.input_layer(x)
        h = h.type(self.dtype)
        
        for block in self.encoder_blocks:
            h = block(h)
        h = self.middle_block(h)
        
        h = h.type(x.dtype)
        z_e = self.to_latent(h)  # [B, latent_channels, 8, 8, 8]
        
        # 2. Pre-quantize attention
        coords = self._make_coords(z_e.shape, z_e.device)
        for attn in self.pre_quantize_attns:
            z_e = attn(z_e, coords)
        
        # 3. Quantize using FSQ
        # FSQ with channel_first=True expects [B, C, D, H, W]
        z_q, indices = self.quantizer(z_e)
        
        # 4. Post-quantize attention
        for attn in self.post_quantize_attns:
            z_q = attn(z_q, coords)
        
        return {
            "z_q": z_q,
            "indices": indices,
            "z_e": z_e,
        }


# ============================================
# Saliency VQ-VAE Decoder (Two-Head)
# ============================================

class SaliencyVQVAEDecoder(nn.Module):
    """
    VQ-VAE Decoder for Saliency Volume with two output heads.
    
    Args:
        out_channels: Output channels per head (1 for logits, 1 for saliency)
        latent_channels: Latent feature channels
        num_res_blocks: Number of residual blocks per resolution
        channels: Channel progression [512, 128, 64]
        num_res_blocks_middle: Number of middle residual blocks
        use_fp16: Whether to use FP16
    """
    def __init__(
        self,
        out_channels: int = 1,
        latent_channels: int = 8,
        num_res_blocks: int = 4,
        channels: List[int] = [512, 128, 64],
        num_res_blocks_middle: int = 4,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        
        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)
        
        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], channels[0], norm_type) for _ in range(num_res_blocks_middle)
        ])
        
        self.decoder_blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.decoder_blocks.extend([
                ResBlock3d(ch, ch, norm_type) for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.decoder_blocks.append(UpsampleBlock3d(ch, channels[i + 1]))
        
        # Two output heads (split channels)
        split_channels = channels[-1] // 2
        
        self.logits_out_layer = nn.Sequential(
            norm_layer(norm_type, split_channels),
            nn.SiLU(),
            nn.Conv3d(split_channels, out_channels, 3, padding=1)
        )
        
        self.saliency_out_layer = nn.Sequential(
            norm_layer(norm_type, split_channels),
            nn.SiLU(),
            nn.Conv3d(split_channels, out_channels, 3, padding=1)
        )
        
        if use_fp16:
            self.convert_to_fp16()
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def convert_to_fp16(self):
        self.use_fp16 = True
        self.dtype = torch.float16
        self.decoder_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
    
    def convert_to_fp32(self):
        self.use_fp16 = False
        self.dtype = torch.float32
        self.decoder_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, latent_channels, 8, 8, 8] quantized latent
        Returns:
            logits: [B, 1, 64, 64, 64] foreground/background logits
            saliency: [B, 1, 64, 64, 64] saliency values
        """
        h = self.input_layer(z)
        h = h.type(self.dtype)
        
        h = self.middle_block(h)
        for block in self.decoder_blocks:
            h = block(h)
        
        h = h.type(z.dtype)
        
        # Split features for two heads
        split_idx = h.shape[1] // 2
        h_logits = h[:, :split_idx, ...]
        h_saliency = h[:, split_idx:, ...]
        
        logits = self.logits_out_layer(h_logits)
        saliency = self.saliency_out_layer(h_saliency)
        
        return logits, saliency


# ============================================
# Mask-Only VQ-VAE Decoder (Single Head)
# ============================================

class MaskVQVAEDecoder(nn.Module):
    """
    VQ-VAE Decoder for Mask/Foreground prediction only.
    Single head output for binary classification.
    """
    def __init__(
        self,
        out_channels: int = 1,
        latent_channels: int = 8,
        num_res_blocks: int = 4,
        channels: List[int] = [512, 128, 64],
        num_res_blocks_middle: int = 4,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        
        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)
        
        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], channels[0], norm_type) for _ in range(num_res_blocks_middle)
        ])
        
        self.decoder_blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.decoder_blocks.extend([
                ResBlock3d(ch, ch, norm_type) for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.decoder_blocks.append(UpsampleBlock3d(ch, channels[i + 1]))
        
        # Single output head for mask/logits
        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1)
        )
        
        if use_fp16:
            self.convert_to_fp16()
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def convert_to_fp16(self):
        self.use_fp16 = True
        self.dtype = torch.float16
        self.decoder_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
    
    def convert_to_fp32(self):
        self.use_fp16 = False
        self.dtype = torch.float32
        self.decoder_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, latent_channels, 8, 8, 8] quantized latent
        Returns:
            logits: [B, 1, 64, 64, 64] foreground/background logits
        """
        h = self.input_layer(z)
        h = h.type(self.dtype)
        
        h = self.middle_block(h)
        for block in self.decoder_blocks:
            h = block(h)
        
        h = h.type(z.dtype)
        logits = self.out_layer(h)
        
        return logits


# ============================================
# Saliency-Only VQ-VAE Decoder (Single Head)
# ============================================

class SaliencyOnlyVQVAEDecoder(nn.Module):
    """
    VQ-VAE Decoder for Saliency regression only.
    Single head output for continuous saliency values.
    """
    def __init__(
        self,
        out_channels: int = 1,
        latent_channels: int = 8,
        num_res_blocks: int = 4,
        channels: List[int] = [512, 128, 64],
        num_res_blocks_middle: int = 4,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        
        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)
        
        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], channels[0], norm_type) for _ in range(num_res_blocks_middle)
        ])
        
        self.decoder_blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.decoder_blocks.extend([
                ResBlock3d(ch, ch, norm_type) for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.decoder_blocks.append(UpsampleBlock3d(ch, channels[i + 1]))
        
        # Single output head for saliency
        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1)
        )
        
        if use_fp16:
            self.convert_to_fp16()
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def convert_to_fp16(self):
        self.use_fp16 = True
        self.dtype = torch.float16
        self.decoder_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
    
    def convert_to_fp32(self):
        self.use_fp16 = False
        self.dtype = torch.float32
        self.decoder_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, latent_channels, 8, 8, 8] quantized latent
        Returns:
            saliency: [B, 1, 64, 64, 64] saliency values
        """
        h = self.input_layer(z)
        h = h.type(self.dtype)
        
        h = self.middle_block(h)
        for block in self.decoder_blocks:
            h = block(h)
        
        h = h.type(z.dtype)
        saliency = self.out_layer(h)
        
        return saliency


class SaliencyDecoderFullMaskHead(SaliencyVQVAEDecoder):
    """Occupancy head reading the full trunk, saliency head reading half of it.

    The saliency head keeps its original half-width input so it inherits pretrained weights
    unchanged; the occupancy head is widened to the trunk's full width, which doubles what it can
    see. The asymmetry is deliberate — occupancy is decided everywhere and benefits from all the
    features, while saliency is a refinement whose pretrained solution is worth preserving.
    """
    def __init__(self, out_channels=1, latent_channels=8, num_res_blocks=4,
                 channels=None, num_res_blocks_middle=4, norm_type="layer", use_fp16=False):
        if channels is None:
            channels = [512, 256, 128, 64]
        super().__init__(out_channels=out_channels, latent_channels=latent_channels,
                         num_res_blocks=num_res_blocks, channels=channels,
                         num_res_blocks_middle=num_res_blocks_middle,
                         norm_type=norm_type, use_fp16=use_fp16)
        full_ch = channels[-1]
        # Replace the occupancy head built by the base class with one taking the full trunk width.
        self.logits_out_layer = nn.Sequential(
            norm_layer(norm_type, full_ch),
            nn.SiLU(),
            nn.Conv3d(full_ch, out_channels, 3, padding=1),
        )

    def forward(self, z):
        h = self.input_layer(z)
        h = h.type(self.dtype)
        h = self.middle_block(h)
        for block in self.decoder_blocks:
            h = block(h)
        h = h.type(z.dtype)
        split_idx = h.shape[1] // 2
        logits = self.logits_out_layer(h)                          # full trunk width
        saliency = self.saliency_out_layer(h[:, split_idx:, ...])  # second half only
        return logits, saliency
