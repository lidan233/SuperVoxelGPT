"""Generate the shape tokens from a prompt and a set of supervoxel centers.

Purpose
    The saliency stage has already decided *where* the budget goes, and CVT has turned that into
    concrete coordinates. What remains is what sits at each center. This model answers that: one
    codebook index per center, which the supervoxel autoencoder decodes into the final surface.

Input
    Encoded text features (77 x 768 from CLIP) and the supervoxel centers in [-0.5, 0.5]. Token
    count follows the coordinates, so sequences run from a few hundred to several thousand.

Output
    Training: cross-entropy over the token field plus its accuracy. Inference: the whole token
    field, reached by iterating a parallel update to its fixed point.
"""
import os
import math
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

VOCAB = 10125          # supervoxel codebook size
BOS_ID = 10125         # sequence start, appended past the codebook
PAD_ID = 10126


class FourierCoords(nn.Module):
    """Positional encoding for a supervoxel center: sin/cos over octave frequencies, plus the raw
    coordinate so the exact position stays recoverable."""

    def __init__(self, hidden: int, num_bands: int = 16):
        super().__init__()
        # non-persistent: the frequencies are the constant 2^k*pi, and the released checkpoints do
        # not contain them. Keeping them out of state_dict prevents a harmless constant-buffer
        # mismatch; callers still use strict=False for legacy checkpoints and warn on every
        # remaining missing or unexpected key.
        self.register_buffer("freqs", (2.0 ** torch.arange(num_bands)) * math.pi, persistent=False)
        self.proj = nn.Linear(num_bands * 6 + 3, hidden)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:   # [B,N,3] in [-0.5,0.5]
        x = coords.unsqueeze(-1) * self.freqs                  # [B,N,3,K]
        feat = torch.cat([torch.sin(x), torch.cos(x)], dim=-1).flatten(-2)   # [B,N,6K]
        return self.proj(torch.cat([feat, coords], dim=-1))


def _resolve_attn_impl() -> str:
    """flash_attention_2, unless `T2S_ATTN_IMPL` names something else.

    Not a preference. The Jacobi decode runs one full-sequence forward per iteration, so the
    `sdpa` fallback materializes the whole N x N score matrix every iteration and is markedly
    slower, while producing the same tokens. Missing flash-attn is raised rather than worked
    around, so the slower path is never taken without asking for it.
    """
    override = os.environ.get("T2S_ATTN_IMPL")
    if override:
        return override

    import torch
    usable = False
    try:
        from transformers.utils import is_flash_attn_2_available
        usable = is_flash_attn_2_available()
    except ImportError:
        # Older transformers do not export the helper; ask the package itself instead.
        try:
            import flash_attn                                   # noqa: F401
            usable = True
        except ImportError:
            usable = False

    if usable and torch.cuda.is_available():
        return "flash_attention_2"

    raise RuntimeError(
        "flash-attn is required and is not usable here "
        f"(flash-attn importable: {usable}, cuda: {torch.cuda.is_available()}).\n"
        "  Install it:  TORCH_CUDA_ARCH_LIST=\"8.0;8.6;8.9+PTX\" MAX_JOBS=8 "
        "pip install flash-attn==2.6.3 --no-build-isolation\n"
        "  Or accept roughly 2.6x slower generation:  export T2S_ATTN_IMPL=sdpa"
    )


class Text3DQwenAR(nn.Module):
    """Text + coordinates -> one token per coordinate.

    Defaults reproduce the released model: Qwen2 12 layers / 896 hidden / GQA 14 heads over 2 KV
    heads / FFN 4864 / RoPE.

    The layer count is part of the checkpoint contract, not a tuning knob: `stage2TextSaliency2Shape.bin`
    is 791 MB, which is exactly 12 layers at these widths. Training with a different count
    produces weights the released inference path cannot load.
    """

    def __init__(
        self,
        text_hidden_dim: int = 768,
        num_layers: int = 12,
        hidden: int = 896,
        num_heads: int = 14,
        num_kv_heads: int = 2,
        ffn: int = 4864,
        max_pos: int = 4096,
        attn_impl: Optional[str] = None,      # None -> _resolve_attn_impl()
        verbose: bool = True,
    ):
        super().__init__()
        # HF Trainer inspects this attribute; leaving it unset makes save_pretrained-style paths fail
        self._keys_to_ignore_on_save = None
        cfg = Qwen2Config(
            vocab_size=VOCAB + 2, hidden_size=hidden, num_hidden_layers=num_layers,
            num_attention_heads=num_heads, num_key_value_heads=num_kv_heads,
            intermediate_size=ffn, max_position_embeddings=max_pos,
            tie_word_embeddings=False,
        )
        cfg._attn_implementation = attn_impl or _resolve_attn_impl()
        self.lm = Qwen2ForCausalLM(cfg)
        self.text_proj = nn.Linear(text_hidden_dim, hidden)
        self.coord_enc = FourierCoords(hidden)
        self.tokenizer = None          # the training loop probes for one; this model never needs it
        if verbose:
            n = sum(p.numel() for p in self.parameters())
            print(f"Text3DQwenAR: layers={num_layers} hidden={hidden} params={n/1e6:.1f}M vocab={VOCAB}+2")

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def _build_embeds(self, text_features, token_in, coords):
        """Input at position i = embedding of the previous token + encoding of position i."""
        tok = self.lm.get_input_embeddings()(token_in) + self.coord_enc(coords)   # [B,N,H]
        prefix = self.text_proj(text_features.float())                            # [B,T,H]
        return torch.cat([prefix, tok], dim=1)

    def forward(
        self,
        text_features: torch.Tensor = None,          # [B,T,768]
        text_ids: torch.Tensor = None,               # accepted for signature compatibility, unused
        text_attention_mask: torch.Tensor = None,    # [B,T]
        token_3d_ids: torch.Tensor = None,           # [B,N] targets
        token_3d_coords: torch.Tensor = None,        # [B,N,3]
        token_3d_attention_mask: Optional[torch.Tensor] = None,
        input_3d_ids: Optional[torch.Tensor] = None,  # feed a draft instead of ground truth
    ) -> Dict[str, torch.Tensor]:
        B, N = token_3d_ids.shape
        dev = token_3d_ids.device
        T = text_features.shape[1]

        # teacher forcing: shift right, BOS at the front
        src = input_3d_ids if input_3d_ids is not None else token_3d_ids
        bos = torch.full((B, 1), BOS_ID, device=dev, dtype=torch.long)
        tin = torch.cat([bos, src[:, :-1].clamp(min=0)], dim=1)

        # Input noise teaches the model to recover from its own mistakes, which is what parallel
        # decoding needs: during Jacobi iterations the context is full of not-yet-correct tokens.
        noise_p = float(os.environ.get("MLLM_INPUT_NOISE", "0")) if self.training else 0.0
        if noise_p > 0:
            corrupt = torch.rand(tin.shape, device=dev) < noise_p
            corrupt[:, 0] = False                                   # never disturb BOS
            tin = torch.where(corrupt, torch.randint(0, VOCAB, tin.shape, device=dev), tin)

        am3 = token_3d_attention_mask if token_3d_attention_mask is not None \
            else torch.ones(B, N, dtype=torch.long, device=dev)
        attn = torch.cat([text_attention_mask.long(), am3.long()], dim=1)
        logits = self.lm(inputs_embeds=self._build_embeds(text_features, tin, token_3d_coords),
                         attention_mask=attn).logits[:, T:, :VOCAB]     # drop the text prefix

        labels = token_3d_ids.clone()
        labels[am3 == 0] = -100
        loss = F.cross_entropy(logits.reshape(-1, VOCAB).float(), labels.reshape(-1), ignore_index=-100)
        with torch.no_grad():
            valid = am3 == 1
            acc = ((logits.argmax(-1) == token_3d_ids) & valid).sum().float() / valid.sum().clamp(min=1).float()
        return {"loss": loss, "accuracy": acc, "logits": logits}

    @torch.no_grad()
    def jacobi_generate(
        self,
        text_features=None,
        text_attention_mask=None,
        coords=None,
        token_3d_attention_mask=None,
        max_iters: int = 128,
        init_tokens=None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Decode every position in parallel until the Jacobi state stops changing.

        This is the stage-2 decoding algorithm, not an approximation to a separate sequential
        decoder. It returns the fixed point when consecutive states agree, or the final iterate
        after ``max_iters`` when they do not.
        """
        B, N = coords.shape[0], coords.shape[1]
        dev = coords.device
        cur = torch.zeros(B, N, dtype=torch.long, device=dev) if init_tokens is None \
            else init_tokens.clone().long()
        am3 = token_3d_attention_mask if token_3d_attention_mask is not None \
            else torch.ones(B, N, dtype=torch.long, device=dev)
        attn = torch.cat([text_attention_mask.long(), am3.long()], dim=1)
        bos = torch.full((B, 1), BOS_ID, device=dev, dtype=torch.long)
        T = text_features.shape[1]

        used = 0
        for it in range(max_iters):
            tin = torch.cat([bos, cur[:, :-1]], dim=1)
            logits = self.lm(inputs_embeds=self._build_embeds(text_features, tin, coords),
                             attention_mask=attn).logits[:, T:, :VOCAB]
            new = torch.where(am3 == 1, logits.argmax(-1), cur)
            used = it + 1
            if torch.equal(new, cur):
                break
            cur = new
        if verbose:
            print(f"[Jacobi] {used} iters (N={N})")
        return cur
