from __future__ import annotations

import torch
import torch.nn as nn

from unnatam.config import UnnatamConfig
from unnatam.model.attention import MQAttention
from unnatam.model.hormone import HormoneRouter
from unnatam.model.mlp import SwiGLU
from unnatam.model.norm import RMSNorm
from unnatam.model.ssm import MambaBlock


class SSMLayer(nn.Module):
    """Pre-norm Mamba mixer block. Mamba does both token-mixing and channel-mixing,
    so there is no separate MLP — matches the Mamba/Jamba convention."""

    def __init__(self, cfg: UnnatamConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mixer = MambaBlock(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            dt_rank=cfg.dt_rank,  # type: ignore[arg-type]  # resolved to int in __post_init__
            dt_min=cfg.dt_min,
            dt_max=cfg.dt_max,
            dt_init_floor=cfg.dt_init_floor,
            use_intra_attn=cfg.use_intra_attn,
            intra_attn_dim=cfg.intra_attn_dim,
            ia_stride=cfg.ia_stride,
        )

    def forward(self, x: torch.Tensor, hormone_vectors: torch.Tensor) -> torch.Tensor:
        del hormone_vectors  # SSM layers don't inject; signature is uniform across layer types.
        return x + self.mixer(self.norm(x))


class AttnLayer(nn.Module):
    """Pre-norm Attention + MLP, with optional hormone injection added to the
    residual stream after the MLP's residual add.

    When cfg.use_hormones is False (Base / IA variants) the HormoneRouter is not
    instantiated and the forward path is identical to a plain Attn+MLP block,
    keeping the parameter count clean for ablations.
    """

    def __init__(self, cfg: UnnatamConfig):
        super().__init__()
        assert cfg.d_ff is not None
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = MQAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            rope_base=cfg.rope_base,
        )
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff)
        self.hormone: HormoneRouter | None = (
            HormoneRouter(
                d_model=cfg.d_model,
                n_hormones=cfg.n_hormones,
                init_gate=cfg.hormone_router_init_gate,
                alpha=cfg.hormone_alpha,
            )
            if cfg.use_hormones
            else None
        )

    def forward(self, x: torch.Tensor, hormone_vectors: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        if self.hormone is not None:
            x = x + self.hormone(x, hormone_vectors)
        return x
