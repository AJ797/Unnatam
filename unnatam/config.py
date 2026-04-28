from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

LayerKind = Literal["ssm", "attn"]


@dataclass
class UnnatamConfig:
    # Backbone
    d_model: int = 2048
    n_layers: int = 24
    vocab_size: int = 32000
    max_seq_len: int = 4096

    # Attention (MQA)
    n_heads: int = 16
    n_kv_heads: int = 1
    head_dim: int = 128
    rope_base: float = 10000.0

    # SSM (Mamba-1 style)
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | Literal["auto"] = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4

    # MLP (SwiGLU). d_ff is computed if None.
    d_ff: int | None = None
    ff_multiple_of: int = 64

    # Hybrid pattern. ssm_attn_ratio=5 means 5 SSM layers per 1 Attn layer.
    # The pattern repeats: [S, S, S, S, S, A, ...] truncated/padded to n_layers.
    ssm_attn_ratio: int = 5

    # Hormone routing (injected after each Attention block's residual)
    n_hormones: int = 7
    hormone_vector_path: str | None = None  # path to .npy of shape (n_hormones, d_model)
    hormone_router_init_gate: float = 0.0   # init value of the output gate (0 → no-op at init)

    # Norm
    norm_eps: float = 1e-5
    norm_kind: Literal["rmsnorm"] = "rmsnorm"

    # Tying
    tie_word_embeddings: bool = True

    # Initialization
    init_std: float = 0.02

    # Computed / cached fields
    layer_kinds: list[LayerKind] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.d_model % self.head_dim != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by head_dim ({self.head_dim})")
        if self.d_model // self.head_dim != self.n_heads:
            raise ValueError(
                f"n_heads ({self.n_heads}) must equal d_model/head_dim ({self.d_model // self.head_dim})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")

        if self.dt_rank == "auto":
            self.dt_rank = max(1, self.d_model // 16)

        if self.d_ff is None:
            target = int(8 / 3 * self.d_model)
            self.d_ff = ((target + self.ff_multiple_of - 1) // self.ff_multiple_of) * self.ff_multiple_of

        self.layer_kinds = self._build_layer_pattern()

    def _build_layer_pattern(self) -> list[LayerKind]:
        # Repeats [S]*ratio + [A], truncated to n_layers.
        unit: list[LayerKind] = ["ssm"] * self.ssm_attn_ratio + ["attn"]
        kinds: list[LayerKind] = []
        while len(kinds) < self.n_layers:
            kinds.extend(unit)
        return kinds[: self.n_layers]

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def n_attn_layers(self) -> int:
        return sum(1 for k in self.layer_kinds if k == "attn")


def unnatam_small() -> UnnatamConfig:
    """~1.5B parameter Unnatam (target for 4050/6GB training)."""
    return UnnatamConfig(
        d_model=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=1,
        head_dim=128,
    )


def unnatam_medium() -> UnnatamConfig:
    """~3B parameter Unnatam (training ceiling on current hardware)."""
    return UnnatamConfig(
        d_model=2560,
        n_layers=30,
        n_heads=20,
        n_kv_heads=1,
        head_dim=128,
    )
