from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rope_cache(seq_len: int, head_dim: int, base: float, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_heads, T, head_dim). Rotate even/odd halves of head_dim.
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim/2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rot


class MQAttention(nn.Module):
    """Multi-Query Attention with RoPE. n_kv_heads can be 1 (MQA) or group size (GQA)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rope_base = rope_base

        self.w_q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.w_k = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.w_v = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * head_dim, d_model, bias=False)

        self._rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._rope_cache_len: int = 0

    def _rope(self, seq_len: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._rope_cache is None
            or self._rope_cache_len < seq_len
            or self._rope_cache[0].device != device
            or self._rope_cache[0].dtype != dtype
        ):
            cos, sin = build_rope_cache(seq_len, self.head_dim, self.rope_base, device, dtype)
            self._rope_cache = (cos, sin)
            self._rope_cache_len = seq_len
        cos, sin = self._rope_cache
        return cos[:seq_len], sin[:seq_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.w_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._rope(T, x.device, x.dtype)
        # RoPE expects (T, head_dim/2). cos/sin from build_rope_cache are (T, head_dim/2).
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Expand KV heads to match Q heads (MQA: 1 → n_heads; GQA: n_kv → n_heads).
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # SDPA picks Flash/efficient kernel automatically when supported.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.w_o(out)
