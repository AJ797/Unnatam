from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fast path: mamba-ssm's fused CUDA kernel. We auto-fall-back to the reference
# scan when the kernel isn't installed (Windows host, CPU dev, smoke tests).
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _mamba_selective_scan_fn

    _HAS_MAMBA_SSM = True
except ImportError:  # pragma: no cover — env-dependent
    _mamba_selective_scan_fn = None
    _HAS_MAMBA_SSM = False


def has_fast_scan() -> bool:
    return _HAS_MAMBA_SSM


class IntraCellularAttention(nn.Module):
    """Attention over SSM state slots applied at every step of the selective scan.

    At each scan step t, the SSM state h_t ∈ ℝ^(B, d_inner, d_state) is
    reinterpreted as d_state "slots" each of dimension d_inner.  In vanilla
    Mamba these slots evolve completely independently — they never cross-talk.
    This module lets them attend to each other *before* the linear readout
    y_t = (h_t ⊙ C_t).sum(-1), making the readout content-dependent at the
    slot level.

    Cost analysis
    -------------
    d_attn defaults to d_state (e.g. 16). The Q and K projections are
    (d_inner → d_attn), applied to d_state slots: the dominant cost per step
    is 2 × d_state × d_inner × d_attn ops.  For d_inner=2048, d_state=16,
    d_attn=16 that is ~1M ops/step, vs the SSM state update at ~130K ops/step.
    This overhead is acceptable for research demonstration runs on the ref
    scan path; fast-kernel support is deferred to future work.

    Initialization
    --------------
    The output gate is zero-initialized (matching the hormone gate convention),
    so the module starts as an exact no-op and the model learns when and how
    much to activate it.
    """

    def __init__(self, d_inner: int, d_state: int, d_attn: int | None = None) -> None:
        super().__init__()
        self.d_attn = d_attn if d_attn is not None else d_state
        self.scale = self.d_attn ** -0.5
        # Tiny Q, K projections — shared across all d_state slots.
        self.W_q = nn.Linear(d_inner, self.d_attn, bias=False)
        self.W_k = nn.Linear(d_inner, self.d_attn, bias=False)
        # Values are the slots themselves — no additional projection needed.
        # Zero-init gate: module is a no-op at initialisation.
        self.gate = nn.Parameter(torch.zeros(1))
        self.gate._no_weight_decay = True  # type: ignore[attr-defined]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, d_inner, d_state)
        slots = h.permute(0, 2, 1)                                  # (B, d_state, d_inner)
        q = self.W_q(slots)                                          # (B, d_state, d_attn)
        k = self.W_k(slots)                                          # (B, d_state, d_attn)
        attn = F.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)  # (B, d_state, d_state)
        out = (attn @ slots).permute(0, 2, 1)                       # (B, d_inner, d_state)
        return h + self.gate * out


def selective_scan_fast(
    u: torch.Tensor,        # (B, L, d_inner)
    delta: torch.Tensor,    # (B, L, d_inner)
    A: torch.Tensor,        # (d_inner, d_state)
    B: torch.Tensor,        # (B, L, d_state)
    C: torch.Tensor,        # (B, L, d_state)
    D: torch.Tensor,        # (d_inner,)
) -> torch.Tensor:
    """mamba-ssm-backed selective scan. Requires CUDA tensors and the kernel install."""
    if _mamba_selective_scan_fn is None:
        raise RuntimeError("mamba-ssm not installed; selective_scan_fast is unavailable")

    # The CUDA kernel requires u and delta in the same dtype, and B/C matching u.
    # Under bf16 autocast some ops can leak fp32, so explicitly align.
    target_dtype = u.dtype
    delta = delta.to(target_dtype)
    B = B.to(target_dtype)
    C = C.to(target_dtype)
    # A and D stay in their parameter dtype (typically fp32) — the kernel expects this.

    # mamba-ssm expects (B, d_inner, L) for u/delta and (B, d_state, L) for B/C.
    u_t = u.transpose(1, 2).contiguous()
    delta_t = delta.transpose(1, 2).contiguous()
    B_t = B.transpose(1, 2).contiguous()
    C_t = C.transpose(1, 2).contiguous()
    y_t = _mamba_selective_scan_fn(
        u_t, delta_t, A, B_t, C_t, D, z=None, delta_bias=None, delta_softplus=False
    )
    return y_t.transpose(1, 2).contiguous()


def selective_scan_ref(
    u: torch.Tensor,                            # (B, L, d_inner)
    delta: torch.Tensor,                        # (B, L, d_inner)
    A: torch.Tensor,                            # (d_inner, d_state)
    B: torch.Tensor,                            # (B, L, d_state)
    C: torch.Tensor,                            # (B, L, d_state)
    D: torch.Tensor,                            # (d_inner,)
    intra_attn: "IntraCellularAttention | None" = None,
    ia_stride: int = 32,
) -> torch.Tensor:
    """Reference selective scan in pure PyTorch.

    Sequential time loop — correct but slow. Used as fallback when mamba-ssm
    isn't available (Windows dev host, CPU smoke tests), as a numerical
    cross-check for the fast kernel, and as the required path when
    intracellular attention is enabled (the fast kernel cannot hook into the
    inner state update loop).

    When intra_attn is provided, the SSM state h_t is passed through
    IntraCellularAttention after every ``ia_stride`` state updates (default 32)
    rather than every step. This preserves the slot-interaction inductive bias
    while reducing the number of intra-attention calls per layer-scan from L
    (=1024 by default) to L/stride (=32), keeping the mechanism tractable under
    our compute budget. Setting ``ia_stride=1`` recovers per-step IA.
    """
    B_, L, d_inner = u.shape
    d_state = A.shape[1]

    deltaA = torch.exp(delta.unsqueeze(-1) * A)                          # (B, L, d_inner, d_state)
    deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)    # (B, L, d_inner, d_state)

    h = torch.zeros(B_, d_inner, d_state, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        h = deltaA[:, t] * h + deltaB_u[:, t]
        # Fire IA every ia_stride steps and always on the last step (so the
        # final readout sees slot interaction, and short test sequences still
        # exercise the IA path).
        if intra_attn is not None and (((t + 1) % ia_stride == 0) or t == L - 1):
            h = intra_attn(h)
        ys.append((h * C[:, t].unsqueeze(1)).sum(dim=-1))                # (B, d_inner)
    y = torch.stack(ys, dim=1)                                           # (B, L, d_inner)
    return y + u * D


class MambaBlock(nn.Module):
    """Mamba-1 selective state-space block (pure PyTorch)."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        use_intra_attn: bool = False,
        intra_attn_dim: int | None = None,
        ia_stride: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        # Project x → (dt, B, C)
        self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # Initialize dt_proj so softplus(dt_proj.bias) is uniform on [dt_min, dt_max].
        dt_init_std = dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Mark so that this bias is excluded from any later re-init.
        self.dt_proj.bias._no_reinit = True  # type: ignore[attr-defined]

        # A: parameterized in log-space, negative real. Shape (d_inner, d_state).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]

        # D: per-channel skip.
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Intracellular attention: slot attention inside the scan loop.
        # When active, forces the ref scan path (fast kernel unsupported).
        # ia_stride controls how often IA is applied within the scan; default 32
        # reduces IA calls per layer-scan from L to L/stride for tractability.
        self.intra_attn: IntraCellularAttention | None = (
            IntraCellularAttention(self.d_inner, d_state, intra_attn_dim)
            if use_intra_attn else None
        )
        self.ia_stride: int = ia_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, _ = x.shape

        xz = self.in_proj(x)                                  # (B, T, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                         # each (B, T, d_inner)

        # Causal depthwise conv over time.
        x_conv = self.conv1d(x_in.transpose(1, 2))[..., :T]   # (B, d_inner, T)
        x_act = F.silu(x_conv).transpose(1, 2)                # (B, T, d_inner)

        # Input-dependent dt, B, C.
        dbc = self.x_proj(x_act)
        dt, B_in, C_in = torch.split(dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                     # (B, T, d_inner)

        A = -torch.exp(self.A_log.float())                    # (d_inner, d_state), negative real
        use_ref = (
            self.intra_attn is not None                        # intracellular attn needs ref scan
            or getattr(self, "_force_ref_scan", False)
            or not (_HAS_MAMBA_SSM and x_act.is_cuda)
        )
        if use_ref:
            y = selective_scan_ref(
                x_act, dt, A, B_in, C_in, self.D,
                intra_attn=self.intra_attn, ia_stride=self.ia_stride,
            )
        else:
            y = selective_scan_fast(x_act, dt, A, B_in, C_in, self.D)

        y = y * F.silu(z)
        return self.out_proj(y)
