"""Smoke tests: build a tiny Unnatam, run a forward pass, check shapes and grads."""

from __future__ import annotations

import torch

from unnatam.config import UnnatamConfig
from unnatam.model import Unnatam
from unnatam.model.block import AttnLayer, SSMLayer
from unnatam.model.ssm import IntraCellularAttention, selective_scan_ref


def _tiny_cfg(**overrides) -> UnnatamConfig:
    base = dict(
        d_model=128,
        n_layers=6,
        vocab_size=512,
        max_seq_len=64,
        n_heads=4,
        n_kv_heads=1,
        head_dim=32,
        d_state=8,
        d_conv=4,
        expand=2,
        ssm_attn_ratio=2,
        n_hormones=7,
    )
    base.update(overrides)
    return UnnatamConfig(**base)


def test_layer_pattern_matches_ratio() -> None:
    cfg = _tiny_cfg(n_layers=6, ssm_attn_ratio=2)
    # Pattern: [S, S, A, S, S, A]
    assert cfg.layer_kinds == ["ssm", "ssm", "attn", "ssm", "ssm", "attn"]
    assert cfg.n_attn_layers == 2


def test_forward_shape() -> None:
    cfg = _tiny_cfg()
    model = Unnatam(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    with torch.no_grad():
        logits = model(ids)
    assert logits.shape == (2, 32, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_backward_runs() -> None:
    cfg = _tiny_cfg()
    model = Unnatam(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(ids)
    loss = logits.float().mean()
    loss.backward()
    # Every trainable parameter should receive a gradient (at least once exercised).
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no grad for: {missing[:5]}"


def test_hormone_gate_zero_means_no_op_at_init() -> None:
    """With gate=0 at init, the AttnLayer's hormone branch should add exactly zero."""
    cfg = _tiny_cfg(hormone_router_init_gate=0.0)
    model = Unnatam(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))

    with torch.no_grad():
        # Capture residual before/after hormone branch in the first AttnLayer.
        attn_layer = next(layer for layer in model.layers if isinstance(layer, AttnLayer))
        x = model.embed(ids)
        for layer in model.layers:
            if layer is attn_layer:
                pre = x
                pre = pre + attn_layer.attn(attn_layer.attn_norm(pre))
                pre = pre + attn_layer.mlp(attn_layer.mlp_norm(pre))
                shift = attn_layer.hormone(pre, model.hormone_vectors)
                assert torch.allclose(shift, torch.zeros_like(shift))
                x = pre + shift
            else:
                x = layer(x, model.hormone_vectors)


def test_param_count_sanity() -> None:
    cfg = _tiny_cfg()
    model = Unnatam(cfg)
    n = model.num_parameters()
    # Tiny model should be well under 10M params; well over 100k.
    assert 100_000 < n < 10_000_000, f"unexpected param count: {n}"


def test_layer_dispatch() -> None:
    cfg = _tiny_cfg()
    model = Unnatam(cfg)
    kinds = [type(layer).__name__ for layer in model.layers]
    expected = ["SSMLayer" if k == "ssm" else "AttnLayer" for k in cfg.layer_kinds]
    assert kinds == expected


# ---------------------------------------------------------------------------
# Intracellular attention tests
# ---------------------------------------------------------------------------

def test_intra_attn_gate_zero_is_noop() -> None:
    """With the zero-init gate, IntraCellularAttention must be an exact no-op."""
    torch.manual_seed(0)
    d_inner, d_state = 64, 8
    module = IntraCellularAttention(d_inner=d_inner, d_state=d_state)
    h = torch.randn(2, d_inner, d_state)
    out = module(h)
    assert torch.allclose(out, h), "gate=0 should make module a no-op"


def test_intra_attn_activates_after_gate_nonzero() -> None:
    """Output should differ from input once gate is non-zero."""
    torch.manual_seed(0)
    d_inner, d_state = 64, 8
    module = IntraCellularAttention(d_inner=d_inner, d_state=d_state)
    with torch.no_grad():
        module.gate.fill_(1.0)
    h = torch.randn(2, d_inner, d_state)
    out = module(h)
    assert not torch.allclose(out, h), "non-zero gate should change the output"


def test_intra_attn_output_shape() -> None:
    """Output shape must match input shape."""
    d_inner, d_state = 128, 16
    module = IntraCellularAttention(d_inner=d_inner, d_state=d_state)
    h = torch.randn(3, d_inner, d_state)
    out = module(h)
    assert out.shape == h.shape


def test_intra_attn_backward() -> None:
    """Gradients must flow through IntraCellularAttention."""
    d_inner, d_state = 64, 8
    module = IntraCellularAttention(d_inner=d_inner, d_state=d_state)
    with torch.no_grad():
        module.gate.fill_(1.0)
    h = torch.randn(2, d_inner, d_state, requires_grad=True)
    out = module(h)
    out.sum().backward()
    assert h.grad is not None
    assert module.W_q.weight.grad is not None
    assert module.W_k.weight.grad is not None


def test_intra_attn_ref_scan_noop_matches_no_attn() -> None:
    """With gate=0 the ref scan + intracellular attention must produce identical
    output to the ref scan without it."""
    torch.manual_seed(42)
    B, L, d_inner, d_state = 1, 16, 32, 8
    u     = torch.randn(B, L, d_inner)
    delta = torch.rand(B, L, d_inner) * 0.1 + 0.01
    A     = -torch.rand(d_inner, d_state)
    Bm    = torch.randn(B, L, d_state)
    C     = torch.randn(B, L, d_state)
    D     = torch.ones(d_inner)

    ia = IntraCellularAttention(d_inner=d_inner, d_state=d_state)
    # gate stays 0 → no-op

    y_plain = selective_scan_ref(u, delta, A, Bm, C, D)
    y_ia    = selective_scan_ref(u, delta, A, Bm, C, D, intra_attn=ia)
    assert torch.allclose(y_plain, y_ia, atol=1e-6), \
        "gate=0 intracellular attn should leave scan output unchanged"


def test_full_model_with_intra_attn_forward_and_backward() -> None:
    """Full forward + backward pass with use_intra_attn=True."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(use_intra_attn=True)
    model = Unnatam(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(ids)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    logits.float().mean().backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no grad for: {missing[:5]}"
