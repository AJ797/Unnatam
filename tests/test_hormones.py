"""Tests for the hormone extraction pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from unnatam.config import UnnatamConfig
from unnatam.hormones.definitions import HORMONE_CONTRASTS, HORMONE_NAMES, N_HORMONES
from unnatam.hormones.extract import extract_hormone_vectors, save_hormone_vectors
from unnatam.model import Unnatam


class CharTokenizer:
    """Minimal stand-in for HF tokenizers: chars → mod-vocab ids."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [ord(c) % self.vocab_size for c in text]


def _tiny_cfg(**overrides) -> UnnatamConfig:
    base = dict(
        d_model=64, n_layers=6, vocab_size=128, max_seq_len=512,
        n_heads=2, n_kv_heads=1, head_dim=32,
        d_state=4, d_conv=4, expand=2,
        ssm_attn_ratio=2, n_hormones=N_HORMONES,
    )
    base.update(overrides)
    return UnnatamConfig(**base)


def test_definitions_have_expected_shape():
    assert N_HORMONES == 7
    assert set(HORMONE_NAMES) == {"ADR", "CDO", "LCO", "NRA", "OXY", "SRO", "SELF"}
    for name in HORMONE_NAMES:
        pairs = HORMONE_CONTRASTS[name]
        assert len(pairs) >= 4, f"{name} has too few contrast pairs ({len(pairs)})"
        for pair in pairs:
            assert pair.positive and pair.negative
            assert pair.positive != pair.negative


def test_extract_shape_and_unit_norm():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = Unnatam(cfg)
    tok = CharTokenizer(vocab_size=cfg.vocab_size)

    vectors = extract_hormone_vectors(model, tok, device="cpu", max_tokens=64)

    assert vectors.shape == (N_HORMONES, cfg.d_model)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"norms not unit: {norms}"


def test_extract_vectors_differ_per_hormone():
    """Different hormones should produce different directions even on a random-init model
    (since their contrast prompts are different texts)."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = Unnatam(cfg)
    tok = CharTokenizer(vocab_size=cfg.vocab_size)
    vectors = extract_hormone_vectors(model, tok, device="cpu", max_tokens=64)

    # Pairwise cosine similarities — none should be ~1 (distinct directions).
    cos = vectors @ vectors.T
    np.fill_diagonal(cos, 0.0)
    assert np.abs(cos).max() < 0.99, f"two hormones collapsed: max off-diag |cos|={np.abs(cos).max():.3f}"


def test_extract_reproducible_for_fixed_model():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = Unnatam(cfg)
    tok = CharTokenizer(vocab_size=cfg.vocab_size)
    v1 = extract_hormone_vectors(model, tok, device="cpu", max_tokens=64)
    v2 = extract_hormone_vectors(model, tok, device="cpu", max_tokens=64)
    np.testing.assert_allclose(v1, v2, atol=1e-6)


def test_save_and_reload_into_model(tmp_path: Path):
    """Round-trip: extract → save → load into a new model with same d_model."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = Unnatam(cfg)
    tok = CharTokenizer(vocab_size=cfg.vocab_size)
    vectors = extract_hormone_vectors(model, tok, device="cpu", max_tokens=64)

    path = tmp_path / "hormones.npy"
    save_hormone_vectors(str(path), vectors)

    cfg2 = _tiny_cfg(hormone_vector_path=str(path))
    torch.manual_seed(1)  # different init for the rest of the model
    model2 = Unnatam(cfg2)
    np.testing.assert_allclose(model2.hormone_vectors.cpu().numpy(), vectors, atol=1e-6)


def test_mismatched_d_model_raises(tmp_path: Path):
    cfg = _tiny_cfg(d_model=64)
    bad = np.zeros((N_HORMONES, 128), dtype=np.float32)
    path = tmp_path / "bad.npy"
    np.save(path, bad)
    cfg.hormone_vector_path = str(path)
    try:
        Unnatam(cfg)
    except ValueError as e:
        assert "hormone_vectors" in str(e)
        return
    raise AssertionError("expected ValueError for d_model mismatch")
