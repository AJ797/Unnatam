from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from unnatam.config import UnnatamConfig
from unnatam.model.block import AttnLayer, SSMLayer
from unnatam.model.hormone import HormoneRouter
from unnatam.model.norm import RMSNorm

UnnatamBlock = AttnLayer  # alias for the layer that owns the hormone routing


class Unnatam(nn.Module):
    def __init__(self, cfg: UnnatamConfig):
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing: bool = False

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Frozen hormone-vector dictionary, shared across attention layers.
        self.register_buffer("hormone_vectors", self._load_hormone_vectors(cfg))

        layers: list[nn.Module] = []
        for kind in cfg.layer_kinds:
            layers.append(SSMLayer(cfg) if kind == "ssm" else AttnLayer(cfg))
        self.layers = nn.ModuleList(layers)

        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

        if cfg.tie_word_embeddings:
            self.lm_head: nn.Linear | None = None
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _load_hormone_vectors(cfg: UnnatamConfig) -> torch.Tensor:
        if cfg.hormone_random:
            # HR-rand ablation: random unit-norm vectors, same shape as extracted bank.
            v = torch.randn(cfg.n_hormones, cfg.d_model)
            v = torch.nn.functional.normalize(v, dim=-1)
            return v
        if cfg.hormone_vector_path is not None:
            arr = np.load(Path(cfg.hormone_vector_path))
            v = torch.from_numpy(arr).float()
            if v.shape != (cfg.n_hormones, cfg.d_model):
                raise ValueError(
                    f"hormone_vectors at {cfg.hormone_vector_path} has shape {tuple(v.shape)}, "
                    f"expected {(cfg.n_hormones, cfg.d_model)}"
                )
            return v
        # Pre-extraction: small random init (gate is 0 so this doesn't affect output).
        return torch.randn(cfg.n_hormones, cfg.d_model) * cfg.init_std

    def _init_weights(self, module: nn.Module) -> None:
        if getattr(module, "_no_reinit", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        if self.gradient_checkpointing and self.training:
            for layer in self.layers:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, self.hormone_vectors, use_reentrant=False
                )
        else:
            for layer in self.layers:
                x = layer(x, self.hormone_vectors)
        x = self.final_norm(x)
        if self.lm_head is None:
            return x @ self.embed.weight.t()
        return self.lm_head(x)

    def freeze_hormone_gates(self, gate_value: float = 1.0) -> None:
        """HR-fixedgate ablation: set every HormoneRouter gate to gate_value and freeze it."""
        for module in self.modules():
            if isinstance(module, HormoneRouter):
                module.gate.data.fill_(gate_value)
                module.gate.requires_grad_(False)

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))
