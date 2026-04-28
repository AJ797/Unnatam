"""Datasets and dataloaders for Unnatam pretraining.

Two datasets:
  * SyntheticTokenDataset — random ids, used for smoke tests and overfit runs.
  * BinaryShardDataset    — memmaps a .bin file of pre-tokenized uint16/uint32
                            tokens (nanoGPT/llm.c convention) and returns
                            random (input, target) windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


class SyntheticTokenDataset(IterableDataset):
    """Yields random token batches of fixed length. Deterministic per-rank seed."""

    def __init__(self, vocab_size: int, seq_len: int, seed: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rng = np.random.default_rng(self.seed)
        while True:
            ids = rng.integers(0, self.vocab_size, size=self.seq_len + 1, dtype=np.int64)
            t = torch.from_numpy(ids)
            yield {"input_ids": t[:-1], "labels": t[1:]}


class BinaryShardDataset(IterableDataset):
    """Iterable dataset over one or more pre-tokenized .bin shards.

    Each shard is a flat array of token ids (dtype matches the dump dtype:
    uint16 for vocab <= 65536, else uint32). At each step we sample a random
    offset and slice out (seq_len + 1) tokens to form an (input, target) pair.
    """

    def __init__(
        self,
        shard_paths: list[str | Path],
        seq_len: int,
        dtype: np.dtype = np.uint16,
        seed: int = 0,
    ):
        super().__init__()
        if not shard_paths:
            raise ValueError("shard_paths is empty")
        self.shard_paths = [Path(p) for p in shard_paths]
        self.seq_len = seq_len
        self.dtype = dtype
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = np.random.default_rng(self.seed + worker_id)

        # Memmap each shard once.
        shards = [np.memmap(p, dtype=self.dtype, mode="r") for p in self.shard_paths]
        weights = np.array([len(s) for s in shards], dtype=np.float64)
        weights /= weights.sum()

        while True:
            shard_idx = int(rng.choice(len(shards), p=weights))
            buf = shards[shard_idx]
            max_start = len(buf) - self.seq_len - 1
            start = int(rng.integers(0, max_start))
            window = buf[start : start + self.seq_len + 1].astype(np.int64, copy=False)
            t = torch.from_numpy(window)
            yield {"input_ids": t[:-1], "labels": t[1:]}


def build_dataloader(
    dataset: IterableDataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
