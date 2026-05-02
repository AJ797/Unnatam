"""Zero-shot evaluation on standard benchmarks via loglikelihood scoring.

Implements the standard "loglikelihood of gold continuation" metric used by
lm-evaluation-harness — for each example we compute logP(continuation | context)
for each candidate and pick the highest. For LAMBADA we score the gold last
word (accuracy = exact match on top-1 token-level loglik prediction).

Avoids the lm-eval-harness dependency and HF model wrapping; pure PyTorch
forward passes through the Unnatam model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn.functional as F


@dataclass
class EvalResult:
    name: str
    metric: str          # "acc" or "acc_norm"
    value: float
    n_examples: int


# ---------------------------------------------------------------------------
# Tokenizer protocol — accepts tiktoken or transformers tokenizer
# ---------------------------------------------------------------------------

class _Tok:
    """Lightweight wrapper that presents .encode(str) -> list[int]."""

    def __init__(self, tokenizer):
        self._tok = tokenizer

    def encode(self, text: str) -> list[int]:
        if hasattr(self._tok, "encode_ordinary"):
            # tiktoken
            return self._tok.encode_ordinary(text)
        if hasattr(self._tok, "encode"):
            out = self._tok.encode(text)
            if isinstance(out, list):
                return out
            if hasattr(out, "ids"):
                return out.ids
            return list(out)
        raise ValueError("Unknown tokenizer interface")


# ---------------------------------------------------------------------------
# Core: loglikelihood of a continuation given a context
# ---------------------------------------------------------------------------

@torch.no_grad()
def _loglikelihood(
    model: torch.nn.Module,
    tokenizer: _Tok,
    context: str,
    continuation: str,
    device: str | torch.device,
    max_length: int = 1024,
) -> tuple[float, bool]:
    """Return (sum_logprob, is_greedy_argmax) for the continuation tokens.

    is_greedy_argmax is True iff every continuation token would have been the
    argmax prediction given the preceding context — used by HellaSwag's acc_norm.
    """
    ctx_ids = tokenizer.encode(context)
    cont_ids = tokenizer.encode(continuation)

    if not cont_ids:
        return 0.0, True

    # Concat and truncate from the LEFT so the continuation always fits.
    full_ids = ctx_ids + cont_ids
    if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]
        # Recompute boundary
        ctx_len = max_length - len(cont_ids)
    else:
        ctx_len = len(ctx_ids)

    if ctx_len < 1:  # need at least one preceding token to get logits
        # Pad with a single token (safe: this only happens for very long completions)
        full_ids = [tokenizer.encode(" ")[0]] + full_ids
        ctx_len = 1

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits = model(input_ids)                             # (1, T, V)
    # Logits at position i predict token at position i+1.
    # We want logits at positions [ctx_len-1 .. T-2] predicting cont_ids.
    target = input_ids[0, ctx_len:]                       # (n_cont,)
    pred_logits = logits[0, ctx_len - 1 : -1]             # (n_cont, V)
    log_probs = F.log_softmax(pred_logits.float(), dim=-1)
    ll = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1).sum().item()
    is_greedy = bool((pred_logits.argmax(dim=-1) == target).all().item())
    return ll, is_greedy


# ---------------------------------------------------------------------------
# Per-benchmark adapters
# ---------------------------------------------------------------------------

def _eval_multiple_choice(
    model, tokenizer, examples: Iterable[dict],
    get_context: Callable[[dict], str],
    get_choices: Callable[[dict], list[str]],
    get_label:   Callable[[dict], int],
    device, max_length: int = 1024,
    length_normalize: bool = False,
    progress_every: int = 100,
) -> tuple[float, int]:
    """Generic multiple-choice eval: pick the choice with highest sum(logP).

    If length_normalize=True, divide by len(continuation tokens) before argmax —
    HellaSwag's acc_norm does this.
    """
    correct = 0
    total = 0
    tok = _Tok(tokenizer)
    for ex in examples:
        ctx = get_context(ex)
        choices = get_choices(ex)
        label = get_label(ex)
        scores = []
        for ch in choices:
            ll, _ = _loglikelihood(model, tok, ctx, ch, device, max_length)
            if length_normalize:
                n_tok = max(1, len(tok.encode(ch)))
                scores.append(ll / n_tok)
            else:
                scores.append(ll)
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        correct += int(pred == label)
        total += 1
        if total % progress_every == 0:
            print(f"  [{total}] running acc={correct/total:.4f}", flush=True)
    return correct / max(1, total), total


def eval_hellaswag(model, tokenizer, device, max_examples: int | None = None,
                   max_length: int = 1024) -> list[EvalResult]:
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    print(f"[hellaswag] {len(ds)} examples")

    def ctx(ex):
        # Standard HellaSwag context: activity_label + ctx_a + ctx_b
        return ex["activity_label"] + ": " + ex["ctx"]
    def choices(ex):
        return [" " + e for e in ex["endings"]]
    def label(ex):
        return int(ex["label"])

    acc, n = _eval_multiple_choice(
        model, tokenizer, ds, ctx, choices, label, device, max_length,
        length_normalize=False,
    )
    acc_norm, _ = _eval_multiple_choice(
        model, tokenizer, ds, ctx, choices, label, device, max_length,
        length_normalize=True,
    )
    return [
        EvalResult("hellaswag", "acc", acc, n),
        EvalResult("hellaswag", "acc_norm", acc_norm, n),
    ]


def eval_lambada(model, tokenizer, device, max_examples: int | None = None,
                 max_length: int = 1024) -> list[EvalResult]:
    """LAMBADA: predict the LAST WORD given the context. Metric: accuracy."""
    from datasets import load_dataset
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    print(f"[lambada] {len(ds)} examples")

    correct = 0
    total = 0
    tok = _Tok(tokenizer)
    for ex in ds:
        text = ex["text"].strip()
        # Split off the last word as the target
        last_space = text.rfind(" ")
        if last_space == -1:
            continue
        ctx_str = text[:last_space]
        cont_str = " " + text[last_space + 1:]
        _, is_greedy = _loglikelihood(model, tok, ctx_str, cont_str, device, max_length)
        correct += int(is_greedy)
        total += 1
        if total % 200 == 0:
            print(f"  [{total}] running acc={correct/total:.4f}", flush=True)
    return [EvalResult("lambada", "acc", correct / max(1, total), total)]


def eval_arc(model, tokenizer, device, subset: str = "ARC-Easy",
             max_examples: int | None = None, max_length: int = 1024) -> list[EvalResult]:
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", subset, split="test")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    short = "arc_easy" if subset == "ARC-Easy" else "arc_challenge"
    print(f"[{short}] {len(ds)} examples")

    def ctx(ex):
        return f"Question: {ex['question']}\nAnswer:"
    def choices(ex):
        return [" " + t for t in ex["choices"]["text"]]
    def label(ex):
        # answerKey is e.g. "A" or "1"; map to index in choices.label list
        return int(ex["choices"]["label"].index(ex["answerKey"]))

    acc, n = _eval_multiple_choice(
        model, tokenizer, ds, ctx, choices, label, device, max_length,
        length_normalize=True,
    )
    return [EvalResult(short, "acc_norm", acc, n)]


def eval_piqa(model, tokenizer, device, max_examples: int | None = None,
              max_length: int = 1024) -> list[EvalResult]:
    from datasets import load_dataset
    # ybisk/piqa is script-based and broken on datasets >= 4.0; try parquet mirrors.
    ds = None
    for repo in ("lighteval/piqa_helm", "tanganke/piqa", "baber/piqa"):
        try:
            ds = load_dataset(repo, split="validation")
            print(f"[piqa] loaded from {repo}")
            break
        except Exception as e:
            print(f"[piqa] {repo} failed: {type(e).__name__}: {str(e)[:120]}")
    if ds is None:
        # Last resort: trust_remote_code on the script-based original
        try:
            ds = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
            print("[piqa] loaded from ybisk/piqa via trust_remote_code")
        except Exception as e:
            print(f"[piqa] all mirrors failed; skipping. Last error: {e}")
            return []
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    print(f"[piqa] {len(ds)} examples")

    def ctx(ex):
        return f"Question: {ex['goal']}\nAnswer:"
    def choices(ex):
        return [" " + ex["sol1"], " " + ex["sol2"]]
    def label(ex):
        return int(ex["label"])

    acc, n = _eval_multiple_choice(
        model, tokenizer, ds, ctx, choices, label, device, max_length,
        length_normalize=True,
    )
    return [EvalResult("piqa", "acc_norm", acc, n)]
