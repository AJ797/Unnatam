"""Fetch a held-out per-genre corpus for routing_controls.py.

Reviewer f2c6 asked for multi-document routing analysis (as opposed to our
original single-passage-per-genre analysis). This helper populates:

    data/heldout_genres/
        news.txt      (one document per line)
        fiction.txt
        code.txt
        dialog.txt
        tech.txt
        urgent.txt

Each file contains N documents; internal newlines are replaced with spaces so
each line is exactly one document (the format routing_controls.py expects).

Sources (all streaming mode; no large downloads):
    news     : abisee/cnn_dailymail       (article field, filter non-urgent)
    fiction  : roneneldan/TinyStories     (short story text)
    code     : codeparrot/github-code     (python only, decent length)
    dialog   : roskoN/dailydialog         (concat multi-turn dialogue)
    tech     : ccdv/arxiv-summarization   (article/abstract)
    urgent   : abisee/cnn_dailymail       (article, filtered for urgency keywords)

Requires: pip install datasets

Usage:
    py scripts/fetch_heldout_genres.py \
        --out data/heldout_genres \
        --n_per_genre 20 \
        --min_chars 300 --max_chars 2000

Then run:
    py scripts/routing_controls.py \
        --ckpt runs/tiny_hr_forced/ckpt_final.pt \
        --corpus_dir data/heldout_genres \
        --n_per_genre 20 --layer 11 --n_permutations 2000 \
        --out runs/tiny_hr_forced/routing_controls.json
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


# Keywords used to identify "urgent" documents in news streams
URGENCY_KEYWORDS = [
    "breaking", "emergency", "evacuate", "evacuation", "alert", "urgent",
    "disaster", "explosion", "fire", "shooting", "attack", "crisis",
    "rescue", "warning", "storm", "hurricane", "flood", "earthquake",
    "collapse", "outbreak", "casualties", "victims", "injured",
]
URGENCY_RE = re.compile(r"\b(" + "|".join(URGENCY_KEYWORDS) + r")\b", re.IGNORECASE)


def _normalise(text: str) -> str:
    """Collapse whitespace, strip, single-line."""
    return re.sub(r"\s+", " ", text or "").strip()


def _length_ok(text: str, lo: int, hi: int) -> bool:
    return lo <= len(text) <= hi


# ---------------------------------------------------------------------------
# Per-genre fetchers. Each returns a generator of doc strings.
# ---------------------------------------------------------------------------

def fetch_news(min_chars, max_chars, want_urgent=False):
    from datasets import load_dataset
    candidates = [
        # (repo, kwargs, text_field)
        ("abisee/cnn_dailymail", {"name": "3.0.0", "split": "test", "streaming": True}, "article"),
        ("ccdv/cnn_dailymail",  {"name": "3.0.0", "split": "test", "streaming": True}, "article"),
        ("cnn_dailymail",       {"name": "3.0.0", "split": "test", "streaming": True}, "article"),
        ("SetFit/bbc-news",     {"split": "test", "streaming": True}, "text"),
        ("ag_news",             {"split": "test", "streaming": True}, "text"),
    ]
    ds = None
    text_field = None
    tag = "urgent" if want_urgent else "news"
    for repo, kw, field in candidates:
        try:
            ds = load_dataset(repo, **kw)
            text_field = field
            print(f"  [{tag}] using {repo} field={field}")
            break
        except Exception as e:
            print(f"  [{tag}] {repo} unavailable: {type(e).__name__}: {str(e)[:80]}")
    if ds is None:
        print(f"  [{tag}] no news dataset available; skipping")
        return

    for row in ds:
        text = _normalise(row.get(text_field, ""))
        if not _length_ok(text, min_chars, max_chars):
            continue
        is_urgent = URGENCY_RE.search(text) is not None
        if want_urgent and not is_urgent:
            continue
        if (not want_urgent) and is_urgent:
            continue
        yield text


def fetch_fiction(min_chars, max_chars):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    for row in ds:
        text = _normalise(row.get("text", ""))
        if _length_ok(text, min_chars, max_chars):
            yield text


def fetch_code(min_chars, max_chars):
    """Try parquet-native code datasets. codeparrot/github-code is script-based
    and no longer supported by recent datasets versions."""
    from datasets import load_dataset

    candidates = [
        # (repo, kwargs, text_field)
        ("bigcode/the-stack-smol", {"data_dir": "data/python", "split": "train", "streaming": True}, "content"),
        ("bigcode/the-stack-smol-xs", {"data_dir": "data/python", "split": "train", "streaming": True}, "content"),
        ("codeparrot/codeparrot-clean-valid", {"split": "train", "streaming": True}, "content"),
        ("nuprl/MultiPL-E", {"name": "humaneval-py", "split": "test", "streaming": True}, "prompt"),
    ]

    ds = None
    text_field = None
    for repo, kw, field in candidates:
        try:
            ds = load_dataset(repo, **kw)
            text_field = field
            print(f"  [code] using {repo} field={field}")
            break
        except Exception as e:
            print(f"  [code] {repo} unavailable: {type(e).__name__}: {str(e)[:100]}")
    if ds is None:
        print("  [code] no code dataset available; skipping")
        return

    for row in ds:
        text = row.get(text_field, "") or ""
        if not text:
            continue
        # Preserve newlines by replacing with a paragraph symbol so each doc
        # remains one line in the output file (routing_controls.py expects
        # one doc per line).
        code_flat = text.replace("\n", " ¶ ").strip()
        if _length_ok(code_flat, min_chars, max_chars):
            yield code_flat


def fetch_dialog(min_chars, max_chars):
    """Try a few conversation datasets. daily_dialog family is script-based;
    prefer parquet-native chat datasets."""
    from datasets import load_dataset

    # ---------- Extractors specific to each dataset's schema ----------

    def _bst(row):
        """blended_skill_talk: previous_utterance + interleaved(free, guided)"""
        prev = row.get("previous_utterance") or []
        free = row.get("free_messages") or []
        guided = row.get("guided_messages") or []
        turns = list(prev)
        for f, g in zip(free, guided):
            if f: turns.append(f)
            if g: turns.append(g)
        return _normalise(" | ".join(str(t) for t in turns if t))

    def _messages_field(row):
        """HuggingFaceH4/no_robots-style: messages = [{role, content}, ...]"""
        msgs = row.get("messages") or row.get("conversations") or []
        if not msgs:
            return None
        parts = []
        for m in msgs:
            if isinstance(m, dict):
                parts.append(m.get("content") or m.get("value") or "")
            else:
                parts.append(str(m))
        return _normalise(" | ".join(p for p in parts if p))

    def _personachat(row):
        """bavard/personachat_truecased: history is a list of strings"""
        turns = row.get("history") or row.get("utterances") or []
        if not turns:
            return None
        if isinstance(turns[0], dict):
            turns = [t.get("text") or t.get("candidates", [""])[0] for t in turns]
        return _normalise(" | ".join(str(t) for t in turns if t))

    def _instruction_pair(row):
        """Instruction/response datasets (Alpaca-style): make it a tiny 2-turn dialog"""
        q = row.get("instruction") or row.get("prompt") or row.get("question") or ""
        a = row.get("response") or row.get("output") or row.get("answer") or ""
        if not (q and a):
            return None
        return _normalise(f"{q} | {a}")

    def _hh_style(row):
        """Anthropic HH-style: 'chosen' field is a multi-turn transcript string"""
        text = row.get("chosen") or row.get("rejected") or row.get("prompt") or ""
        # HH transcripts use "\n\nHuman: ... \n\nAssistant: ..." pattern
        text = re.sub(r"\n\n(Human|Assistant):\s*", " | ", text or "")
        return _normalise(text)

    def _generic_utterances(row):
        """Legacy schema: utterances/dialog/dialogue"""
        turns = row.get("utterances") or row.get("dialog") or row.get("dialogue") or []
        if isinstance(turns, dict):
            turns = turns.get("utterance", turns.get("text", []))
        if not turns:
            return None
        if isinstance(turns[0], dict):
            turns = [t.get("text", "") for t in turns]
        return _normalise(" | ".join(str(t) for t in turns if t))

    # ---------- Candidate list, in preference order ----------
    candidates = [
        ("HuggingFaceH4/no_robots",        {"split": "test",  "streaming": True}, _messages_field),
        ("bavard/personachat_truecased",   {"split": "test",  "streaming": True}, _personachat),
        ("Anthropic/hh-rlhf",              {"split": "test",  "streaming": True}, _hh_style),
        ("HuggingFaceH4/ultrachat_200k",   {"split": "test_sft", "streaming": True}, _messages_field),
        ("tatsu-lab/alpaca",               {"split": "train", "streaming": True}, _instruction_pair),
        ("blended_skill_talk",             {"split": "test",  "streaming": True}, _bst),
        # legacy script-based fallbacks (may fail on recent datasets versions)
        ("roskoN/dailydialog",             {"split": "test",  "streaming": True}, _generic_utterances),
    ]

    ds = None
    extractor = None
    src_name = None
    for repo, kw, extract in candidates:
        try:
            ds = load_dataset(repo, **kw)
            extractor = extract
            src_name = repo
            print(f"  [dialog] using {repo}")
            break
        except Exception as e:
            print(f"  [dialog] {repo} unavailable: {type(e).__name__}: {str(e)[:80]}")
    if ds is None:
        print("  [dialog] no dialog dataset available; skipping")
        return

    # Peek the first row for schema diagnostics
    peeked = False
    yielded = 0
    for row in ds:
        if not peeked:
            print(f"  [dialog] first-row keys: {list(row.keys())[:8]}")
            peeked = True
        text = extractor(row)
        if text and _length_ok(text, min_chars, max_chars):
            yield text
            yielded += 1
            if yielded >= 500:  # safety cap; caller stops at n_per_genre
                break

    if yielded == 0:
        print(f"  [dialog] WARN: {src_name} loaded but extractor produced 0 valid docs")


def fetch_tech(min_chars, max_chars):
    """Try a few academic/technical datasets. arxiv-summarization is often
    script-based; use parquet-native alternatives."""
    from datasets import load_dataset

    candidates = [
        ("ccdv/arxiv-summarization",   {"split": "test", "streaming": True},        ("abstract", "article")),
        ("armanc/scientific_papers",   {"name": "arxiv", "split": "test", "streaming": True}, ("article", "abstract")),
        ("scientific_papers",          {"name": "arxiv", "split": "test", "streaming": True}, ("article", "abstract")),
        ("neuralwork/arxiver",         {"split": "train", "streaming": True},       ("abstract", "markdown")),
        ("gfissore/arxiv-abstracts-2021", {"split": "train", "streaming": True},   ("abstract",)),
        ("EleutherAI/pile-freelaw",    {"split": "train", "streaming": True},       ("text",)),  # legal/technical last-resort
    ]

    ds = None
    fields = None
    for repo, kw, flds in candidates:
        try:
            ds = load_dataset(repo, **kw)
            fields = flds
            print(f"  [tech] using {repo} fields={flds}")
            break
        except Exception as e:
            print(f"  [tech] {repo} unavailable: {type(e).__name__}: {str(e)[:80]}")
    if ds is None:
        print("  [tech] no tech dataset available; skipping")
        return

    for row in ds:
        # Try each candidate field until one has content
        text = ""
        for f in fields:
            text = _normalise(row.get(f, "") or "")
            if text:
                break
        if _length_ok(text, min_chars, max_chars):
            yield text


GENRE_FETCHERS = {
    "news":    lambda mc, xc: fetch_news(mc, xc, want_urgent=False),
    "fiction": fetch_fiction,
    "code":    fetch_code,
    "dialog":  fetch_dialog,
    "tech":    fetch_tech,
    "urgent":  lambda mc, xc: fetch_news(mc, xc, want_urgent=True),
}


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/heldout_genres"))
    p.add_argument("--n_per_genre", type=int, default=20)
    p.add_argument("--min_chars", type=int, default=300)
    p.add_argument("--max_chars", type=int, default=2000)
    p.add_argument("--genres", nargs="+", default=list(GENRE_FETCHERS.keys()),
                   help="which genres to fetch (default: all six)")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    for g in args.genres:
        if g not in GENRE_FETCHERS:
            print(f"[skip] unknown genre {g}")
            continue
        out_path = args.out / f"{g}.txt"
        if out_path.exists() and len(out_path.read_text(encoding="utf-8").splitlines()) >= args.n_per_genre:
            print(f"[skip] {out_path} already has enough docs")
            continue

        print(f"[fetch] {g} → {out_path}")
        docs: list[str] = []
        gen = GENRE_FETCHERS[g](args.min_chars, args.max_chars)
        for doc in gen:
            docs.append(doc)
            if len(docs) >= args.n_per_genre:
                break
            if len(docs) % 5 == 0:
                print(f"  ... {len(docs)}/{args.n_per_genre}")

        if len(docs) < args.n_per_genre:
            print(f"  WARN: only got {len(docs)}/{args.n_per_genre} — corpus may be too filtered")

        out_path.write_text("\n".join(docs), encoding="utf-8")
        print(f"  wrote {len(docs)} docs to {out_path}")

    print("\n[done] corpus at", args.out.resolve())
    print("\nNow run:")
    print(f"  py scripts/routing_controls.py \\")
    print(f"      --ckpt runs/tiny_hr_forced/ckpt_final.pt \\")
    print(f"      --corpus_dir {args.out} \\")
    print(f"      --n_per_genre {args.n_per_genre} --layer 11 --n_permutations 2000 \\")
    print(f"      --out runs/tiny_hr_forced/routing_controls.json")
    print("\nAnd for the untrained-router baseline:")
    print(f"  py scripts/routing_controls.py \\")
    print(f"      --untrained \\")
    print(f"      --corpus_dir {args.out} \\")
    print(f"      --n_per_genre {args.n_per_genre} --layer 11 --n_permutations 2000 \\")
    print(f"      --out runs/untrained/routing_controls.json")


if __name__ == "__main__":
    main()
