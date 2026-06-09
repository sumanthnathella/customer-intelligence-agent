"""
shared/text.py — Lightweight text utilities (no external NLP deps).

Provides sentence segmentation and BM25-style extractive compression, used to
turn long transcripts into sharp evidence snippets. Inspired by Harness-1's
sentence-BM25 compression of search results (keep only the most query-relevant
sentences instead of truncating arbitrarily).
"""
from __future__ import annotations

import math
import re

# TWCS conversations are concatenated with " | "; also split on sentence enders.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\s*\|\s*|\n+")
_WORD = re.compile(r"[a-z0-9']+")

_BM25_K1 = 1.5
_BM25_B = 0.75


def split_sentences(text: str) -> list[str]:
    """Split text into trimmed, non-empty sentence-like fragments."""
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def _tokenize(s: str) -> list[str]:
    return _WORD.findall(s.lower())


def compress_to_sentences(
    text: str,
    query: str,
    k: int = 2,
    max_chars: int = 500,
) -> str:
    """Return the ``k`` sentences most relevant to ``query`` via BM25.

    Sentences are treated as the document collection; the query terms (e.g. an
    L5 leaf definition) score each sentence. Selected sentences are returned in
    their original order so the snippet stays readable. Falls back gracefully
    for short texts or empty queries.
    """
    sentences = split_sentences(text)
    if len(sentences) <= k:
        return text.strip()[:max_chars]

    q_terms = set(_tokenize(query))
    if not q_terms:
        return " ".join(sentences[:k])[:max_chars]

    sent_tokens = [_tokenize(s) for s in sentences]
    n = len(sentences)
    avgdl = sum(len(t) for t in sent_tokens) / max(n, 1)

    # document frequency per term across sentences
    df: dict[str, int] = {}
    for toks in sent_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    scored: list[tuple[float, int]] = []
    for i, toks in enumerate(sent_tokens):
        dl = len(toks)
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for t in q_terms:
            if t not in tf:
                continue
            idf = math.log(1 + (n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            denom = tf[t] + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / max(avgdl, 1.0))
            score += idf * (tf[t] * (_BM25_K1 + 1)) / denom
        scored.append((score, i))

    top = sorted(scored, key=lambda x: x[0], reverse=True)[:k]
    if all(s <= 0 for s, _ in top):
        return " ".join(sentences[:k])[:max_chars]
    keep = sorted(i for _, i in top)
    return " ".join(sentences[i] for i in keep)[:max_chars]


def keyword_overlap_score(text: str, query: str) -> float:
    """Fraction of distinct query terms present in ``text`` (0..1).

    Cheap relevance signal for grep-style transcript search.
    """
    q = set(_tokenize(query))
    if not q:
        return 0.0
    t = set(_tokenize(text))
    return len(q & t) / len(q)
