"""
analytics/subthemes.py — Emergent sub-themes *in words* within each L5.

The L5 leaf tells you the bucket ("damaged_item_received"); this tells you what
is actually going on inside it: the distinctive issues customers raise, grouped,
counted, and each illustrated with a representative quote.

Method (deterministic, no LLM):
  1. Fit one TF-IDF model over the whole corpus (1–2 grams, domain stop-words).
  2. For each L5, score every term by *distinctiveness* = mean TF-IDF inside the
     L5 minus mean TF-IDF in the rest of the corpus. High-distinctiveness terms
     are what makes this pain point different from everything else.
  3. Keep the top non-overlapping phrases, count how many of the L5's contacts
     mention each, and attach the highest-severity contact as the exemplar quote.

This turns "66 damaged-item contacts" into "battery/charging (18), cracked
screen (12), packaging/leaking (9)" — the ammo needed to investigate.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from shared.text import compress_to_sentences

logger = logging.getLogger(__name__)

# Support-desk boilerplate and TWCS artifacts that are not informative themes.
_DOMAIN_STOPWORDS = {
    "dm", "pm", "dms", "please", "sorry", "hi", "hey", "hello", "thanks", "thank",
    "team", "send", "us", "know", "let", "ll", "ve", "re", "hear", "help", "happy",
    "apologies", "apologize", "reach", "able", "look", "looking", "looked", "info",
    "information", "account", "number", "email", "phone", "name", "zip", "details",
    "detail", "assist", "issue", "issues", "concern", "concerns", "experience",
    "service", "customer", "order", "orders", "amp", "https", "http", "com", "www",
    "dont", "didnt", "cant", "wont", "ive", "im", "youre", "weve", "thats", "got",
    "get", "would", "could", "like", "want", "need", "just", "going", "make", "sure",
    "today", "day", "days", "week", "time", "back", "still", "via", "also", "one",
    "see", "take", "way", "thank_you", "let_us", "send_us", "reach_out", "dm_us",
    "happened", "thing", "things", "guys", "gonna", "wanna", "yall", "lol",
    "yeah", "okay", "oh", "good", "really", "guy", "people", "say", "said", "tell",
}


def _representative_quote(texts: pd.Series, severities: pd.Series, term: str) -> dict:
    """Pick the highest-severity contact mentioning ``term`` and compress it."""
    pattern = re.compile(r"\b" + re.escape(term.replace("_", " ")) + r"\b", re.IGNORECASE)
    mask = texts.str.contains(pattern, na=False)
    if not mask.any():
        return {}
    sub = severities[mask].sort_values(ascending=False)
    idx = sub.index[0]
    snippet = compress_to_sentences(str(texts.loc[idx]), query=term.replace("_", " "), k=1, max_chars=200)
    return {"snippet": snippet, "severity": float(severities.loc[idx]), "count": int(mask.sum())}


def _dedupe_overlapping(terms: list[str]) -> list[str]:
    """Drop a phrase if it is wholly contained in an already-kept phrase."""
    kept: list[str] = []
    for t in terms:
        t_norm = t.replace("_", " ")
        if any(t_norm in k.replace("_", " ") or k.replace("_", " ") in t_norm for k in kept):
            continue
        kept.append(t)
    return kept


def compute_subthemes(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    l5_col: str = "l5_id",
    severity_col: str = "severity",
    top_themes: int = 5,
    min_docs_per_theme: int = 4,
    min_l5_docs: int = 15,
    candidate_terms: int = 25,
) -> dict[str, list[dict]]:
    """Return {l5_id: [subtheme, ...]} ordered by distinctiveness.

    Each subtheme: {label, count, share, quote, severity}. L5s with too few
    contacts to characterise are returned with an empty list.
    """
    if df.empty or text_col not in df.columns:
        return {}

    texts_all = df[text_col].fillna("").astype(str)
    if severity_col in df.columns:
        sev_all = df[severity_col].fillna(0.0).astype(float)
    else:
        sev_all = pd.Series(1.0, index=df.index)

    stop = list(TfidfVectorizer(stop_words="english").get_stop_words() | _DOMAIN_STOPWORDS)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-z][a-z]{2,}\b",
        stop_words=stop,
        min_df=5,
        max_df=0.4,
        sublinear_tf=True,
        norm="l2",
    )
    try:
        x = vectorizer.fit_transform(texts_all)
    except ValueError:
        # Vocabulary empty (tiny/degenerate corpus) — nothing to mine.
        return {l5: [] for l5 in df[l5_col].astype(str).unique()}

    vocab = np.array(vectorizer.get_feature_names_out())
    n_total = x.shape[0]
    global_sum = np.asarray(x.sum(axis=0)).ravel()

    out: dict[str, list[dict]] = {}
    for l5_id, idx in df.reset_index(drop=True).groupby(l5_col).groups.items():
        rows = np.array(list(idx))
        n_cls = len(rows)
        l5_key = str(l5_id)
        if n_cls < min_l5_docs:
            out[l5_key] = []
            continue

        class_sum = np.asarray(x[rows].sum(axis=0)).ravel()
        class_mean = class_sum / n_cls
        bg_n = n_total - n_cls
        bg_mean = (global_sum - class_sum) / bg_n if bg_n > 0 else np.zeros_like(class_sum)
        distinct = class_mean - bg_mean

        top_idx = np.argsort(distinct)[::-1][:candidate_terms]
        candidates = [vocab[i] for i in top_idx if distinct[i] > 0]
        candidates = _dedupe_overlapping(candidates)

        cls_texts = texts_all.iloc[rows]
        cls_sev = sev_all.iloc[rows]

        themes: list[dict] = []
        for term in candidates:
            q = _representative_quote(cls_texts, cls_sev, term)
            if not q or q["count"] < min_docs_per_theme:
                continue
            themes.append(
                {
                    "label": term.replace("_", " "),
                    "count": q["count"],
                    "share": round(q["count"] / n_cls, 3),
                    "quote": q["snippet"],
                    "severity": round(q["severity"], 1),
                }
            )
            if len(themes) >= top_themes:
                break
        out[l5_key] = themes

    return out
