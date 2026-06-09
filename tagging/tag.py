"""
tagging/tag.py — Tag conversations with L5 + severity signals using local Ollama.

Output per conversation (Table A):
    conversation_id, l5_id, sentiment, churn_intent, financial_harm,
    safety_legal, repeat_contact, unresolved, confidence

Design:
- JSON-constrained output via Ollama `format` parameter.
- Batched (N conversations per LLM call).
- Cached by hash(text + pack_version + prompt_version) — re-runs are free.
- Resumable: manifest tracks completed batches.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import pandas as pd

from shared.config import (
    API_TAGGER_BATCH_SIZE,
    PACK_VERSION,
    PROMPT_VERSION,
    TAGGER_BATCH_SIZE,
    TAGGER_CACHE_DIR,
    TAGGER_MODEL,
)

logger = logging.getLogger(__name__)

_SENTIMENT_MAP = {
    "negative": "neg",
    "positive": "pos",
    "very negative": "very_neg",
    "very_negative": "very_neg",
    "very positive": "pos",
    "very_positive": "pos",
}


_VALID_SENTIMENTS = {"very_neg", "neg", "neutral", "pos"}


def _normalise_sentiment(raw: str) -> str:
    """Map free-form LLM sentiment to a canonical enum value (defaults to 'neg')."""
    key = str(raw).strip().lower()
    key = _SENTIMENT_MAP.get(key, key)
    return key if key in _VALID_SENTIMENTS else "neg"


def _snap_l5(raw: str, valid_ids: list[str]) -> str | None:
    """Snap an LLM-emitted l5_id onto the closest valid taxonomy leaf.

    LLMs frequently truncate ids or emit the wrong L1 prefix (e.g.
    ``order_status_inquiry`` or ``post_sale__purchase__...__login_issue``).
    We recover these by matching on the leaf segment, then on substring.
    Returns ``None`` when nothing plausible matches.
    """
    if not raw:
        return None
    if raw in valid_ids:
        return raw
    leaf = raw.split("__")[-1]
    exact_leaf = [v for v in valid_ids if v.split("__")[-1] == leaf]
    if exact_leaf:
        return exact_leaf[0]
    contains = [v for v in valid_ids if leaf and leaf in v]
    if contains:
        return min(contains, key=len)
    return None


# ---------------------------------------------------------------------------
# JSON schema for constrained output
# ---------------------------------------------------------------------------

_TAG_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "conversation_id": {"type": "string"},
                    "l5_id": {"type": "string"},
                    "sentiment": {"type": "string", "enum": ["very_neg", "neg", "neutral", "pos"]},
                    "churn_intent": {"type": "integer", "enum": [0, 1]},
                    "financial_harm": {"type": "integer", "enum": [0, 1]},
                    "safety_legal": {"type": "integer", "enum": [0, 1]},
                    "repeat_contact": {"type": "integer", "enum": [0, 1]},
                    "unresolved": {"type": "integer", "enum": [0, 1]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "conversation_id",
                    "l5_id",
                    "sentiment",
                    "churn_intent",
                    "financial_harm",
                    "safety_legal",
                    "repeat_contact",
                    "unresolved",
                ],
            },
        }
    },
    "required": ["tags"],
}

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(batch: list[dict[str, str]], taxonomy_nodes: dict[str, Any]) -> str:
    leaf_list = "\n".join(
        f"  {lid}: {node.get('definition', lid)}"
        for lid, node in list(taxonomy_nodes.items())[:50]
    )
    conv_block = "\n".join(
        f'[{item["conversation_id"]}] {item["text"][:200]}'
        for item in batch
    )
    return f"""You are a customer support analyst tagging support conversations.

Taxonomy leaves (L5 IDs and definitions):
{leaf_list}

For each conversation below, output exactly one JSON tag with:
- l5_id: the single best matching leaf ID from the taxonomy
- sentiment: very_neg / neg / neutral / pos  (overall customer tone)
- churn_intent: 1 if customer signals leaving, else 0
- financial_harm: 1 if mentions overcharge/refund/money loss, else 0
- safety_legal: 1 if mentions safety hazard or legal threat, else 0
- repeat_contact: 1 if customer says this is a repeat contact/complaint, else 0
- unresolved: 1 if issue is still unresolved at end, else 0
- confidence: your confidence 0-1

Conversations:
{conv_block}

Return ONLY a JSON object in this exact format:
{{
  "tags": [
    {{
      "conversation_id": "<id from above>",
      "l5_id": "<best matching leaf ID>",
      "sentiment": "neg",
      "churn_intent": 0,
      "financial_harm": 0,
      "safety_legal": 0,
      "repeat_contact": 0,
      "unresolved": 1,
      "confidence": 0.85
    }},
    ...
  ]
}}

One entry per conversation, in the same order as listed above. Output ONLY the JSON, no explanation.
"""


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_key(conversation_id: str, text: str) -> str:
    h = hashlib.sha256(
        f"{conversation_id}|{text[:1000]}|{PACK_VERSION}|{PROMPT_VERSION}".encode()
    ).hexdigest()
    return h


def _load_from_cache(conversation_id: str, text: str) -> dict[str, Any] | None:
    key = _cache_key(conversation_id, text)
    path = TAGGER_CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def _save_to_cache(conversation_id: str, text: str, result: dict[str, Any]) -> None:
    key = _cache_key(conversation_id, text)
    path = TAGGER_CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(result))


# ---------------------------------------------------------------------------
# API call (OpenRouter / Anthropic / OpenAI via LiteLLM)
# ---------------------------------------------------------------------------

def _is_api_model(model: str) -> bool:
    """Detect if model string refers to an API provider (not local Ollama)."""
    return "/" in model and not model.startswith("ollama")


def _call_llm(prompt: str, model: str = TAGGER_MODEL) -> dict[str, Any] | None:
    if _is_api_model(model):
        return _call_api(prompt, model)
    return _call_ollama(prompt, model)


def _call_api(prompt: str, model: str) -> dict[str, Any] | None:
    """Call an external LLM API via LiteLLM."""
    try:
        import litellm
        litellm.drop_params = True
    except ImportError as exc:
        raise ImportError("litellm package required for API tagging: pip install litellm") from exc

    try:
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": "You are a customer support analyst. Output strictly valid JSON with a top-level 'tags' array."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content or "{}"
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        return json.loads(raw)
    except Exception as exc:
        logger.error("API LLM call failed (%s): %s", model, exc)
        return None


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, model: str = TAGGER_MODEL) -> dict[str, Any] | None:
    try:
        import ollama  # type: ignore
    except ImportError as exc:
        raise ImportError("ollama package required: pip install ollama") from exc

    try:
        resp = ollama.generate(
            model=model,
            prompt=prompt,
            format=_TAG_SCHEMA,
            options={"temperature": 0},
            keep_alive="20m",
        )
        raw = resp.get("response", "{}")
        return json.loads(raw)
    except Exception as exc:
        logger.error("Ollama call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main tagging function
# ---------------------------------------------------------------------------

def tag_conversations(
    df: pd.DataFrame,
    taxonomy_nodes: dict[str, Any],
    model: str = TAGGER_MODEL,
    batch_size: int | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Tag a DataFrame of conversations.

    Input df: conversation_id, text (and optionally created_at).
    Output: same rows + l5_id, sentiment, churn_intent, financial_harm,
                            safety_legal, repeat_contact, unresolved, confidence.
    """
    if batch_size is None:
        batch_size = API_TAGGER_BATCH_SIZE if _is_api_model(model) else TAGGER_BATCH_SIZE

    records: list[dict[str, Any]] = []
    uncached_batch: list[dict[str, str]] = []
    uncached_ids: list[str] = []

    for _, row in df.iterrows():
        cid = str(row["conversation_id"])
        text = str(row.get("text", ""))

        if use_cache:
            cached = _load_from_cache(cid, text)
            if cached:
                records.append(cached)
                continue

        uncached_batch.append({"conversation_id": cid, "text": text})
        uncached_ids.append(cid)

        if len(uncached_batch) >= batch_size:
            _process_batch(uncached_batch, taxonomy_nodes, model, use_cache, df, records)
            uncached_batch = []
            uncached_ids = []

    if uncached_batch:
        _process_batch(uncached_batch, taxonomy_nodes, model, use_cache, df, records)

    if not records:
        return pd.DataFrame(columns=[
            "conversation_id", "l5_id", "sentiment",
            "churn_intent", "financial_harm", "safety_legal",
            "repeat_contact", "unresolved", "confidence",
        ])

    return pd.DataFrame(records)


def _process_batch(
    batch: list[dict[str, str]],
    taxonomy_nodes: dict[str, Any],
    model: str,
    use_cache: bool,
    df_orig: pd.DataFrame,
    results: list[dict[str, Any]],
) -> None:
    prompt = _build_prompt(batch, taxonomy_nodes)
    response = _call_llm(prompt, model=model)

    if response is None:
        logger.warning("LLM returned None for batch of %d; using fallback.", len(batch))
        for item in batch:
            fallback = _fallback_tag(item["conversation_id"], list(taxonomy_nodes.keys()))
            results.append(fallback)
        return

    # Handle both {"tags": [...]} and direct [...] response shapes
    if isinstance(response, dict):
        tags = response.get("tags", [])
    elif isinstance(response, list):
        tags = response
    else:
        tags = []
    tag_by_id = {t["conversation_id"]: t for t in tags if isinstance(t, dict) and "conversation_id" in t}

    valid_ids = list(taxonomy_nodes.keys())
    for item in batch:
        cid = item["conversation_id"]
        tag = tag_by_id.get(cid, _fallback_tag(cid, valid_ids))
        tag["conversation_id"] = cid
        # Normalise sentiment to contracted enum values
        tag["sentiment"] = _normalise_sentiment(tag.get("sentiment", "neg"))
        # Snap l5_id onto a valid taxonomy leaf (LLMs often truncate / mis-prefix)
        snapped = _snap_l5(str(tag.get("l5_id", "")), valid_ids)
        tag["l5_id"] = snapped or (valid_ids[0] if valid_ids else "unknown")
        if use_cache:
            _save_to_cache(cid, item["text"], tag)
        results.append(tag)


def _fallback_tag(conversation_id: str, leaf_ids: list[str]) -> dict[str, Any]:
    """Safe default when LLM fails."""
    return {
        "conversation_id": conversation_id,
        "l5_id": leaf_ids[0] if leaf_ids else "unknown",
        "sentiment": "neg",
        "churn_intent": 0,
        "financial_harm": 0,
        "safety_legal": 0,
        "repeat_contact": 0,
        "unresolved": 1,
        "confidence": 0.0,
    }
