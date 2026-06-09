# Tagging Producer (one-time)

A **replaceable producer** that turns transcripts into tags. Runs **locally on Ollama** — free,
unlimited, no rate caps. Lives in `tagging/`; imports only `shared/`. A BYO user with existing tags
can skip this module entirely (just satisfy Table A of [`INPUT_CONTRACT.md`](INPUT_CONTRACT.md)).

> Enrichment (operational dimensions) is a **separate** producer — see `datagen/`. Severity, metrics,
> and gbrain writes happen in the **core** (`analytics/`), not here.

## Stages (`tagging/run.py`)

```
ingest → taxonomy (build only) → tag
```

| Stage | File | Output |
|-------|------|--------|
| Ingest | `ingest.py` | conversations parquet (build set ~1M; reserve ~1.8M) |
| Taxonomy | `taxonomy.py` | `taxonomy.json` schema pack (≤100 L5), one-time on build |
| Tag | `tag.py` | per-transcript `{l5_id, sentiment, severity SIGNALS, confidence}` |

The full build (`tagging → datagen → analytics → gbrain`) is wired by the top-level orchestrator; see
below.

## Local model

- **`qwen2.5:7b`** via Ollama, **JSON-schema-constrained** output (`format` = schema) for reliable
  single-leaf assignment. Tools-capable, 32k ctx, ~4.7GB — fast on the M4 24GB.
- **`nomic-embed-text`** for taxonomy embeddings/clustering.
- Fallbacks for nuance if needed: `gemma4:12b`.

## Throughput, caching, resumability

- **Batched**: N transcripts per LLM call, returning a JSON array.
- **Cached**: keyed by `hash(transcript_text + pack_version + prompt_version)` → re-runs are free.
- **Resumable**: a manifest tracks completed batches; interrupt/restart is safe (matters for the
  millions in test batches).
- **Sample size**: ~10k for build (configurable). Test batches process arbitrary slices of the reserve.

## Commands

```bash
# Tagging only (produces Table A tags)
uv run python -m tagging.run --build --sample 10000
uv run python -m tagging.run --test-batch 1
uv run python -m tagging.run --resume

# Full demo build, orchestrated end-to-end (tagging → datagen → analytics → gbrain)
uv run python -m build --demo --sample 10000
uv run python -m build --test-batch 1
```

The top-level `build` orchestrator calls the producers + core in order; the modules never import each
other (orchestrator-only wiring keeps the boundary clean).

## Determinism

Seeded datagen + cached tags + frozen taxonomy ⇒ reproducible analytics. The same input always
produces the same gbrain state (important for trustworthy trend/z-score comparisons).
