# `tagging/` — Producer (one-time)

A **replaceable producer** that turns raw transcripts into tags (Table A of the input contract).
Runs locally on Ollama (free, unlimited). Imports only `shared/`. BYO users with existing tags skip it.

See [`../docs/TAGGING.md`](../docs/TAGGING.md) for the full spec. Enrichment lives in `datagen/`;
severity/metrics/gbrain-writes live in `analytics/`.

## Modules (to implement)
| File | Role |
|------|------|
| `ingest.py` | `twcs.csv` → reconstructed conversations (build vs test split) |
| `taxonomy.py` | embed+cluster → LLM-named L1–L5 schema pack (≤100 L5) |
| `tag.py` | classify each transcript into one L5 leaf + **severity signals** (local LLM, JSON-constrained) |
| `run.py` | tagging orchestrator: `--build`, `--test-batch N`, `--resume` |

## Run
```bash
uv run python -m tagging.run --build --sample 10000   # tags only
uv run python -m build --demo --sample 10000          # full build (tagging→datagen→analytics→gbrain)
```
