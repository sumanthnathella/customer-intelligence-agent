---
name: tag-transcripts
description: Run the build or test-batch tagging pipeline (ingest → enrich → taxonomy → tag → aggregate → write gbrain) locally on Ollama.
tools: ["Read", "Bash", "Edit"]
when_to_use: When new transcripts arrive, or to (re)build the taxonomy + tagged sample.
---

# Skill: Tag Transcripts

## Goal
Turn raw transcripts into tagged, aggregated facts written to gbrain — locally and for free.

## Preconditions
- `data/raw/twcs/twcs.csv` present (unzip `archive.zip`).
- Ollama running with `qwen2.5:7b` and `nomic-embed-text`.

## Steps
1. **Build (one-time, orchestrated)**
   ```bash
   uv run python -m build --demo --sample 10000
   ```
   Runs `tagging → datagen → analytics → gbrain`: produces `taxonomy.json` (≤100 L5), tags, synthetic
   dims (demo), metrics/drivers, and the populated L5-centric gbrain. (Tagging only: `python -m tagging.run --build`.)
2. **Verify** the taxonomy: unique paths, ≤100 leaves, low `__unmapped__` rate (`docs/TAXONOMY.md`).
3. **Test batch (ongoing)**
   ```bash
   uv run python -m build --test-batch 1
   ```
   Tags the next reserve slice against the **frozen** pack, re-runs analytics, and appends
   `period_metric` nodes + updated driver edges to gbrain.
4. **Resume** an interrupted run with `--resume` (batches are cached + manifested).

## Done when
- gbrain has new/updated `pain_point` + `period_metric` nodes and a JSON snapshot for the run.

## Guardrails
- Never invent taxonomy paths during tagging (classification only).
- Bulk tagging stays on Ollama — do not call OpenRouter here.
