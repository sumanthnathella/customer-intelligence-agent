---
name: surface-pain-points
description: Rank the most egregious L5 pain points by volume and recent-week z-score, then deep-dive each with evidence, dimension cuts, and trend vs prior periods.
tools: ["Read"]
when_to_use: The core analysis step the agent performs before writing a report.
---

# Skill: Surface Pain Points

## Goal
Identify and explain the pain points that matter most *right now*.

## Steps
1. `get_taxonomy()` — load the L1–L5 schema pack.
2. Rank candidates:
   - `get_top_l5(by="egregiousness", n=10)`
   - `get_zscore_spikes(weeks=4)` — what changed recently.
3. For each candidate (deduped):
   - `get_exemplars(l5_id, k=3)` — verbatim evidence.
   - `cut_by_dimension(l5_id, dim)` for the dims it over-indexes on (carrier, region, channel, segment…).
   - `read_memory(l5_id)` — prior periods + prior insights → compute trend deltas.
4. Form a root-cause hypothesis + recommended action per pain point.

## Output
A structured set of findings (per `docs/REPORT_SPEC.md`) ready for `generate-report`.

## Guardrails
- Use only tool-provided numbers (gbrain/aggregates). Never estimate metrics.
- Surface spikes alongside raw volume — don't bury recency.
