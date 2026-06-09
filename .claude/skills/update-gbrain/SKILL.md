---
name: update-gbrain
description: Write a run's findings (insights, edges) into the gbrain graph memory and snapshot them, keeping the graph fresh via auto-linking.
tools: ["Read"]
when_to_use: After surfacing pain points, to persist insights so future runs build on them.
---

# Skill: Update gbrain

## Goal
Persist this run's reasoning into memory so the brain accumulates across periods.

## Steps
1. For each finding from `surface-pain-points`, call `write_memory(insight)` with:
   - `summary`, `root_cause`, `recommended_action`, `status` (`draft` until reviewed).
   - `evidence` = exemplar `transcript_id`s + `period_metric` ids backing the claim.
2. The store auto-links typed edges (no LLM): `explains` (insight→pain_point),
   `cites` (insight→exemplar/period_metric).
3. Confirm the run's `run` node + JSON snapshot exist in `gbrain/store/snapshots/`.

## Done when
- New `insight` nodes are linked to their pain points and cite evidence.
- `read_memory(l5_id)` on a future run returns these insights for trend comparison.

## Guardrails
- Don't duplicate an existing insight — update it (idempotent upsert by content/id).
- Insights are hypotheses; mark `status=draft` until a human confirms.
