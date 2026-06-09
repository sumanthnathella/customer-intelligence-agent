# Evaluation & Verification

How we keep the system correct and trustworthy.

## Unit tests (`tests/unit/`)
- **Taxonomy** — paths are unique; ≤100 L5 leaves; every leaf has a stable `l5_id`.
- **Severity** — rubric maps known signal combos to expected 1–5 scores; safety_legal dominates; clamps.
- **Z-score** — known weekly series produce expected z-scores; spike detection edge cases (zero variance, short history).
- **Egregiousness** — monotonic in each input; deterministic given weights; `w4` redistributes when no monetary fact.
- **Drivers** — lift/support/p-value correct on a fixed contingency; BH-FDR filters noise; min-support honored.
- **gbrain** — `upsert_node`/`upsert_edge` idempotency; `affects` edge props update (not duplicate); `traverse` both directions (pain-first + ops-first); snapshot round-trips.
- **Contract** — `shared/contract.py` accepts a valid BYO dataset and rejects missing columns / broken joins.

## Driver-recovery test (methodology validation)
- `datagen/` plants drivers via its documented model; `analytics/drivers.py` must **recover** them:
  assert recovered lift ≈ planted lift (within tolerance) for the planted (l5, dimension) pairs, and
  near-1.0 lift for un-planted pairs. Validates the exact analysis a BYO user would run on real data.

## Integration test (`tests/integration/`)
- Tiny bundled **fixture** (~200 conversations) runs the full pipeline end-to-end → a populated gbrain
  + a rendered report. Asserts: report has all required sections, every numeric claim maps to a node id,
  gbrain has expected node/edge counts.

## Tagging quality (sample audit)
- Hold out a small **human-labeled** set; measure agreement of `l5_id` assignment and severity.
- Track `__unmapped__` rate; if high, the taxonomy is too narrow → re-induce.

## Agent quality
- **Faithfulness**: assert no number in the report is absent from gbrain (automated check).
- Optional ADK eval harness (LLM-as-judge) for narrative quality of deep dives.

## Determinism gate
- Re-running `--build` on the same input yields byte-identical aggregates (seeded + cached).

## Commands
```bash
uv run pytest tests/unit tests/integration
```
