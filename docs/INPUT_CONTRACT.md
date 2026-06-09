# Input Contract (Bring Your Own Data)

The **dataset-agnostic core** (`analytics/` → `gbrain/` → `agent/`) depends only on this contract — not
on twcs, not on our tagger, not on synthetic dimensions. Conform to it and the agent works on *your* data.

The two demo producers are optional conveniences:
- `tagging/` — produces **Table A** if you don't already have tags.
- `datagen/` — produces **Table B** for the dimensionless twcs demo. **Replace it with your real operational metrics.**

## Table A — Tagged transcripts

One row per analyzed conversation.

| Column | Type | Required | Notes |
|--------|------|----------|-------|
| `conversation_id` | str | ✓ | join key |
| `created_at` | datetime | ✓ | drives weekly volume + z-score |
| `text` | str | ✓ | used for exemplars |
| `l5_id` | str | ✓ | leaf id from the taxonomy schema pack |
| `sentiment` | enum | ✓ | very_neg / neg / neutral / pos |
| `churn_intent` | 0/1 | ✓ | severity signal |
| `financial_harm` | 0/1 | ✓ | severity signal |
| `safety_legal` | 0/1 | ✓ | severity signal |
| `repeat_contact` | 0/1 | ✓ | severity signal |
| `unresolved` | 0/1 | ✓ | severity signal |
| `confidence` | float | optional | tagger confidence |

> Already have a severity *number* instead of signals? Provide `severity` (1–5) and the core will use it
> directly, skipping the rubric.

## Table B — Operational dimensions

One row per `conversation_id` (or per `order_id` joinable to it). **Columns are whatever operational
parameters you have** — the engine discovers drivers over all provided dimension columns automatically.

| Column | Type | Required | Notes |
|--------|------|----------|-------|
| `conversation_id` | str | ✓ | join key to Table A |
| `<dimension>` | categorical | ≥1 | e.g. `carrier`, `fulfillment_type`, `vehicle_type`, `region`, `service_level`, `customer_segment` |
| `<fact>` | numeric | optional | e.g. `order_total` (powers `order_value_at_risk`), `quantity` |

- Declare which columns are **dimensions** vs **facts** in `shared/config.py` (or auto-infer by dtype).
- Facts can be auto-bucketed to also serve as dimensions (e.g. `order_total` → size bins).
- No monetary fact? The egregiousness `value` weight redistributes automatically.

## The taxonomy schema pack

Tagging requires a frozen `taxonomy.json` (the L1–L5 schema pack). BYO users either reuse ours, induce
their own with `tagging/taxonomy.py`, or supply their own taxonomy file. See [TAXONOMY.md](TAXONOMY.md).

## Validation

`shared/contract.py` validates both tables (required columns, types, join integrity, ≥1 dimension) before
the core runs — failing loudly with a clear message so BYO users get fast feedback.
