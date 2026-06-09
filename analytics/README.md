# `analytics/` — Dataset-Agnostic Core

Turns **tagged transcripts ⋈ operational dimensions** (the [input contract](../docs/INPUT_CONTRACT.md))
into severity, metrics, egregiousness, and **operational drivers**, then writes the L5-centric brain.
Knows nothing about twcs, Ollama, or synthetic data — it works on **any** conforming dataset.

Full spec: [`../docs/ANALYTICS.md`](../docs/ANALYTICS.md).

## Modules (to implement)
| File | Role |
|------|------|
| `severity.py` | severity signals → deterministic 1–5 score (rubric in `shared/config.py`) |
| `metrics.py` | weekly volume + recent-week z-score (real `created_at`) |
| `egregiousness.py` | percentile-blended ranking (`0.35/0.25/0.25/0.15`) |
| `drivers.py` | over-indexing (lift) + significance (z-test + Benjamini–Hochberg), per L5 |
| `build_brain.py` | upsert L5 entities + append `period_metric` + update `affects` driver edges |

Imports `shared/` and `gbrain/` only — never `tagging/` or `datagen/`.
