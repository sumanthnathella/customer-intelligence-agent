# Analytics Engine

The dataset-agnostic core (`analytics/`) that turns **tagged transcripts ⋈ operational dimensions**
into the metrics, severity, egregiousness ranking, and **operational drivers** that make the report
insightful. It reads the [input contract](INPUT_CONTRACT.md) and writes the [L5-centric brain](GBRAIN.md).

> **Rule:** the LLM never produces a number. Everything here is deterministic and reproducible; the
> agent only *narrates* these outputs.

## 1. Severity — signals → deterministic score

Severity is **not** picked by the LLM. The tagger extracts structured **signals**; `severity.py`
computes the score with a fixed, tunable rubric (re-tunable without re-tagging).

**Signals extracted per transcript** (by `tagging/tag.py`, JSON-constrained):

| Signal | Type | Example cues |
|--------|------|--------------|
| `sentiment` | very_neg / neg / neutral / pos | overall tone |
| `churn_intent` | 0/1 | "cancelling", "switching", "never again" |
| `financial_harm` | 0/1 | "charged twice", "want my money", "overcharged" |
| `safety_legal` | 0/1 | injury, hazard, fraud, "my lawyer" |
| `repeat_contact` | 0/1 | "3rd time", "still waiting", "for days" |
| `unresolved` | 0/1 | "no response", "still broken" |

**Rubric** (defaults; in `shared/config.py`):
```
base = {very_neg:3, neg:2, neutral:1, pos:1}[sentiment]
severity = clamp(base
                 + 1*churn_intent
                 + 1*financial_harm
                 + 2*safety_legal
                 + 1*(repeat_contact OR unresolved),
                 1, 5)
```
Auditable (you can see why a transcript scored 5), consistent across millions of rows, and adjustable.

## 2. Volume & recent-week z-score (`metrics.py`)

- Bucket tagged transcripts into ISO weeks from real `created_at`.
- For each L5: weekly volume series.
- **z-score** of the latest week vs trailing baseline (default **8 weeks, min 3**):
  `z = (v_latest − mean(baseline)) / std(baseline)`; std floored to avoid divide-by-zero.
- A high positive z flags a **recent spike** — surfaced separately from raw volume.

## 3. Egregiousness (`egregiousness.py`)

Per L5 over the window, four raw measures → **percentile rank across all L5s** (robust) → weighted blend:

```
egregiousness(L5) = 0.35·pct(volume)
                  + 0.25·pct(mean_severity)
                  + 0.25·pct(zscore_spike)
                  + 0.15·pct(order_value_at_risk)
```
- `order_value_at_risk` = Σ order_total of the L5's transcripts (or severity-weighted). If a dataset has
  no monetary dimension, `w4` redistributes to the others.
- The report shows the **component breakdown**, so a ranking is never an opaque number.
- Weights live in `shared/config.py`.

## 4. Driver analysis — tying pain to operational parameters (the core insight)

This is what makes the report more than a group-by. For each pain point we find the **operational
dimension values that over-index** — i.e., that *drive* the pain.

### Over-indexing (lift)
For pain point `P` (an L5) and dimension value `d`:
```
lift(P,d)    = P(d | P) / P(d)            # >1 ⇒ d over-represented among P
support(P,d) = count(P ∧ d)
excess(P,d)  = count(P ∧ d) − expected    # absolute extra volume attributable to d
```

### Significance & guards (avoid noise)
- Keep a driver only if `support ≥ MIN_SUPPORT` (default 30) **and** a two-proportion z-test on
  `P(P|d)` vs `P(P|¬d)` is significant.
- **Benjamini–Hochberg** FDR correction across all (P,d) cells tested (hundreds), so we don't surface
  false positives by chance.
- Optionally severity-weighted: compare mean severity in cell vs overall.

### Output per L5
A ranked list of drivers, e.g.:
> `post_sale ▸ delivery ▸ late_delivery ▸ carrier_delay ▸ stuck_in_transit_7d+`
> drivers: `vehicle_type=drone` (lift 3.1×, n=412, p<0.001), `region=Northeast` (2.2×),
> `service_level=2-day` (1.8×)

### Optional one interaction
When no single dimension explains the pain, test the top **2-way** combo (e.g. `vehicle_type × season`)
with the same support/significance guards. Capped at one to keep cells large and the report readable.

## 5. Two views, one store

Both come from the same per-L5 `affects` driver edges in gbrain — no separate computation:
- **Pain-first** (primary deep dives): group by L5 → its top drivers.
- **Ops-first** ("Operations Hotspots"): group `affects` edges by dimension value → its top L5s.
  The view an ops owner reads: *"drone deliveries → these 3 complaints."*

## 6. Persisted by L5 — the full dimension cube, not just drivers

The whole **(L5 × dimension-value) breakdown** is materialized into the brain as `affects` edges —
**not only the significant drivers**. Each edge carries both the **raw aggregation** and the **analytics**:
```
{ dimension, value,
  support,        # raw count  ("fulfillment_type=BOPIS showed up 100 times for this L5")
  share,          # support / L5 volume
  lift, p_value,  # over-indexing + significance
  significant,    # bool flag → this cell is a driver
  excess,         # extra volume vs expected
  period,
  history: [ {period, support, share, lift}, ... ]   # rolling time-series for trends
}
```
- **Heatmap**: an L5's full set of `affects` edges = its dimension heatmap (counts/share for every value);
  `significant` flags which cells are real drivers. The agent renders this directly.
- **Trends**: `history` lets us trend a cell over time — e.g. *"drone lift for late_delivery rose 3.1× →
  3.8×"* or *"BOPIS share of returns climbed 12% → 19%"* — without recomputation.
- To keep it bounded, persist cells meeting a small **min-volume** (top-K per dimension + all significant);
  tiny long-tail cells roll into an `other` bucket. Always updates the *same* L5 entity, so trends stay
  consistent into the future. See [GBRAIN.md](GBRAIN.md).

## 7. Validation (because drivers are recoverable)

In the demo, `datagen/` plants drivers via a documented model; the engine recovers them statistically
without seeing that model. A unit test asserts **recovered lift ≈ planted lift** within tolerance — which
also validates the methodology you'd run on real data. See [EVALUATION.md](EVALUATION.md).
