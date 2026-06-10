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
raw    = base
       + 1*churn_intent
       + 1*financial_harm
       + 2*safety_legal
       + 1*(repeat_contact OR unresolved)   # at most +1 from these two combined
severity = clamp(raw, 1, 5)
```

**How the OR gate works:** `repeat_contact` and `unresolved` are checked with a
bitwise OR (`|`). If either flag is 1 (or both are 1), the OR result is > 0,
which converts to integer `1`. This guarantees at most `+1` from these two
signals combined, not `+2`.

**Example trace:**
```
sentiment = "very_neg"          → base = 3
churn_intent = 1                 → +1
financial_harm = 1               → +1
safety_legal = 0                 → +0
repeat_contact = 1, unresolved = 1
  (1 | 1) > 0 → True → 1         → +1   # NOT +2; OR-gate caps at +1
raw = 6  →  clamp(1, 5)  →  severity = 5.0
```

Auditable (you can see why a transcript scored 5), consistent across millions
of rows, and adjustable.

## 2. Volume & recent-week z-score (`metrics.py`)

- Bucket tagged transcripts into ISO weeks from real `created_at`.
- For each L5: weekly volume series.
- **z-score** of the latest week vs trailing baseline (default **8 weeks, min 3**):
  `z = (v_latest − mean(baseline)) / std(baseline)`; std floored to avoid divide-by-zero.

**What the baseline and std mean:** The baseline is the **average number of
contacts per week** for this L5 over the trailing 8 weeks. The std (standard
deviation) measures how much those weekly counts **typically bounce around**.
Example: if an L5 averages 140 contacts/week with std = 15, then 170 contacts
this week is `z = (170 − 140) / 15 = 2.0` — a genuine spike, not normal
fluctuation.

**Insufficient history:** If an L5 has fewer than 3 weeks of data, z-score
returns NaN (not calculated) rather than a misleading value. This is surfaced
in the report as "insufficient history" rather than silently omitted.

- A high positive z flags a **recent spike** — surfaced separately from raw volume.

## 3. Egregiousness (`egregiousness.py`)

**When it runs:** Egregiousness is computed **once per analytics run**, after
severity and z-scores are calculated but before driver analysis. It is a
**ranking score**, not a per-transcript metric.

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

**What the severity percentile means:** Each L5's `severity_avg` is computed
over the **full analysis window** (all weeks), not just the latest week. The
percentile rank then answers: "How severe is this pain point compared to *all
other* pain points?" A `pct_severity = 0.95` means this L5 is in the top 5%
most severe across the entire window.

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

**Step-by-step example:**

For `vendor = acme_marketplace` and L5 `duplicate_charge`:

| | Has duplicate_charge | Does NOT have duplicate_charge | Total |
|---|---|---|---|
| **acme** | 470 | 1,340 | 1,810 |
| **NOT acme** | 770 | 3,420 | 4,190 |
| **Total** | 1,240 | 4,760 | 6,000 |

1. **p(L5 | acme)** = 470 / 1,810 = **26.0%** (acme contacts that are duplicate-charge)
2. **p(L5 | not acme)** = 770 / 4,190 = **18.4%** (non-acme contacts that are duplicate-charge)
3. **lift** = 26.0% / 18.4% = **1.41×** (acme contacts are 1.41× more likely to be duplicate-charge)

A lift of 1.0 means "no effect." A lift of 1.41× means acme is genuinely
over-represented in this pain point.

### Significance & guards (avoid noise)

**Two-proportion z-test:** The pipeline asks: "Could this 1.41× pattern be a
fluke?" It compares the proportion of duplicate-charge contacts among acme
(26.0%) vs. among non-acme (18.4%) using a z-test. The p-value measures how
likely the difference is due to random chance alone.

- **p = 0.0003** → extremely unlikely to be a fluke → **keep it**
- **p = 0.3** → could easily happen by chance → **discard it**

**Benjamini–Hochberg FDR correction:** The pipeline tests thousands of
`(dimension, value)` pairs. By chance alone, some will look significant. BH-FDR
asks: "If I call everything with `p < 0.05` significant, how many of those will
be false alarms?" It ranks all p-values and applies a stricter cutoff so that
no more than 5% of the kept drivers are expected to be false. Only drivers
that survive this correction are surfaced.

**Support guard:** `support ≥ 30` (configurable). Patterns based on 3 contacts
are statistical ghosts, not actionable drivers.

### Output per L5
A ranked list of drivers, e.g.:
> `post_sale ▸ delivery ▸ late_delivery ▸ carrier_delay ▸ stuck_in_transit_7d+`
> drivers: `vehicle_type=drone` (lift 3.1×, n=412, p<0.001), `region=Northeast` (2.2×),
> `service_level=2-day` (1.8×)

### Optional one interaction
When no single dimension explains the pain, test the top **2-way** combo (e.g. `vehicle_type × season`)
with the same support/significance guards. Capped at one to keep cells large and the report readable.

## 5. Sub-themes (`subthemes.py`)

**When it runs:** Once per analytics run, after severity, z-scores, and drivers
are computed but before everything is written to gbrain. It turns "1,240
duplicate-charge contacts" into the specific sub-issues customers actually
mention, mined directly from transcript text.

**How it works (no LLM, fully deterministic):**

1. **Build a vocabulary** across the entire corpus. Extract 1-word and 2-word
   phrases, strip English stop-words plus domain stop-words ("please", "sorry",
   "DM us", "thanks"). Only keep phrases that appear in at least 5 contacts but
   fewer than 40% of all contacts.

2. **Fit TF-IDF once on the whole corpus.** TF-IDF measures how *important* a
   phrase is in a document relative to the whole collection. A phrase that appears
   frequently in one pain point but rarely elsewhere gets a high score.

3. **For each L5, compute distinctiveness:**
   ```
   distinctiveness = mean TF-IDF inside this L5  −  mean TF-IDF in all other L5s
   ```
   A high positive number means: "This phrase is unusually common *here* and
   unusually rare *everywhere else*."

4. **Keep the top non-overlapping phrases.** If "charged my card" and
   "charged my card twice" are both candidates, keep only the longer one. Skip
   phrases that appear in fewer than 4 contacts.

5. **Attach a representative quote.** For each kept phrase, find the
   highest-severity contact that mentions it. Compress that contact to its most
   relevant sentence (BM25). That becomes the quote.

> **Future upgrade — LLM-based theme canonicalization (v2):** If TF-IDF themes
> feel too fragmented (e.g. "double charged" and "overbilled" treated as
> separate), an LLM pass can ingest batched transcripts, extract themes in
> natural language, and collapse near-duplicates into canonical labels. No
> embeddings or clustering required. TF-IDF is the v1 baseline because it is
> deterministic, fast, and requires no LLM calls at scale.

## 6. Facet composition (`facet_breakdown`)

Dominant, over-indexing groups per dimension (share = % of *this L5's* contacts;
lift from driver analysis).

**How to read one line:**

```
                    ┌─ share = 470 / 1,240 = 38%
                    │   (of all duplicate-charge contacts, 38% are acme)
vendor = acme_marketplace ─┤
                    └─ lift = 2.1×
                        (acme contacts are 2.1× more likely to be
                         duplicate-charge than random chance)
```

**Two numbers, two different questions:**

| Number | Question it answers | Example |
|--------|---------------------|---------|
| **Share** | "What % of *this pain point's* contacts have this property?" | 38% of duplicate-charge complaints are acme |
| **Lift** | "Is that share unusually high compared to the whole population?" | Yes — 2.1× more than random |

**Why both matter:**

- **Share alone is misleading.** If acme is 90% of your business, seeing 38%
  acme in this pain point is not interesting — it is just proportionate. The
  lift tells you this is *not* proportionate; it is a genuine concentration.

- **Lift alone is misleading.** A lift of 10× with only 3 contacts is a
  statistical ghost. The share tells you this is a *real* group (470 contacts,
  not 3).

**How it is built:** The `facet_breakdown` function takes the driver edges from
step 4, groups them by dimension, and for each keeps the top 3 values by share —
but only if they represent at least 5% of the L5's contacts. Each kept value
carries its lift and significance flag forward from the driver analysis.

> **Why this helps:** This answers "*Who* is getting hit, and *where* should I
> focus?" It turns a raw number into a **profile** of the pain point. Without
> this breakdown, you would see "1,240 duplicate charges" and have no idea where
> to start.

## 7. Two views, one store

Both come from the same per-L5 `affects` driver edges in gbrain — no separate computation:
- **Pain-first** (primary deep dives): group by L5 → its top drivers.
- **Ops-first** ("Operations Hotspots"): group `affects` edges by dimension value → its top L5s.
  The view an ops owner reads: *"drone deliveries → these 3 complaints."*

## 8. Persisted by L5 — the full dimension cube, not just drivers

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

## 9. Validation (because drivers are recoverable)

In the demo, `datagen/` plants drivers via a documented model; the engine recovers them statistically
without seeing that model. A unit test asserts **recovered lift ≈ planted lift** within tolerance — which
also validates the methodology you'd run on real data. See [EVALUATION.md](EVALUATION.md).
