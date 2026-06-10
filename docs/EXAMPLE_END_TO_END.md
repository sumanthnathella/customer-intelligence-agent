# End-to-End Example: One Conversation → The Report

Traces a **single support conversation** through every stage of the pipeline, from
raw text to its appearance in the final report. Reference for how the modules connect
via the input contract and `gbrain`.

Pipeline (wired only in `build.py`; modules never import each other):

```
tagging/   → Table A (tagged transcripts)
datagen/   → Table B (operational dimensions)   [demo only — replace with real metrics]
analytics/ → gbrain update + snapshot
agent/     → report from gbrain
```

***

## 0. Raw input

A conversation reconstructed by `tagging/ingest.py` (TWCS tweets stitched into a thread):

```python
{
  "conversation_id": "c_001",
  "created_at": "2024-05-14T09:22:00Z",
  "text": "This is the THIRD time I'm contacting you. You charged my card twice "
          "for an order that never shipped. I want my money back or I'm closing my account."
}
```

***

## 1. Tagging → Table A  (`tagging/tag.py`)

`tag_conversations()` processes the row:

1. **Cache check** — `_cache_key("c_001", text)` = `sha256("c_001|<text[:1000]>|v1|v1")`.
   Looks for `data/tagger_cache/<hash>.json`. Miss → added to the batch.
   *(The cache is one JSON file per conversation under `data/tagger_cache/`.)*
2. **Prompt** — `_build_prompt` injects up to 50 taxonomy leaves + the conversation
   (truncated to 200 chars), asking for exactly one JSON tag per conversation.
3. **LLM call** — local Ollama with `format=_TAG_SCHEMA` (constrained JSON decoding),
   or an API model via LiteLLM. `temperature=0` for determinism.
4. **Repair** — `_normalise_sentiment` maps free-form tone to the enum; `_snap_l5`
   snaps a truncated / mis-prefixed id onto a real taxonomy leaf.
5. **Cache + collect** — the repaired tag is written back to
   `data/tagger_cache/<hash>.json` and appended to the output frame.

> **`repeat_contact` note:** set from *stated recurrence inside the single
> conversation* ("this is the THIRD time"), not a cross-conversation join. It is a
> self-reported-recurrence signal, not a customer-history lookup.

**Output row (Table A):**

```python
{
  "conversation_id": "c_001",
  "l5_id": "post_sale__billing__overcharge__duplicate_charge",
  "sentiment": "very_neg",
  "churn_intent": 1,      # "closing my account"
  "financial_harm": 1,    # "charged my card twice" / "want my money back"
  "safety_legal": 0,
  "repeat_contact": 1,    # "THIRD time I'm contacting you"
  "unresolved": 1,        # "order that never shipped"
  "confidence": 0.92
}
```

Written to `data/tagged_build.parquet`.

***

## 2. Datagen → Table B  (`datagen/`)  *(demo only)*

The demo has no real operational system, so `datagen/` synthesises operational
dimensions **after tagging** using a documented latent driver model
`P(dim | l5, signals)`. Drivers are *planted* and therefore *recoverable* — the
pipeline doubles as a validation harness (`tests/test_driver_recovery.py`).

**Operational row (Table B) for `c_001`:**

```python
{
  "conversation_id": "c_001",
  "product_category": "electronics",
  "vendor": "acme_marketplace",
  "fulfillment_type": "3p_marketplace",
  "carrier": "regional_x",
  "region": "US-NE",
  "payment_method": "credit_card",
  "order_total": 249.00
}
```

Written to `data/ops_build.parquet`. **In production this whole stage is replaced by
your real operational metrics** joined on `conversation_id`.

***

## 3. Analytics → gbrain  (`analytics/build_brain.py`)

`run()` loads both parquets, validates the contract, then runs each step below.
Throughout, we follow `c_001` *and its L5* `post_sale__billing__overcharge__duplicate_charge`.
Assume the build window has **10,000** conversations and this L5 has **1,240** of
them (`p(L5) = 1,240 / 10,000 = 12.4%`).

**1. Severity** (`severity.py`) — deterministic rubric, no LLM. For `c_001`:

```
base(very_neg)        = 3
+ churn_intent (1)    = +1
+ financial_harm (1)  = +1
+ safety_legal (0)    = +0
+ (repeat_contact OR unresolved) = +1
──────────────────────────────────
raw = 6  →  clip(1, 5)  →  severity = 5.0
```

> **Why an OR gate?** `repeat_contact` and `unresolved` both indicate the same
> underlying signal — recurrence / non-resolution. Giving `+1` for each
> independently would make most angry repeat contacts hit the ceiling of 5,
> destroying the scale's ability to distinguish degrees. The OR gate caps the
> recurrence bonus at **one point total**, regardless of how many recurrence
> flags are set.
>
> **Why clip to 5?** The rubric is a **bounded scale 1–5**. A "perfect storm"
> like `c_001` hits every signal, so the raw sum is 6. The ceiling clamp means
> it maxes out at 5.0, leaving headroom for less-severe conversations (e.g.
> `neg` + `financial_harm` only = 3.0). Averaged over all 1,240 contacts in this
> L5, `severity_avg = 4.30` — most contacts don't hit every signal at once.

---

**2. Weekly metrics + z-scores** (`metrics.py`) — group by `(l5_id, ISO week)` on
the real `created_at`. For this L5 the latest week has **180** contacts. The
**trailing 8-week baseline** is the historical weekly volume for this same L5
(the number of contacts it received in each of the previous 8 weeks):

```
previous 8 weeks:  [125, 132, 141, 138, 150, 145, 148, 155]
baseline_mean      = 139.5   contacts/week (average)
baseline_std       = 14.8    contacts/week (standard deviation — how much the
                           weekly counts typically swing up or down)
```

> **What is std?** Standard deviation. If the mean is 139.5 and std is 14.8, then
> most weeks fall within roughly `139.5 ± 14.8` (about 125–155 contacts). A week
> of 180 is well outside that normal range.

The **z-score** measures how many "standard deviations" the latest week is above
(or below) the historical average:

```
z = (latest - baseline_mean) / baseline_std
  = (180 - 139.5) / 14.8
  = 2.74        # ≥ 2.0 spike threshold → flagged as a recent spike
```

> **What does z=2.74 mean?** The latest week (180 contacts) is 2.74 standard
> deviations above the normal weekly average. In a normal distribution, that
> happens less than 1% of the time by chance — so this is a genuine spike, not
> random noise.

---

**3. Egregiousness** (`egregiousness.py`) — computed **once per analytics run**
after severity and z-scores are ready. It ranks **this L5 against every other L5**
in the window on four dimensions (volume, severity, z-spike, order-value),
normalizes each to a percentile in [0,1], then blends:

```
                    rank vs all L5s   weight
volume              96th %ile        × 0.35 = 0.336
severity            89th %ile        × 0.25 = 0.2225
spike (z)           78th %ile        × 0.25 = 0.195
order_value         80th %ile        × 0.15 = 0.120
                                      egregiousness = 0.8735
```

> **What is "severity" here?** It is `severity_avg` — the mean severity of **all**
> contacts for this L5 across the **entire analysis window** (not just the latest
> week). For this L5, `severity_avg = 4.30` (from step 1). That 4.30 is 89th
> percentile vs all other L5s, so this pain point has worse-than-average severity.
>
> **What is "volume"?** `total_volume` = total contacts for this L5 across the
> entire window (1,240). The 96th percentile means very few pain points have more
> total contacts.
>
> **What is "spike (z)"?** The z-score from step 2 (`z = 2.74`). The 78th
> percentile means some other L5s spiked even harder this week.
>
> This is a **relative** score — `0.8735` means this pain point is more
> egregious than ~87% of all pain points in the window.
>
> (If no monetary column exists, the 0.15 value weight redistributes equally to
> the other three.)

---

**4. Driver analysis** (`drivers.py`) — join Table A ⋈ Table B, then for each
`(l5_id, dimension=value)` compute over-indexing **lift** = `p(L5 | value) / p(L5)`.

Contingency table for `vendor=acme_marketplace`:

```
                  L5=duplicate_charge  All other L5s   Total
                  ───────────────────  ──────────────  ─────
acme_marketplace        470                 1,335     1,805
Other vendors           770                 7,425     8,195
                  ───────────────────  ──────────────  ─────
Total                 1,240                 8,760    10,000
```

```
p(L5)               = 1,240 / 10,000 = 0.124     # baseline
p(L5 | acme)        =   470 /  1,805 = 0.260     # conditional
lift                = 0.260 / 0.124  = 2.1×      # over-indexing
support             = 470  (≥ 30 min)             ✓
two-prop z-test → p = 0.0003                    (significant after BH-FDR)
```

> **Step-by-step in plain English:**
>
> Imagine you walk into a room with 10,000 complaints on the wall.
> You pick one at random. What are the odds it is about duplicate charges?
> **12.4%** — that's `p(L5)`. This is your baseline; it tells you how common
> this issue is overall.
>
> Now imagine a *different* room — this one has only the 1,805 complaints that
> involve `acme_marketplace`. You pick one at random from *this* room.
> What are the odds it is about duplicate charges?
> **26.0%** — that's `p(L5 | acme)`. The "`| acme`" means "given that this
> complaint involves acme_marketplace."
>
> **Lift** is the ratio of those two probabilities:
> `0.260 / 0.124 = 2.1`. It answers: "How much more likely is duplicate charging
> among acme complaints compared to complaints overall?" Answer: **2.1× more**.
>
> **Support** = 470. This just means "we have 470 real complaints to back this up."
> We require at least 30 (`DRIVER_MIN_SUPPORT`) so we do not trust a pattern
> built on 3 or 4 contacts.
>
> **What are we actually testing?** The pipeline runs a **two-proportion z-test**.
> It compares two groups:
>
> | Group | Total contacts | Contacts about duplicate charges | Rate |
> |-------|---------------|----------------------------------|------|
> | **acme_marketplace** | 1,805 | 470 | **26.0%** |
> | **All other vendors** | 8,195 | 770 | **9.4%** |
>
> The gap is **16.6 percentage points** (26.0% − 9.4%).
>
> **The null hypothesis:** "acme has no special relationship to duplicate charges.
> The 26.0% and 9.4% rates are just random sampling noise — if we had a bigger
> dataset, both groups would converge to the same true rate."
>
> **The test:** "Assuming the null hypothesis is true, what is the probability
> of observing a gap of 16.6 points or larger purely by chance?"
>
> **The answer: p = 0.0003 (0.03%).**
>
> In other words: if acme truly had nothing to do with duplicate charges, and
> we ran this experiment thousands of times, we would see a gap this extreme
> only 3 times out of 10,000. That is strong evidence the gap is **real**, not a fluke.
>
> **Benjamini–Hochberg (BH-FDR):** Why do we need this?
>
> The pipeline tests **thousands** of `(dimension, value)` pairs. Every vendor,
> every fulfillment type, every region, every payment method, every carrier —
> for every L5. If we declared anything with `p < 0.05` as "significant," we'd
> be flooded with false positives. Here's why:
>
> | Tests run | Random "significant" at p<0.05 (false positives) |
> |-----------|---------------------------------------------------|
> | 100       | ~5 by pure chance |
> | 1,000     | ~50 by pure chance |
> | 5,000     | ~250 by pure chance |
>
> **The problem:** without correction, ~250 of your "significant drivers" would
> be total noise. You'd waste engineering time chasing ghosts.
>
> **What BH-FDR does:** It re-adjusts every p-value based on how many tests
> you ran. The more tests, the stricter the bar becomes. After adjustment, we
> only keep a driver if `corrected_p < 0.05`. This guarantees that, among all
> the drivers you call "significant," no more than 5% are expected to be false.
>
> For `acme_marketplace`, the raw p-value was 0.0003. After BH-FDR correction
> across thousands of tests, it gets adjusted upward (made more conservative) but
> still stays well below 0.05. It survives the stricter bar. A borderline case
> with a raw p-value of 0.04 would not — the correction would push it above 0.05
> and it would be discarded as likely noise.

Same for `fulfillment_type=3p_marketplace` → lift `1.7×`, support `508`, `p=0.0011`.

---

**5. Sub-themes** (`subthemes.py`) — runs **once per analytics run**, after
severity, z-scores, and drivers are computed but before everything is written to
gbrain. It turns "1,240 duplicate-charge contacts" into the specific sub-issues
customers actually mention, mined directly from transcript text.

**How it works (no LLM, fully deterministic):**

1. **Build a vocabulary** across the entire corpus. Extract 1-word and 2-word
   phrases, strip English stop-words ("the", "and") plus domain stop-words
   ("please", "sorry", "DM us", "thanks", "team", "reach out"). Only keep
   phrases that appear in at least 5 contacts but fewer than 40% of all contacts.

2. **Fit TF-IDF once on the whole corpus.** TF-IDF measures how *important* a
   phrase is in a document relative to the whole collection. A phrase that appears
   frequently in one pain point but rarely elsewhere gets a high score.

3. **For each L5, compute distinctiveness:**
   ```
   distinctiveness = mean TF-IDF inside this L5  −  mean TF-IDF in all other L5s
   ```
   A high positive number means: "This phrase is unusually common *here* and
   unusually rare *everywhere else*."

4. **Keep the top non-overlapping phrases.** If "charged my card" and "charged my
   card twice" are both candidates, keep only the longer one. Skip phrases that
   appear in fewer than 4 contacts.

5. **Attach a representative quote.** For each kept phrase, find the
   highest-severity contact that mentions it. Compress that contact to its most
   relevant sentence. That becomes the quote.

**For this L5:**

```
• "double charged on a single order"  — 312 contacts (25%)
     quote: "charged my card twice for an order that never shipped"
     why: highest-distinctiveness phrase; appears in 312 contacts but is
          rare in every other L5.
• "refund not received"             — 204 contacts (16%)
     quote: "want my money back or I'm closing my account"
     why: second-highest distinctiveness; a separate issue within the same L5.
```

> **Why this matters:** The L5 tells you the *bucket* ("duplicate charge").
> Sub-themes tell you *what is actually happening inside it* — battery issues vs
> cracked screens, or in this case double-charging vs missing refunds. You get
> the specific words customers use and a real quote to investigate with.
>
> **Future upgrade — LLM-based theme canonicalization (v2):**
> If TF-IDF themes feel too fragmented (e.g. "double charged" and "overbilled"
> treated as separate), an LLM pass can canonicalize them. Modern LLMs with
> large context windows can ingest batched transcripts directly, extract themes
> in natural language, and collapse near-duplicates into canonical labels.
> No embeddings or clustering required — the LLM reads the raw text, proposes
> themes, and maps each contact to the closest theme. This is kept as a
> future enhancement; TF-IDF is the v1 baseline because it is deterministic,
> fast, and requires no LLM calls at scale.

---

**6. Facet composition** (`facet_breakdown`) — dominant, over-indexing groups per
dimension (share = % of *this L5's* contacts; lift from step 4):

```
vendor           = acme_marketplace  →  470/1,240 = 38%,  2.1× (over-indexed)
fulfillment_type = 3p_marketplace    →  508/1,240 = 41%,  1.7× (over-indexed)
```

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

- **Share alone is misleading.** If acme is 90% of your business, seeing 38% acme
  in this pain point is not interesting — it is just proportionate. The lift
  tells you this is *not* proportionate; it is a genuine concentration.

- **Lift alone is misleading.** A lift of 10× with only 3 contacts is a statistical
  ghost. The share tells you this is a *real* group (470 contacts, not 3).

**How it is built:** The `facet_breakdown` function takes the driver edges from
step 4, groups them by dimension (vendor, fulfillment_type, etc.), and for each
keeps the top 3 values by share — but only if they represent at least 5% of the
L5's contacts. Each kept value carries its lift and significance flag forward
from the driver analysis.

> **Why this helps:** This answers "*Who* is getting hit, and *where* should I
> focus?" It turns a raw number (1,240 duplicate-charge complaints) into a
> **profile** of the pain point. You see that more than 1 in 3 complaints come
> from one vendor, and that vendor is 2.1× over-indexed — a narrow, actionable
> surface area. Without this, you would see "1,240 duplicate charges" and have no
> idea where to start.

---

**7. Systemic bridges** (`systemic.py`) — after driver analysis, the pipeline
rolls up every `(dimension, value)` pair and asks: "How many *different* pain
points does this value drive?"

```
acme_marketplace drives:
  • duplicate_charge          (lift 2.1×, verified)
  • refund_delayed            (lift 1.8×, verified)
  • wrong_item_shipped        (lift 1.5×, verified)
  • billing_discrepancy       (lift 1.4×, verified)
  ─────────────────────────────────────────────────
  = 4 different L5s  →  flagged as BRIDGE (≥ 3 threshold)
```

**What is a bridge vs. a singleton?**

| Type | Definition | Example | What to do |
|------|-----------|---------|-----------|
| **Bridge** | Significant driver for ≥ 3 L5s | `vendor=acme_marketplace` drives 4 different billing/shipping issues | Fix the vendor relationship once → relief across 4 pain points |
| **Singleton** | Significant driver for exactly 1 L5 | `payment_method=crypto` drives only `payment_declined` | Fix the specific payment flow |

**Why bridges are surfaced at the top:** They are the highest-leverage fixes.
A singleton tells you "this specific thing is broken." A bridge tells you
"this operational lever is broken, and it's making *many* things go wrong."
Fix the lever once, you relieve multiple pain points.

The **systemic_score** = `(#L5s reached × mean lift)` normalized to [0,1].
A bridge affecting 4 L5s at 1.8× average lift scores higher than one affecting
3 L5s at 1.2×. Bridges are sorted by this score so the most impactful levers
surface first in the report.

---

**8. Importance tier + trend** (`systemic.py`) — **not a calculation, just a
lookup table** that buckets already-computed numbers into human-readable labels.

```
Egregiousness → Importance tier (warm-start curation tag)
  ≥ 0.85  →  very_high
  ≥ 0.60  →  high
  ≥ 0.35  →  fair
  else    →  low

Z-score → Trend (direction label)
  ≥  2.0  →  rising   (recent spike)
  ≤ -2.0  →  falling  (recent drop)
  else    →  stable   (within normal range)
```

For this L5: `egregiousness = 0.8735 ≥ 0.85` → `very_high`; `z = 2.74 ≥ 2.0` →
`rising`. These tags are **carried across runs** as warm-start curation, so the
system remembers that this pain point was escalating rather than rebuilding the
ranking from scratch each time.

---

**9. Verification cache** — after driver analysis, the pipeline takes the top 3
drivers per L5 and **re-checks them against the raw data** before writing them
to the report. This is a defense against any bug or edge case that could
produce a bogus driver.

**What is being validated:**

```
Claim:  "vendor=acme_marketplace over-indexes for duplicate_charge"
Evidence: lift=2.1×, support=470, p=0.0003
Check:    lift > 1  AND  support ≥ 30  AND  significant=True
Result:   PASS → verdict = "verified"
```

**What if it fails?** If a driver somehow had `lift = 0.8` (under-indexes),
`support = 5` (too few contacts), or `significant = False` (not statistically
sound), the check fails → verdict = "unsupported". That driver is filtered out
and never surfaces in the report.

**What gets persisted in `gbrain`:**

```
verification node (SHA1 of claim):
  - claim:    "vendor=acme_marketplace over-indexes for duplicate_charge"
  - verdict:  "verified"
  - detail:   {lift: 2.1, support: 470, p_value: 0.0003, ...}
  - evidence: ["dim:vendor:acme_marketplace"]
  - run_id:   "run_2024_05_14_001"
  - edges:    verifies → pain_point node
              cites    → dimension node
```

This creates an **audit trail**. If a stakeholder asks "Why did you say acme
is a driver?", you point to the verification record. The claim, evidence, and
verdict are all stored and traceable — not just an assertion in a report.

---

**10. Exemplars** — the highest-severity transcripts for this L5 (including `c_001`,
severity 5.0) are **BM25-compressed** (`shared/text.py`) to the 2 sentences most
relevant to the L5 definition, then stored.

Everything is **upserted into `gbrain`** (SQLite at `gbrain/store/`) as durable,
L5-centric entities:

- `pain_point` node `post_sale__billing__overcharge__duplicate_charge` carrying
  volume, severity, z-score, egregiousness, `importance`, `trend`, `facets`,
  `subthemes`, `top_drivers`, `last_seen`.
- `affects` edges → each driver dimension value (with `lift`, `support`, `p`, history).
- `dimension` nodes (systemic rollup); `exemplar` nodes; `verification` records.
- The **run is registered and snapshotted**, so movement across runs is real.

> **Production swap:** `gbrain` is not tied to SQLite. `GBrainStore` is an
> interface class (`gbrain/store.py`) with methods like `upsert_node`, `add_edge`,
> `query`, `traverse`. SQLite is the v1 implementation because it is zero-ops,
> single-file, and Python-native. To scale to millions of conversations, swap in a
> BigQuery or Neo4j backend by implementing the same interface — only the store
> class changes; `analytics/`, `agent/`, and `retrieval.py` are untouched.

`run()` returns a summary dict (`run_id`, `period`, counts, `top_egregious`,
`top_bridges`, `snapshot` path) — passed forward to the report stage.

***

## 4. Agent → Report  (`agent/report.py`)

`render()` reads **only** `gbrain` (numbers never come from the LLM) via
`gbrain/retrieval.py`:

- `get_curated_set` — importance-tagged pain points with cross-run movement.
- `get_systemic_drivers(only_bridges=True)` — the highest-leverage levers.
- `get_zscore_spikes` — recent significant spikes.
- `get_ops_hotspots` — ops-first (dimension → top L5s) view.

It writes `reports/report_<run_id>.md` and `.json`. Our `c_001` surfaces inside its
L5's section:

```markdown
## Curated Pain Points

### 1. [VERY HIGH] post_sale ▸ billing ▸ overcharge ▸ duplicate_charge  ↑
- **ID:** `post_sale__billing__overcharge__duplicate_charge`  ·  _▲ ESCALATED_
- **Egregiousness:** 0.8735 (volume=1,240 (+180), severity=4.30, z=2.74)
  - *Why #1:* highest egregiousness in the window; egregiousness ≥ 0.85 → tier `very_high`.
  - *↑ rising:* z-score 2.74 ≥ 2.0 threshold → recent spike.
  - *▲ ESCALATED:* importance moved from `high` (last run) → `very_high` (this run).
- **What's inside (dominant groups):**
  - `vendor=acme_marketplace` — 38% of contacts, 2.1× vs baseline *(over-indexed)*
    - *Translation:* more than 1 in 3 complaints come from one vendor; they are
      2.1× more likely to produce duplicate charges than random chance.
  - `fulfillment_type=3p_marketplace` — 41% of contacts, 1.7× vs baseline *(over-indexed)*
    - *Translation:* marketplace fulfillment is over-represented in this issue.
- **Issue themes:**
  - **double charged on a single order** — 312 contacts (25%) — _"charged my card twice for an order that never shipped"_
    - *Translation:* the dominant sub-theme, mined from actual transcript text.
  - **refund not received** — 204 contacts (16%) — _"want my money back or I'm closing my account"_
    - *Translation:* second distinct issue within the same L5, with a representative quote.
- **Verified drivers (statistical):**
  - ✓ `vendor=acme_marketplace` — lift 2.1×, n=470, p=0.0003
  - ✓ `fulfillment_type=3p_marketplace` — lift 1.7×, n=508, p=0.0011
    - *Translation:* every claim is re-checked deterministically against the
      data store; ✓ means `lift > 1` and `support ≥ 30` both held.
```

If `vendor=acme_marketplace` also drives other L5s, it appears at the top under
**Systemic Operational Drivers (Bridges)** — "fix once, relieve many."

***

## How to run it

```bash
# Full demo (tag → datagen → analytics → report)
uv run python build.py --demo --sample 10000

# Re-render the report from current gbrain state (no recompute)
uv run python build.py --report-only

# Grow the brain with another batch
uv run python build.py --test-batch 1
```

Outputs:

- Tags: `data/tagged_build.parquet`  ·  cache: `data/tagger_cache/*.json`
- Ops:  `data/ops_build.parquet`
- Brain: `gbrain/store/` (+ `snapshots/`)
- Report: `reports/report_<run_id>.md` and `.json`

***

## Why this design

- **Producers are replaceable, the core is dataset-agnostic.** Swap `tagging/` for
  BYO tags and `datagen/` for real ops metrics; `analytics/` → `gbrain/` → `agent/`
  are untouched. The only contract is Table A ⋈ Table B (on `conversation_id`).
- **Three layers of LLM-output defense** (constrained decoding → normalise → snap)
  guarantee a clean taxonomy join.
- **Numbers live in `gbrain`, not the model.** The agent narrates; every cited driver
  is verified (✓/✗) against the store.
- **Durable, incremental memory.** Each run upserts and snapshots, so movement
  (new / escalated / de-escalated) and trends reflect real change over time.
