# Decisions (ADRs)

Chronological record of design decisions. Locked = agreed; Open = needs sign-off before/at review.

## Locked

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Dataset-agnostic core** (`analytics/`+`gbrain/`+`agent/`) + **replaceable producers** (`tagging/`, `datagen/`), connected via the input contract + gbrain. | Open-source: BYO transcripts + real operational metrics; producers are one-time and swappable. |
| D2 | **Data** = local `twcs.csv` (~2.8M). ~1M build, ~1.8M test reserve, split by conversation. | Real transcripts at scale; no leakage; reserve enables ongoing growth. |
| D3 | **Demo operational dimensions are synthetic**, seeded + driver-conditioned (see D13/D16). | twcs lacks order data; enables dimension cuts; reproducible; clearly labeled simulated. |
| D4 | **Data-driven L1–L5 taxonomy**, unique paths, **≤100 L5 leaves**, L1 fixed to pre/sale/post. | Matches the data; bounded; report always lifecycle-organized. |
| D5 | **Local Ollama `qwen2.5:7b`** for all bulk tagging; `nomic-embed-text` for clustering. | Free, unlimited, fast on M4 24GB; avoids rate caps. |
| D6 | **OpenRouter `nemotron-3-ultra:free`** for the agent only; local `gemma4:31b` fallback. | Frontier reasoning where it matters; resilient + free. |
| D7 | **Single ReAct agent** (not multi-agent). | Few LLM calls, simple, debuggable; analytics already deterministic. |
| D8 | **gbrain = graph + SQLite + JSON**, taxonomy as schema pack, auto-link on write. | Factual + semantic retrieval; portable; accumulates over periods. |
| D9 | **Real `created_at`** drives weekly volume + z-score. | Faithful spike detection. |
| D10 | **`.claude/` harness** (skills/agents/rules/hooks), ECC-style. | Explicit, reusable, shareable agent behavior. |
| D11 | **Numbers from aggregates/gbrain only**; LLM narrates, never computes. | Trustworthy, auditable reports. |
| D12 | **Severity = deterministic rubric over LLM-extracted signals** (churn/financial/safety/repeat/unresolved), not an LLM number. | Auditable, consistent across millions, re-tunable without re-tagging. |
| D13 | **Synthetic enrichment is a separate, optional `datagen/` module** (DEMO only), not core. | BYO users bring real operational metrics; core stays dataset-agnostic. |
| D14 | **gbrain is L5-centric**: durable L5 entities *updated* each run (rolling stats, driver edges, period series, insights). | Consistent trends into the future; no run-dump blobs. |
| D15 | **Driver engine = over-indexing (lift) + significance (z-test + BH FDR)**, materialized as updatable `affects` edges per L5. | Ties pain to operational levers with statistical backing; both report views from one store. |
| D16 | **`datagen/` plants drivers via a documented latent model** `P(dim\|l5,signals)`; pipeline tags **before** enrich. | Makes drivers real + recoverable; doubles as validation; honest "simulated" labeling. |
| D17 | **Report has both views**: pain-first deep dives + ops-first hotspots; single-dim lift + at most one 2-way interaction. | Serves both pain owners and ops owners; keeps cells large + report readable. |

## Resolved (signed off in review)

| # | Question | Decision |
|---|----------|----------|
| O1 | Egregiousness weights `w1..w4` (volume, severity, spike, value). | **`0.35 / 0.25 / 0.25 / 0.15`** (percentile-normalized); `w4` redistributes if no monetary fact. |
| O2 | Z-score baseline window. | **trailing 8 weeks, min 3 weeks** history. |
| O3 | Agent insight writes. | **auto-commit as `status=draft`** until reviewed. |
| O4 | Build sample size. | **10k**. |
| O5 | Deep-dive count. | **top 10 egregious + top 5 spikes** (content = driver analysis). |
| O6 | Clustering for induction. | **HDBSCAN**, agglomerative fallback to hit ≤100. |
| O7 | Severity model. | **Signals → deterministic rubric** (D12). |
| O8 | Synthetic-dimension relationship. | **Documented driver model in `datagen/`** (D16). |
| O9 | Report views / dim depth. | **Both views + 1 interaction** (D17). |

## Open

_None blocking. Future: pgvector engine for shared brains; taxonomy migration job across pack versions._

## Inspirations referenced
gbrain (graph memory, schema packs, write→auto-link loop) · ECC (harness: skills/agents/rules/hooks)
· awesome-llm-apps (memory + RAG + agent-skill patterns).
