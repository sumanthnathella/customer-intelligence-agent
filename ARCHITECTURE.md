# Architecture

System design for the Customer Intelligence Agent. This document is the source of truth for
*how the pieces fit*; deep dives live in [`docs/`](docs/).

## 1. Goals & non-goals

**Goals**
- Turn raw support transcripts into a ranked, evidence-backed view of the **most egregious customer pain points**.
- Tag every analyzed transcript into a **data-driven L1→L5 taxonomy** (≤100 unique L5 leaves).
- Rank pain points by **volume** and by **recent-week z-score** (spike detection on the real timeline).
- Persist findings in a **graph memory (gbrain)** that **accumulates** across runs/periods.
- Produce a **board-quality Markdown report** with exemplars, dimension cuts, and recommended actions.

**Non-goals**
- Real-time / streaming tagging (this is batch).
- Tagging *every* transcript with a frontier model (cost-prohibitive; we tag a stratified sample).
- A general chatbot — the agent is a focused pain-point analyst.

## 2. Dataset-agnostic core + replaceable producers (the core idea)

This is an **open-source agent**: anyone should be able to point it at *their own* transcripts and
*their own* operational metrics and get a report. So the design splits into a **dataset-agnostic core**
(analytics + gbrain + agent) and **replaceable, one-time producers** (tagging + datagen) that a user can
swap for their own data.

```
  ┌──────────── REPLACEABLE PRODUCERS (one-time; swap for your own) ────────────┐
  │  tagging/   transcripts ─► tags (L5 + signals)        [or bring your tags]   │
  │  datagen/   conversations ─► synthetic operational dims  [DEMO ONLY;         │
  │             (documented driver model)                    bring your real     │
  │                                                           operational metrics]│
  └───────────────────────────────────┬─────────────────────────────────────────┘
                       INPUT CONTRACT  │  (tagged transcripts ⋈ operational dimensions)
  ┌────────────────────────── DATASET-AGNOSTIC CORE ───────────▼─────────────────┐
  │  analytics/  severity · weekly z-score · egregiousness · driver/lift per L5   │
  │                              │ writes                                         │
  │                   ┌──────────▼───────────┐                                    │
  │                   │   gbrain (memory)    │  L5-centric graph; entities        │
  │                   │  L5 entities + edges │  UPDATED each run (not run-dumps)   │
  │                   └──────────▲───────────┘                                    │
  │  agent/      ReAct analyst (Nemotron) ─► query gbrain ─► report.md            │
  └──────────────────────────────────────────────────────────────────────────────┘
```

**Why this split:**
- **Tagging and datagen are one-time and dataset-specific** — they run rarely, locally, offline.
- **A bring-your-own-data (BYO) user replaces them**: supply your own tags (skip `tagging/`) and your
  own real operational metrics (skip `datagen/` — it exists only to demo on the dimensionless twcs data).
- **The core never sees raw data** — only the **input contract** ([`docs/INPUT_CONTRACT.md`](docs/INPUT_CONTRACT.md)):
  tagged transcripts joined to operational dimensions. So the agent ships and runs on any conforming dataset.
- **The agent is shareable** with just the gbrain artifact + an OpenRouter key.

## 3. Repository layout

```
customer-intelligence-agent/
├── README.md                 # orientation
├── ARCHITECTURE.md           # this file
├── CLAUDE.md                 # agent-harness entrypoint (ECC-style)
├── pyproject.toml            # deps (uv-managed)
├── .env.example              # OPENROUTER_API_KEY, OLLAMA_*, paths
├── build.py                  # top-level orchestrator: tagging → datagen → analytics → gbrain
│
├── shared/                   # CONTRACTS (no business logic) — the dataset-agnostic boundary
│   ├── config.py             # paths, model slugs, sample size, thresholds, seeds
│   ├── schemas.py            # pydantic: Transcript, Tag(+signals), Dimensions, PainPoint,
│   │                         #           PeriodMetric, Driver, Insight, Edge
│   └── contract.py           # validate a BYO dataset against the INPUT CONTRACT
│
├── tagging/                  # ── PRODUCER (one-time, dataset-specific, local LLM) ──
│   ├── ingest.py             # twcs.csv → reconstructed conversations
│   ├── taxonomy.py           # embed+cluster → LLM-named L1–L5 schema pack (≤100 L5)
│   ├── tag.py                # transcript → L5 leaf + severity SIGNALS + sentiment + confidence
│   └── run.py                # orchestrator: --build / --test-batch N
│
├── datagen/                  # ── PRODUCER (DEMO ONLY; replace with your real operational metrics) ──
│   ├── driver_model.py       # documented latent P(dimension | L5, signals) — planted drivers
│   ├── dimensions.py         # operational-dimension value catalog
│   └── generate.py           # tagged conversations → synthetic operational table (seeded)
│
├── analytics/                # ── DATASET-AGNOSTIC CORE (tags ⋈ dims → brain) ──
│   ├── severity.py           # signals → deterministic severity (rubric)
│   ├── metrics.py            # weekly volume, recent-week z-score
│   ├── egregiousness.py      # percentile-blended ranking
│   ├── drivers.py            # over-indexing (lift) + significance, per L5
│   └── build_brain.py        # upsert L5 entities + period_metrics + driver edges
│
├── gbrain/                   # ── MEMORY (L5-centric graph) ──
│   ├── store.py              # GBrainStore: engine contract (upsert/edge/query/traverse/vector/snapshot)
│   ├── graph.py              # L5 entities + typed edges + auto-link
│   ├── retrieval.py          # hybrid vector + graph retrieval
│   ├── schema_pack.py        # taxonomy-as-schema-pack loader
│   └── store/                # artifacts: gbrain.db (SQLite), snapshots/*.json
│
├── agent/                    # ── READ PLANE (shareable ADK agent) ──
│   ├── __init__.py
│   ├── agent.py              # single ReAct LlmAgent (Nemotron + local fallback)
│   ├── tools.py              # read tools over gbrain + write_memory + generate_report
│   ├── report.py             # report renderer (REPORT_SPEC.md)
│   └── fast_api_app.py       # `agents-cli playground` / serving entrypoint
│
├── docs/                     # deep dives (see index in README)
├── .claude/                  # harness: agents/ skills/ rules/ settings.json
├── reports/                  # generated reports (dated)
├── data/                     # raw + processed (gitignored)
└── tests/                    # unit + integration
```

## 4. Data flow (end-to-end)

> Note the **ordering**: tagging happens **before** enrichment so the demo driver model can condition
> synthetic dimensions on the assigned pain point. For a BYO user with real dimensions, steps 2 and 4 are
> simply *their data*, and the flow starts at step 5.

1. **Ingest** ([`docs/DATA_MODEL.md`](docs/DATA_MODEL.md)) — `twcs.csv` (~2.8M tweets) → conversations via
   `in_response_to_tweet_id`. **~1M for build**, **~1.8M reserved for test batches**, split by conversation.
   Keep real `created_at` for the time axis. *(producer: `tagging/`)*
2. **Induce taxonomy** ([`docs/TAXONOMY.md`](docs/TAXONOMY.md)) — embed a stratified ~10k sample with local
   `nomic-embed-text`, cluster, then `qwen2.5:7b` names L1 (pre/sale/post) → L2→L3→L4→L5. Unique paths,
   **≤100 L5 leaves**, frozen as a **schema pack**. *(producer: `tagging/`)*
3. **Tag** ([`docs/TAGGING.md`](docs/TAGGING.md)) — `qwen2.5:7b` (JSON-constrained) assigns each transcript to
   one L5 leaf + **severity signals** (churn/financial/safety/repeat/unresolved) + sentiment + confidence.
   *(producer: `tagging/`)*
4. **Enrich (DEMO)** ([`docs/DATA_MODEL.md`](docs/DATA_MODEL.md)) — `datagen/` draws operational dimensions
   from a **documented driver model** `P(dim | L5, signals)`, so drivers are real + recoverable.
   **BYO users skip this and join their real operational metrics.** *(producer: `datagen/`)*
5. **Analyze** ([`docs/ANALYTICS.md`](docs/ANALYTICS.md)) — **dataset-agnostic core**: severity from signals;
   weekly volume + z-score; egregiousness; **driver/lift per L5** (over-indexing + significance).
6. **Build brain** ([`docs/GBRAIN.md`](docs/GBRAIN.md)) — **upsert L5 entities** (durable), append this
   period's `period_metric`, **update `affects` driver edges** (lift/support/p per dimension-value), refresh
   exemplars; auto-link; snapshot. The brain is keyed by L5 and **updated**, not dumped per run.
7. **Agent reads + reasons** ([`docs/AGENT.md`](docs/AGENT.md)) — selects top egregious L5s (volume + spike),
   deep-dives each (drivers, exemplars, trend vs prior periods), writes insights back, renders the report.
8. **Report** ([`docs/REPORT_SPEC.md`](docs/REPORT_SPEC.md)) — dated, board-quality Markdown (+ JSON).

## 5. gbrain: graph memory (the contract)

Inspired by [gbrain](https://github.com/garrytan/gbrain): **vectors find what's *semantically* close;
the graph finds what's *factually* connected.** We use a hybrid of both.

- **Engine contract** — `GBrainStore` exposes a small, stable API (`upsert_node`, `add_edge`,
  `query`, `traverse`, `vector_search`, `snapshot`). Default engine = **SQLite + JSON** (portable, no
  services); the contract leaves room for a pgvector engine later.
- **Schema pack** — the L1–L5 taxonomy is a versioned pack (like gbrain's typed taxonomy). The pack
  threads through every read/write so a new run can't contaminate a different taxonomy version.
- **L5-centric (not run-centric)** — each **L5 pain point is a durable entity**. A run *updates* it:
  rolling stats, a new `period_metric`, refreshed driver (`affects`) edges, accumulated insights. Trends
  stay consistent into the future because we always update the *same* entity, never dump one-off run blobs.
- **Both report views from one store** — pain-first (group by L5 → its `affects` edges) and ops-first
  (group `affects` edges by dimension → top L5s) are two traversals of the same L5-keyed graph.
- **Auto-link** is pure pattern matching (no LLM) on taxonomy path + drivers, so the graph stays fresh cheaply.

Node & edge types, retrieval, and the SQLite schema are specified in [`docs/GBRAIN.md`](docs/GBRAIN.md).

## 6. Models & cost

| Task | Engine | Why |
|------|--------|-----|
| Bulk tagging (10k + millions) | local **Ollama `qwen2.5:7b`** | free, unlimited, JSON-constrained, fast on M4 24GB |
| Taxonomy embeddings/clustering | local **`nomic-embed-text`** | already installed, no download |
| Agent reasoning / report | **OpenRouter `nvidia/nemotron-3-ultra:free`** | frontier reasoning, only a few calls/run |
| Agent fallback (rate limit) | local **`gemma4:31b` / `qwen3.5:27b`** | resilience, fully offline |

This split keeps the system **free to run** while still using a frontier model where it matters.

## 7. The agent harness (`.claude/`)

Following [ECC](https://github.com/affaan-m/ecc): the harness makes the agent's behavior explicit and reusable.

- **`skills/`** — primary workflow surface (`SKILL.md` each): `surface-pain-points`, `tag-transcripts`,
  `update-gbrain`, `generate-report`.
- **`agents/`** — scoped subagent definitions: `pain-point-analyst`, `taxonomy-architect`, `report-writer`.
- **`rules/`** — always-follow guidelines (`common.md`, `python.md`): research-first, no fabricated numbers,
  deterministic pipeline, cite gbrain node IDs in the report.
- **`settings.json`** — hooks (e.g., block committing data artifacts; lint on edit).

## 8. Quality & verification

- **Determinism** — pipeline is seeded; re-runs reproduce the same tags/aggregates.
- **Report quality bar** — every claim traces to a gbrain node ID + exemplar; numbers come from
  aggregates, never the LLM. See [`docs/REPORT_SPEC.md`](docs/REPORT_SPEC.md).
- **Tests** — unit (z-score, egregiousness, taxonomy uniqueness/cap, gbrain upsert+auto-link),
  integration (tiny fixture end-to-end → populated gbrain + report). See [`docs/EVALUATION.md`](docs/EVALUATION.md).

## 9. Open design questions (for review)

Tracked in [`docs/DECISIONS.md`](docs/DECISIONS.md) §Open. Highlights: exact egregiousness weighting,
z-score baseline window length, and whether the agent should auto-write insights or propose-then-confirm.
