# gbrain — Graph Memory

The persistent memory and the **only contract** between the analytics core and the agent.
Design principle (from [gbrain](https://github.com/garrytan/gbrain)):

> **Vector search returns chunks that are *semantically* close. The graph returns chunks that are
> *factually* connected. Hybrid search pulls from both; auto-linking on every write keeps the graph fresh.**

**L5-centric, not run-centric.** Each L5 pain point is a **durable entity** that runs *update* — its
rolling stats, its driver edges, its period time-series, its insights all accumulate against the *same*
node. We never write throwaway per-run blobs, so trends stay consistent into the future.

**Two layers, one store.** gbrain does two jobs at once:
- **Materialized aggregates (the numbers).** `analytics/` precomputes every metric deterministically and
  stores them **as node/edge props** — L5 volume, the full **(L5 × dimension-value) heatmap** (`support`,
  `share`, `lift`), weekly series. The agent **reads** these; it never re-aggregates (rule D11).
- **Graph structure (the connections).** The same nodes/edges let the agent *traverse* relationships
  (L5→drivers, L5→spikes, L5→prior insights, dimension→top L5s) to assemble a deep dive.

So the graph is the **index/structure**; the aggregates are the **numbers carried on it**. An L5's full
set of `affects` edges *is* its dimension heatmap — already computed, persisted, and trendable.

## Engine contract (`gbrain/store.py`)

A small, stable interface so the storage engine can evolve without touching callers:

```
upsert_node(type, id, props, embedding?) -> node_id
add_edge(src_id, type, dst_id, props?)   -> edge_id
get_node(id) / get_edges(id, type?)
traverse(start_id, edge_types, depth)    -> subgraph        # factual connections
vector_search(type, query_embedding, k)  -> nodes           # semantic neighbors
query(type, filters)                     -> nodes           # structured slice
snapshot(run_id) -> path                 # JSON export of the run
```

**Default engine = SQLite + JSON snapshots** (portable, zero services). The contract leaves room for a
`pgvector` engine later for multi-user/shared deployments.

## Node types

| Node | Key | Notable props |
|------|-----|----------------|
| `pain_point` | `l5_id` | full L1–L5 path, definition, first_seen, last_seen, pack_version |
| `period_metric` | `l5_id@week` | volume, share, severity_avg, sentiment_avg, zscore, egregiousness, order_value_impact |
| `dimension` | `dim:value` | dimension name, value (e.g. `carrier:FedEx`); discovered from the data, not hardcoded |
| `exemplar` | `transcript_id` | snippet, severity, sentiment, week, brand |
| `insight` | `insight_id` | summary, root_cause, recommended_action, status, run_id, evidence[] |
| `run` | `run_id` | window, n_transcripts, pack_version, created_at |

`pain_point` carries **rolling, updatable** props: `total_volume`, `severity_avg`, `latest_zscore`,
`latest_egregiousness`, `top_drivers` (denormalized for fast reads). These are *overwritten* each run;
the per-period history lives in `period_metric` nodes.

## Edge types (typed, directional)

| Edge | From → To | Meaning |
|------|-----------|---------|
| `child_of` | pain_point → pain_point | taxonomy hierarchy (L5→L4→…→L1 rollup) |
| `measured_in` | pain_point → period_metric | a weekly measurement of a pain point |
| `spiked_in` | pain_point → period_metric | high-z-score weeks (factual "spike" connection) |
| `affects` | pain_point → dimension | **the (L5 × dimension-value) heatmap cell** — props `{support, share, lift, p_value, significant, excess, period, history[]}`, **updated each run** |
| `exemplified_by` | pain_point → exemplar | representative transcript |
| `explains` | insight → pain_point | an insight about a pain point |
| `cites` | insight → exemplar / period_metric | evidence backing the insight |
| `co_occurs` | pain_point → pain_point | appear together in conversations |

## The write loop (per run) — update by L5

```
analyze → upsert L5 entity → append period_metric → update affects(driver) edges → refresh exemplars → auto-link → snapshot
```

- **Upsert L5 entity**: `pain_point` is keyed by `l5_id`; we update its rolling props (don't duplicate).
- **Append period_metric**: one new `l5_id@week` node per period → the time-series that powers trends/z-scores.
- **Update driver edges**: recompute `affects` per dimension-value with fresh `{lift, support, p_value}`;
  upsert the edge so a driver's *own* trend is queryable (e.g. lift 3.1× → 3.8×).
- **Auto-link** (no LLM): taxonomy path → `child_of`; high z-score period → `spiked_in`; shared
  conversations → `co_occurs`. Cheap and always fresh.
- **Accumulation**: the brain gets richer as test batches stream in — same entities, more history.

## Hybrid retrieval (`gbrain/retrieval.py`)

The agent asks for "the most egregious pain points" and gets a **subgraph**, not just rows:
1. **Structured** — top L5s by `egregiousness` and by `zscore` (`query` + `traverse measured_in`).
2. **Graph expansion** — follow `affects` (driver dimensions + lift), `spiked_in` (which weeks),
   `co_occurs` (related pain), `exemplified_by` (evidence), and prior `insight`/`explains` from earlier runs.
3. **Semantic** — `vector_search` over `pain_point`/`insight` embeddings to surface analogues the
   structured filters might miss.

### Two report views, one store
- **Pain-first** (deep dives): start at an L5 → traverse its `affects` edges → its top drivers.
- **Ops-first** ("Operations Hotspots"): start at a `dimension` node → traverse incoming `affects` edges
  → the top L5s concentrated on that operational segment. Same edges, opposite direction.

## SQLite schema (sketch)

```
nodes(id TEXT PK, type TEXT, props JSON, embedding BLOB, pack_version TEXT, updated_at)
edges(id TEXT PK, src TEXT, type TEXT, dst TEXT, props JSON, UNIQUE(src,type,dst))  -- upsert driver edges
runs(run_id TEXT PK, window TEXT, n INTEGER, pack_version TEXT, created_at)
-- indexes on nodes(type), edges(src,type), edges(dst,type)
```

`UNIQUE(src,type,dst)` makes `affects`/`measured_in` **upsertable** — a run updates the edge's props
(lift, support, p) instead of duplicating, which is what keeps the brain L5-centric.

JSON snapshot per run in `gbrain/store/snapshots/<run_id>.json` for portability + diffing.
