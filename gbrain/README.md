# `gbrain/` — Graph Memory (the contract)

The persistent memory and the **only interface** between the build and read planes. Graph + SQLite +
JSON snapshots. The L1–L5 taxonomy is a versioned **schema pack**. Every write **auto-links** typed
edges so retrieval can follow *factual* connections, not just semantic ones.

See [`../docs/GBRAIN.md`](../docs/GBRAIN.md) for node/edge types, retrieval, and the SQLite schema.

## Modules (to implement)
| File | Role |
|------|------|
| `store.py` | `GBrainStore` engine contract (upsert/edge/query/traverse/vector_search/snapshot) |
| `graph.py` | nodes + typed edges + auto-link rules |
| `retrieval.py` | hybrid vector + graph retrieval |
| `schema_pack.py` | taxonomy-as-schema-pack loader/versioning |

## Artifacts (gitignored)
```
store/
├── gbrain.db          # SQLite (nodes, edges, runs)
├── taxonomy.json      # active schema pack
└── snapshots/*.json   # per-run JSON exports
```
