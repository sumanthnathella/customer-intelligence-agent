# L1–L5 Pain-Point Taxonomy (Schema Pack)

The taxonomy is a **versioned schema pack** (concept borrowed from
[gbrain](https://github.com/garrytan/gbrain)). It defines the *shape* of the brain; every gbrain
read/write is scoped to a pack version so different runs can't contaminate each other.

## Structure

A strict 5-level tree. Each leaf is a **unique full path** `L1 ▸ L2 ▸ L3 ▸ L4 ▸ L5` with a stable `l5_id`.

| Level | Meaning | Source |
|-------|---------|--------|
| **L1** | Lifecycle phase | Fixed: `pre_sale`, `sale`, `post_sale` |
| **L2** | Pain area within phase | Data-driven (named by LLM) |
| **L3** | Issue type | Data-driven |
| **L4** | Cause / mechanism | Data-driven |
| **L5** | Specific failure (leaf) | Data-driven, **≤100 total** |

Example path:
`post_sale ▸ delivery ▸ late_delivery ▸ carrier_delay ▸ package_stuck_in_transit_7d+`

## Constraints (locked)

- **Data-driven**, not hand-authored — induced from the real transcripts.
- **Unique paths**: no two L5 leaves share an identical L1→L5 path; `l5_id` is stable across runs.
- **Cap = 100 L5 leaves.** Clustering targets ≤100; overflow clusters merge into nearest leaf.
- L1 is fixed to the three lifecycle phases so reports are always organized pre/sale/post.

## Induction procedure (`tagging/taxonomy.py`)

1. **Embed** a stratified ~10k sample of customer utterances with local `nomic-embed-text`.
2. **Cluster** (e.g. HDBSCAN / agglomerative) to ~60–100 fine clusters; compute cluster exemplars.
3. **Name & nest** — local `qwen2.5:7b` reads each cluster's exemplars and proposes its
   `L1▸L2▸L3▸L4▸L5` path. A reconciliation pass merges near-duplicate paths and enforces the cap.
4. **Freeze** — write `gbrain/store/taxonomy.json` (the schema pack): `version`, `created_at`,
   `leaves[] = {l5_id, l1, l2, l3, l4, l5, centroid, exemplar_ids, definition}`.

## Tagging contract (`tagging/tag.py`)

Once frozen, tagging is **classification into existing leaves** — the model may **not** invent paths.
Each transcript → `{l5_id, severity(1–5), sentiment, root_cause, dimension_impacted, confidence}`.
Low-confidence / out-of-taxonomy items are routed to a `__unmapped__` holding leaf for later review.

## Versioning & evolution

- Re-inducing the taxonomy creates a **new pack version**; old gbrain nodes keep their original version.
- A future `migrate` job can remap old `l5_id`s to a new pack (à la gbrain `unify-types`).
- Test batches tag against the **frozen** pack so trends stay comparable across periods.
