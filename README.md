# Customer Intelligence Agent

> Surface the most egregious customer pain points from millions of support transcripts —
> tag them into a data-driven L1–L5 taxonomy, rank them by volume and recent-week spikes,
> and grow a persistent graph "brain" (gbrain) that gets smarter every time new transcripts arrive.

The repo is a **dataset-agnostic core** plus **replaceable, one-time producers**, so you can point the
agent at *your own* transcripts + operational metrics:

| Part | Folder | Role | Swappable? |
|------|--------|------|------------|
| **Tagging** | [`tagging/`](tagging/) | transcripts → tags (L5 + signals), local LLM | yes — bring your own tags |
| **Datagen** | [`datagen/`](datagen/) | synthetic operational dims (**demo only**) | yes — bring your real metrics |
| **Analytics** | [`analytics/`](analytics/) | severity · z-score · egregiousness · **drivers** | core |
| **gbrain** | [`gbrain/`](gbrain/) | L5-centric graph memory | core |
| **Agent** | [`agent/`](agent/) | reads gbrain → quality report | core (shareable) |

Producers and core communicate **only** through the [**input contract**](docs/INPUT_CONTRACT.md)
(tagged transcripts ⋈ operational dimensions) and the **gbrain** memory. No module imports another
producer; the agent ships with just the gbrain artifact.

```
 transcripts ─►[tagging]─► tags ┐
                                 ├─(input contract)─►[analytics]─►(gbrain L5 memory)─►[agent]─► report.md
 operational ─►[datagen*]─► dims ┘     *demo only; BYO real metrics
```

## Why this design

- **Tagging + datagen are one-time and dataset-specific** — they run locally/offline and are **replaceable**.
  A BYO user supplies their own tags and **real** operational metrics and skips them.
- **The core is dataset-agnostic** — it depends only on the input contract, so it runs on any conforming dataset.
- **The agent is light + shareable** — anyone can point it at a gbrain and get a quality report (Nemotron reasoning only).
- **gbrain is L5-centric memory** (inspired by [gbrain](https://github.com/garrytan/gbrain)): durable
  pain-point entities *updated* each run, with **driver edges** (operational over-indexing) and a *schema-pack* taxonomy.

## Quick links

- **System design**: [`ARCHITECTURE.md`](ARCHITECTURE.md) ← start here
- **Agent harness** (skills/agents/rules): [`.claude/`](.claude/) and [`CLAUDE.md`](CLAUDE.md)
- **Input contract (BYO data)**: [`docs/INPUT_CONTRACT.md`](docs/INPUT_CONTRACT.md)
- **Analytics engine (severity, egregiousness, drivers)**: [`docs/ANALYTICS.md`](docs/ANALYTICS.md)
- **Graph memory design**: [`docs/GBRAIN.md`](docs/GBRAIN.md)
- **L1–L5 taxonomy (schema pack)**: [`docs/TAXONOMY.md`](docs/TAXONOMY.md)
- **Demo data + synthetic dimensions**: [`docs/DATA_MODEL.md`](docs/DATA_MODEL.md)
- **Tagging producer**: [`docs/TAGGING.md`](docs/TAGGING.md)
- **Agent design**: [`docs/AGENT.md`](docs/AGENT.md)
- **Report quality spec**: [`docs/REPORT_SPEC.md`](docs/REPORT_SPEC.md)
- **Decisions (ADRs)**: [`docs/DECISIONS.md`](docs/DECISIONS.md)

## Status

� **Implemented & runnable.** The full pipeline (tagging → datagen → analytics → gbrain → report) is
implemented and covered by the test suite (`uv run pytest`). See [`docs/DECISIONS.md`](docs/DECISIONS.md)
for locked decisions.

```bash
uv sync                                          # install deps
uv run python -m tagging.run --build --sample 10000   # tag (local Ollama qwen2.5:7b)
uv run python build.py --demo --skip-tagging          # datagen → analytics → gbrain → report
uv run python build.py --report-only                  # re-render report from current gbrain
uv run pytest -q                                      # run the test suite
```

## Inspirations

- [garrytan/gbrain](https://github.com/garrytan/gbrain) — graph-based memory, schema packs, the write→auto-link loop.
- [affaan-m/ECC](https://github.com/affaan-m/ecc) — the agent harness model (skills, agents, rules, hooks).
- [Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps) — memory + RAG + agent-skill patterns.
