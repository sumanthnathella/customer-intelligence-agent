# CLAUDE.md — Agent Harness Entrypoint

This file orients any coding/agent harness (Claude Code, Codex, Cursor, Windsurf) working in this repo.
It mirrors the [ECC](https://github.com/affaan-m/ecc) model: **skills** are the primary workflow surface,
**agents** are scoped subagents, **rules** are always-follow, **hooks** fire on tool events.

## What this project is

A **dataset-agnostic core** + **replaceable producers**. Read [`ARCHITECTURE.md`](ARCHITECTURE.md) first.
- **Producers** (`tagging/`, `datagen/`): one-time, dataset-specific, local. Swappable for BYO data.
- **Core** (`analytics/` → `gbrain/` → `agent/`): depends only on the [input contract](docs/INPUT_CONTRACT.md).
- **Memory** (`gbrain/`): L5-centric graph + SQLite; durable pain-point entities updated each run.

## Golden rules (see `.claude/rules/`)

1. **Research-first.** Before editing, read the relevant `docs/*.md`. Do not invent schema or columns.
2. **Never fabricate numbers.** All metrics come from `analytics/` / gbrain. The LLM narrates; it does not compute.
3. **Respect module boundaries.** Producers (`tagging/`, `datagen/`) never import each other or the core.
   The core (`analytics/`→`gbrain/`→`agent/`) never imports a producer. Everyone may import `shared/`.
   Cross-module wiring happens only in the top-level `build.py` orchestrator.
4. **Determinism.** The pipeline is seeded and reproducible. No randomness without a fixed seed.
5. **Local-first cost.** Bulk LLM work runs on Ollama. Reserve OpenRouter Nemotron for the agent.
6. **Cite evidence.** Report claims reference gbrain node IDs + exemplar transcript IDs.

## Skills (`.claude/skills/`)

| Skill | Purpose |
|-------|---------|
| `tag-transcripts` | Run the build/test-batch tagging + analytics pipeline |
| `surface-pain-points` | Rank + deep-dive the most egregious L5s (with operational drivers) |
| `update-gbrain` | Write/auto-link a run's findings into L5-centric memory |
| `generate-report` | Render the quality report (pain-first + ops-first) from gbrain |

## Subagents (`.claude/agents/`)

| Agent | Scope |
|-------|-------|
| `taxonomy-architect` | Induce/curate the L1–L5 schema pack |
| `pain-point-analyst` | Reason over aggregates + gbrain to pick + explain egregious issues |
| `report-writer` | Compose the board-quality Markdown report |

## Commands (planned)

```bash
uv run python -m build --demo --sample 10000    # one-time: tagging→datagen→analytics→gbrain
uv run python -m build --test-batch 1           # next held-out slice → append to gbrain
uv run python -m agent.report                   # render report from current gbrain
agents-cli playground                           # interactive agent
```

## Current phase

� Implemented & runnable. The full pipeline and test suite are in place. When changing behavior,
read the relevant `docs/*.md` first, keep module boundaries intact, and update/extend `tests/`
rather than weakening them.
