# Intelligence Agent (Read Plane)

A **single ReAct `LlmAgent`** (Google ADK) that reads gbrain and produces the report. Shareable:
anyone with the gbrain artifact + an OpenRouter key can run it. Lives in `agent/`; never imports `tagging/`.

## Why single ReAct (not multi-agent)

The heavy analytics already happened deterministically in the pipeline. The agent is a **thin
reasoning/synthesis layer** — so a single tool-using ReAct loop keeps LLM calls few (free-tier safe),
is easy to debug, and mirrors the existing web-research agent pattern. (A multi-agent SequentialAgent
was considered and rejected for now — see `docs/DECISIONS.md`.)

## Model

- Primary: **OpenRouter `nvidia/nemotron-3-ultra:free`** via `LiteLlm`.
- Fallback (on rate limit / error): local **`gemma4:31b`** or **`qwen3.5:27b`** via Ollama.

## Tools (`agent/tools.py`) — thin wrappers over gbrain

**Read**
- `get_taxonomy()` — the schema pack (L1–L5 tree).
- `get_top_l5(by="egregiousness"|"zscore"|"volume", n)` — ranked pain points (+ egregiousness component breakdown).
- `get_zscore_spikes(weeks)` — pain points spiking in recent weeks.
- `get_drivers(l5_id)` — the operational drivers (over-indexed dimension values) with `lift, support, p_value`.
- `get_ops_hotspots(dimension?)` — **ops-first** inversion: top pain points per operational segment.
- `get_exemplars(l5_id, k)` — representative transcripts (evidence).
- `read_memory(l5_id)` — prior periods + prior insights + prior driver lifts (trend deltas).

**Write**
- `write_memory(insight)` — persist an `insight` node + `explains`/`cites` edges.
- `generate_report(...)` — render `reports/report_<window>.md` per `REPORT_SPEC.md`.

> Numbers come **only** from these tools (i.e., from aggregates/gbrain). The model narrates; it never computes metrics.

## Reasoning flow (one ReAct loop)

```
1. get_taxonomy + get_top_l5(egregiousness) + get_zscore_spikes   → candidate egregious L5s
2. for each top L5: get_drivers + get_exemplars + read_memory      → operational drivers + evidence + trend
3. get_ops_hotspots                                               → ops-first "Operations Hotspots" view
4. synthesize root cause + action (tied to driving lever); write_memory(insight)
5. generate_report                                               → dated quality report (+ JSON)
```

Only a handful of LLM calls per run, keeping the free tier comfortable.

## Running

```bash
uv run python -m agent.report        # headless: render report from current gbrain
agents-cli playground                # interactive
```

`agent/fast_api_app.py` exposes the ADK app for `agents-cli playground` / serving. Configured via
`agents-cli-manifest.yaml` with `agent_directory: agent`.
