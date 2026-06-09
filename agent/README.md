# `agent/` — Read Plane (shareable)

The Google ADK single-ReAct intelligence agent. Reads **gbrain**, reasons with OpenRouter Nemotron
(local fallback), and writes the report. **Never imports `tagging/`.** Shareable with anyone who has
the gbrain artifact + an OpenRouter key.

See [`../docs/AGENT.md`](../docs/AGENT.md) for the full spec.

## Modules (to implement)
| File | Role |
|------|------|
| `agent.py` | the ReAct `LlmAgent` (Nemotron primary, gemma4:31b fallback) |
| `tools.py` | read tools over gbrain + `write_memory` + `generate_report` |
| `report.py` | renderer for `docs/REPORT_SPEC.md` |
| `fast_api_app.py` | `agents-cli playground` / serving entrypoint |

## Run
```bash
uv run python -m agent.report     # headless report from current gbrain
agents-cli playground             # interactive
```
