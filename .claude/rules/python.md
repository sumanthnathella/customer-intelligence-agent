# Rules — Python

- **Runtime**: Python 3.11+, managed with **uv**. Run code via `uv run ...`. Add deps with `uv add`.
- **Style**: ruff (lint+format), line length 88. Type hints on public functions.
- **Data**: pandas + pyarrow (parquet). Prefer vectorized ops; stream/iterate for the millions-row reserve.
- **LLM calls**: Ollama via `ollama` client (bulk tagging) with `format=<json schema>`; OpenRouter via
  `LiteLlm` (agent). Always set timeouts; always cache bulk calls.
- **Validation**: pydantic models in `shared/schemas.py` for every artifact crossing a boundary.
- **Determinism**: pass an explicit `seed`; never rely on dict/set ordering for outputs.
- **Errors**: fail loudly in the pipeline (data integrity), degrade gracefully in the agent (fallback model).
- **Tests**: pytest. New metric logic ships with a unit test. Keep an integration fixture runnable offline.
- **No notebooks in the repo** for production logic — keep logic importable from `tagging/`, `agent/`, `gbrain/`.
