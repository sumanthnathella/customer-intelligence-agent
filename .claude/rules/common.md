# Rules — Common (always follow)

1. **Research-first.** Read the relevant `docs/*.md` before editing. Never invent schema, columns, or metrics.
2. **Module boundaries are sacred.** Producers (`tagging/`, `datagen/`) must not import each other or the
   core; the core (`analytics/`→`gbrain/`→`agent/`) must not import a producer. All may import `shared/`.
   Cross-module wiring lives only in the top-level `build.py`. A change that couples modules is a bug.
3. **Numbers come from data, not the model.** All metrics originate in `analytics/` / gbrain. The LLM
   narrates and hypothesizes; it never computes or guesses a number.
4. **Evidence or it didn't happen.** Every report claim cites a gbrain node id + ≥1 exemplar transcript id.
5. **Determinism.** Seed everything. No unseeded randomness. Re-runs must reproduce aggregates.
6. **Local-first cost.** Bulk LLM work runs on Ollama. OpenRouter Nemotron is reserved for the agent.
7. **Honesty about synthetic data.** Order dimensions are synthetic — label them as such in outputs.
8. **Small, scoped changes.** Don't refactor beyond the task. Preserve config values and structure.
9. **Schema-pack discipline.** Tagging classifies into existing L5 leaves; it never invents taxonomy paths.
10. **No secrets in code.** `OPENROUTER_API_KEY` lives in `.env` only.
