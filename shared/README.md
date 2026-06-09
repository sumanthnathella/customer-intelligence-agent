# `shared/` — Cross-Plane Contracts

Contracts shared across all modules. **No business logic** — just config, typed schemas, and the BYO
input-contract validator. Every module may import `shared/`; `shared/` imports none of them.

## Modules (to implement)
| File | Role |
|------|------|
| `config.py` | paths, model slugs, sample size, severity rubric, egregiousness weights, z-score window, seeds |
| `schemas.py` | pydantic models: `Transcript`, `Tag`(+signals), `Dimensions`, `PainPoint`, `PeriodMetric`, `Driver`, `Insight`, `Edge` |
| `contract.py` | validate a BYO dataset (Table A + Table B) against the [input contract](../docs/INPUT_CONTRACT.md) |

> The operational-dimension **value catalog** lives in `datagen/` (demo), not here — the core discovers
> dimensions from the data, so `shared/` stays dataset-agnostic.
