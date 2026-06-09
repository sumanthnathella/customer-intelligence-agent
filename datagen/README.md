# `datagen/` — Synthetic Operational Dimensions (DEMO ONLY)

> ⚠️ **This module is a demo stand-in. Replace it with your real operational metrics.**
> The twcs transcripts have no order data, so this generates synthetic operational dimensions to
> demonstrate the driver analysis. A bring-your-own-data user **deletes/ignores** this and supplies
> Table B of [`../docs/INPUT_CONTRACT.md`](../docs/INPUT_CONTRACT.md) instead.

Produces operational dimensions for **already-tagged** conversations, drawn from a **documented driver
model** so the analytics engine has real, recoverable drivers to find. Imports only `shared/`.

See [`../docs/DATA_MODEL.md`](../docs/DATA_MODEL.md) for the dimension catalog and planted drivers.

## Modules (to implement)
| File | Role |
|------|------|
| `dimensions.py` | operational-dimension value catalog |
| `driver_model.py` | documented latent `P(dimension \| l5_id, signals)` — the planted drivers |
| `generate.py` | tagged conversations → synthetic operational table (seeded by `conversation_id`) |

Because the driver model is documented, `tests/` asserts the analytics engine **recovers** the planted
lifts — which doubles as validation of the methodology.
