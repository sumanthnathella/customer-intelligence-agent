---
name: generate-report
description: Render the board-quality Markdown (+ JSON) report from gbrain per docs/REPORT_SPEC.md, with every claim traceable to a node id and exemplar.
tools: ["Read"]
when_to_use: Final step — produce the shareable deliverable.
---

# Skill: Generate Report

## Goal
Produce `reports/report_<window>.md` (+ `.json`) that a stakeholder can act on.

## Steps
1. Gather the findings + insights produced by `surface-pain-points` and `update-gbrain`.
2. Call `generate_report(...)` to render the required sections (`docs/REPORT_SPEC.md`):
   header → exec summary → egregious-by-volume → spikes-by-zscore → per-PP deep dives →
   what's-new-since-last-run → appendix (methodology + caveats).
3. Self-check before finishing:
   - Every number maps to a gbrain node id (faithfulness check).
   - Every deep dive has ≥2 verbatim exemplars + a recommended action.
   - Synthetic dimensions labeled as synthetic.

## Done when
- Both `.md` and `.json` exist for the window and pass the faithfulness self-check.

## Guardrails
- The model writes prose only; numbers are injected from gbrain, never typed by the LLM.
