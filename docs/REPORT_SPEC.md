# Report Specification (Quality Bar)

The deliverable is a **board-quality** Markdown report (+ machine-readable JSON). Every quantitative
claim must be **traceable** to a gbrain node ID and at least one exemplar transcript. The LLM writes
prose; it never produces numbers.

## File outputs
- `reports/report_<window>.md` — the narrative report.
- `reports/report_<window>.json` — structured findings (for dashboards / diffing).

## Required sections

1. **Header** — window analyzed, # transcripts, taxonomy pack version, run id, generated-at.
2. **Executive summary** — 5–8 bullets: the top egregious pain points, biggest movers vs last period,
   estimated order-value at risk. Each bullet cites `l5_id`.
3. **Most egregious pain points (by volume)** — table: L1–L5 path, volume, share, avg severity, avg
   sentiment, egregiousness score.
4. **Biggest spikes (by recent-week z-score)** — table: path, z-score, this-week vs baseline volume,
   first-spike week. This is the "what changed *recently*" view.
5. **Per pain-point deep dives** (top 10 egregious + top 5 spikes) — for each:
   - One-line definition + full L1–L5 path + `l5_id` + egregiousness **component breakdown**.
   - **Operational drivers** — the dimension values that over-index, with **lift, support (n), p-value**
     (e.g. `vehicle_type=drone` lift 3.1×, n=412, p<0.001). This ties the pain to operational levers.
   - **Trend** vs prior periods (▲/▼ deltas pulled from gbrain) — including whether a *driver's* lift is rising.
   - **Evidence** — 2–3 verbatim exemplar snippets with `transcript_id`.
   - **Root cause** (hypothesis) + **Recommended action** tied to the driving lever + owner-area.
6. **Operations Hotspots (ops-first)** — inverted view: for high-impact operational segments
   (`carrier=FedEx`, `fulfillment_type=BOPIS`, `vehicle_type=drone`, …), the top pain points concentrated
   there. The view an ops owner reads. Built from the same `affects` edges, traversed by dimension.
7. **What's new since last run** — insights added to gbrain this run; resolved/regressed items; driver shifts.
8. **Appendix** — methodology (severity rubric, egregiousness weights, z-score window, lift + FDR), and
   data caveats. **Operational dimensions in the demo are simulated** — stated plainly.

## Quality rules
- **Evidence or it didn't happen** — no claim without a node ID + exemplar.
- **Recency matters** — spikes (z-score) are surfaced alongside raw volume, not buried.
- **Actionable** — every deep dive ends with a concrete recommended action and owner-area.
- **Honest caveats** — synthetic dimensions are clearly labeled as synthetic.
- **Diffable** — JSON mirrors the markdown so runs can be compared programmatically.

## Methodology references
Severity rubric, egregiousness formula, and the driver/lift engine are fully specified in
[`ANALYTICS.md`](ANALYTICS.md). Weights/windows live in `shared/config.py` and are recorded in
[`DECISIONS.md`](DECISIONS.md).
