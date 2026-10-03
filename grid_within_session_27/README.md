# grid_within_session_27

**Within-session** decoder × objective grid with adaptation-data learning curves.

- **Authoritative plan:** [`PLAN.md`](PLAN.md) — read this first.
- **What it tests:** for each *frozen* decoder (ridge/wiener/kf_posvel/mlp/gru), which **adapters**
  (label-free vs labeled, decoder-agnostic vs **decoder-aligned**) recover R² under drift, and **how
  much adaptation data** they need (the learning curve).
- **Motivation (the flaw being fixed):** our earlier `25`/`26` adapters minimized **proxy objectives**
  (feature-distribution matching) that are **not aligned with the end-to-end decoder**, and were given a
  fixed window size (unfair to high-capacity adapters).
- **Related:** `../adapters/` (adapter library) · `../drafts/17_plan_27_adapter_curves.md` (discussion
  trail) · `../findings/2026-10-03_lowrank_normalizer.md`, `..._normalizer_perspective.md`.

Status: **PLAN ONLY** — no code yet.
