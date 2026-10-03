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

## Progress log
- **P0 done** — skeleton, `config.json`, scratch layout.
- **P1 done** — `../adapters/` library (12 closed-form adapters) + `selftest.py`; **ALL PASS** on Torch
  (job `19112422`). SMOKE caught two real bugs: an `insert_line` splice that silently emptied
  `MomDiagSelf`'s body, and an ill-posed `cca` (unpaired CCA ≡ ZCA) → replaced by **subspace alignment**
  (`subspace`, verified: principal angle 49.3° → 17.1°).
- **P2 tracer done** — `cache.py --n 1` → `CO-20131003`: 71 units, T=66321, 8/8 windows valid, 8 dirs,
  **1.0 s, 227 MB** → `artifacts/perich_subC/`.
- **Deviation from plan (documented):** `reg_ref` is subsumed by `zca` (unpaired LSQ is ill-posed);
  `cca` → `subspace` (SA). Both noted in `adapters/feature.py`.
- **Next:** P3 `fit.py` (sufficient stats → transforms) + P4 `evaluate.py` (apply + R² → long row);
  then full `cache.py` over 53 sessions; then the tracer bullet (`--n 1 --nwin 2 ridge --N 1.0`).
