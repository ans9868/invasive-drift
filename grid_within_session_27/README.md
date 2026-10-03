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

## Progress log 2 (Step 0.5)
- **Step 0.5 done** — `metrics.py` + `selftest_metrics.py` (ALL PASS, job `19114193`), per-window **context
  block** in `cache.py`, non-breaking **KF `trace(P)`** recorder, smoke runs both selftests.
- **Smoke caught 3 more real bugs:** `MomDiagSelf` emptied by an insert-splice; `cache.py` unpacked the
  **left** singular vectors instead of the right (`Vh`); a selftest put the direction **exactly on a bin
  edge** (45°) → use 22.5°.
- **Context verified** (`CO-20131003`, window 0): `rate_mean=3.47 Hz`, `rate_median=1.38`,
  `frac_silent=0.169`, `active_units=59/71`, `mean_pairwise_corr=0.083`, `pc1_var=0.19`, `eff_dim=18.4`,
  **`subspace_angle_deg=47.7`** (matches the ~56° rotation seen earlier), `speed_mean=4.31`,
  `moving_frac=0.33`, `dir_coverage=8/8`.
- **Next:** **Step 0** — `baselines.py` (`r2_mean`, `r2_persist_lag1/lag12`, `r2_target` vs ridge, 53
  sessions) → **Block 3**.

## Progress log 3 (Step 0.6 — decoder cache)
- **Step 0.6 done** — pickle cache + config-hash guard + `selftest_cache.py`. Job `19114705`: **ALL PASS**.
- **Round-trip PROVEN:** fit → pickle → load → **byte-identical predictions** for all five decoders
  (`ridge/wiener/kf_posvel/mlp/gru`, `maxdiff=0.00e+00`). Guards verified: `hash_mismatch`, `missing`,
  `corrupt` all detected.
- **Smoke caught a CRITICAL bug:** `wiener` (L=5), `mlp` (L=3), `gru` (L=10) return **len(X)−L**
  predictions → predictions were being compared to the **wrong target rows**. Fixed with `align_tail()`;
  rule added to PLAN §5.
- Cache on a real session: `decoders(ecd63a298da0): ridge=0.452 wiener=0.513 kf_posvel=0.475 mlp=0.996
  gru=0.557 (20.8s)`.
- ⚠️ **FLAG:** `r2_burnin` is **in-sample** (fit and scored on the same burn-in rows) → inflated, especially
  `mlp=0.996` (held-out it was 0.576). Proposed fix: score `r2_burnin` on a **held-out half of burn-in**.
  Decide before P2-scale.
- **Next:** **Step 0** — `baselines.py` → **Block 3**.
