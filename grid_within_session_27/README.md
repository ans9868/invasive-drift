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

Status: **P0–P4 code-complete; grid NOT yet run** (all four selftests green on Torch; tracer validated).

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

## Progress log 4 (Idea 16 sanity + windows removed)
- **Idea 16 DONE — perfect reproduction.** `sanity_8020.py` (job `19118091`) on 53 sessions:
  **ridge 0.357 · wiener 0.404 · kf_posvel 0.393** — *exactly* the decoder zoo's numbers. The pipeline
  (load → bin → standardise → `align_tail` → R²) is validated end-to-end against an independent script,
  and the negative `r2_burnin_out` values are confirmed a **setting** effect, not a bug.
- **Windows REMOVED from the grid** (Idea 19). `cache.py` now stores **`gfit_mask`/`geval_mask`**
  (single 20/80-of-online split) + **`blk_mask`/`blk_t0`/`blk_t1`** (2-min blocks, **staleness only**) +
  **`ctx_sess`** and **`ctx_blk`** (two context scopes). `config.json`: `block_min=2.0`, `min_blocks=2`.
  `cache_version → 3`.
- **Next:** re-run smoke (all 4 selftests + `cache --n 1 --with-decoders`) → **Block 2c**.

## Progress log 5 (pre-run hardening, 2026-10-03)
Four changes so the CSV carries everything analysis needs — **all must precede the grid run**, because
`run_grid.py` discards the adapter object and the predictions:

1. **`r2_vw`** added to `metrics.py` (+ 4 tests in `selftest_metrics.py`). **Finding: `r2_vw` ≡ `r2_all`
   (pooled) by algebraic identity**, and both ≡ sklearn `multioutput='variance_weighted'` = the FALCON/NoMAD
   metric. So our existing raw R² was *already* on FALCON's scale — verified numerically
   (|pooled − vw| = 1.4e-13), and contrasted against a *uniform* mean which gave −0.32 vs +0.88.
2. **Adapter metadata columns** (`form · label_use · aligned · causal · uses_targets · uses_decoder ·
   is_trainable`) via `run_grid.py::adapter_meta()`. Without these, P1/P2/P3/P5 cannot be grouped by family
   from the CSV. Static check: 12 registered == 12 configured, no key collisions.
3. **`refit_decoders` now includes `gru`** (was `ridge/wiener/kf_posvel/mlp`), so every active decoder has an
   oracle row and the OR/ZS framing is uniform. New `refit_available` column makes any gap self-explaining.
4. **`active_decoders` = all five.** Plus surfaced `r2_burnin_in`/`r2_burnin_out` (already computed in
   `cache.fit_decoders`, previously discarded → the overfit gap was unreportable) and the
   `eval_contiguous` bool that PLAN §5 requires for `lag_bins` validity.

⚠️ **Not fixed here:** P4 as written in `../findings/2026-10-03_adapter_grid_framing.md` ("ΔR² tracks drift
magnitude, not elapsed time") is **not testable by this grid** — `run_grid.py` emits `block_idx=-1`,
`t_start_min=NaN`, one row per cell, so there is **no time axis**. P4 belongs to `staleness.py`.
Also flagged: P2's two members differ in **label use** — `subspace` is clean (unlabeled) but
`centroid_proc` is **gray** (uses target identity) — so P2 must be evaluated on `subspace` with
`centroid_proc` reported separately.

5. **The L3 tracer caught a real pre-existing bug.** `out_affine` violated the PLAN §5 row-alignment
   rule: `np.linalg.lstsq` received `X=(n-L,3)` from a lagged decoder's `predict()` but a FULL-length
   `y` -> `LinAlgError: Incompatible dimensions`. It is the only adapter that consumes `y` in `fit`, and
   only lagged decoders (wiener L=5, mlp L=3, gru L=10) shorten predictions. Fixed in
   `adapters/output.py` (align `y` to the TAIL of `V`); a `LaggedDec` stub + regression tests added to
   `adapters/selftest.py` -- the old test passed an already-matching-length `y` with a full-length
   decoder stub, so it could never have seen this.
   Verified after the fix: tracer emitted **260 rows x 73 cols**, `L3_COL_CHECK: PASS`, zero `[FAIL]`,
   and on REAL data `r2_vw = 0.30065293128765525` vs `r2_all = 0.30065293128766213` -- the identity
   confirmed to 14 s.f.
