# 17 — PLAN for experiment `27`: adapter grid × adaptation-data learning curves

**Status:** PLAN ONLY (no code yet). Supersedes the earlier fixed-window idea for `27`.
**Depends on:** `25_lowrank_normalizer.py`, `26_perspective_normalizer.py` (results below).
**Related:** `drafts/ideas.md` (Idea 8), `findings/2026-10-03_lowrank_normalizer.md`,
`findings/2026-10-03_normalizer_perspective.md`.

---

## 1. The question
We found a **small label-free win** (per-unit moment matching, +0.015 R²) and concluded that "more
complex" adapters don't help. **But that comparison gave every adapter the *same fixed window*** — which
structurally **penalizes high-capacity adapters**. This experiment replaces it with a **learning curve**:

> **How does R² depend on the amount of adaptation data, for each adapter (labeled vs unlabeled)?**
> Do the heavier adapters merely need **more data** (⇒ data-starved, would improve), or are they
> **capped by their objective** (⇒ genuinely no better)?

## 2. Why this is the right framing
- **For UNLABELED adapters, adaptation data is FREE** — no labels to run out of. So "needs more data" is
  **not** a weakness for them: give them the whole recording. A data-hungry *unlabeled* adapter that
  eventually beats moment-matching is a **real win**.
- **For LABELED adapters the curve is the calibration budget** — the practically important question
  ("how many labeled minutes to reach R² = X?").
- It converts a negative ("bigger didn't help") into an honest, quantitative statement
  ("bigger didn't help **at this data budget**; here is the curve and the slope").

## 3. What we already know (baselines to beat / bounds)
| item | value | source |
|---|---|---|
| `frozen` (no adapter) | **0.288** | `25`,`26` |
| `mom_diag` (per-unit mean+std) | **0.303** (+0.015, sd 0.016, 94% of sessions, p≈5e-12) | `26` |
| `mom_global` (1 scalar) | 0.281 (worse) | `26` |
| `cov_lowrank` (rank-5 cov) | 0.288 (0) | `25` |
| centroid Procrustes (lr2…lr7) | **negative** (full-rank catastrophic, −375) | `25` |
| `refit` (supervised) | **0.446** (headroom 0.158) | `25` |

## 4. Design

### 4.1 Session protocol (per session)
```
|<-- BURN-IN 20% -->|<------------------ ONLINE 80% ------------------>|
   frozen decoder            windows (8), each split by TIME:
   + reference (moments,     |<------ FIT POOL first 80% ------>|<- EVAL last 20% ->|
    centroid landmarks)                      grow N = f*|pool|, f in {0.1,0.25,0.5,1.0}
```
- **BURN-IN** fixes the scaler `(m0,s0)`, the **frozen decoder** (ridge, labels allowed = calibration),
  and the **reference** moments/landmarks.
- **Per window** we keep a **fixed EVAL set (last 20% by time)** and vary only the **FIT POOL** size `N`.
  Fitting on the *earlier* part and scoring on the *later* part makes every adapter **causal within the
  window** (also fixes the two-pass concern flagged in `26`).
- All adapters in a row see the **same N samples** and are scored on the **same eval set** → the only
  things varying are **type** and **N**.

### 4.2 Adaptation-data sweep
`N in {0.10, 0.25, 0.50, 1.00} * |fit pool|` (~4–25 k samples @ 20 ms).
Report per adapter: **R²(N)**, **slope** (ΔR² per decade of N), **N\* to beat `frozen`**,
**N\* to reach `refit`**, **saturation N**.

### 4.3 Adapter grid (the thing being compared)

**UNLABELED** (clean — no velocity, no target identity)
| # | name | form | params (d≈100) |
|---|---|---|---|
| 1 | `none` | frozen | 0 |
| 2 | `mom_global` | 1 global mean+std | 2 |
| 3 | `mom_diag` *(incumbent)* | per-unit mean+std | 2d ≈ 200 |
| 4 | `reg_ref` | least-squares map `Z_fit → Z_ref` | d²+d ≈ 10 100 |
| 5 | `cov_zca` | **full-rank** covariance match | d²/2 ≈ 5 000 |
| 6 | `cov_lowrank` | mean + rank-5 covariance | dk+k²/2 ≈ 525 |
| 7 | `cca` | shared subspace (k dims) | 2dk+k² ≈ 1 050 |

**LABELED** (uses velocity)
| # | name | form | params |
|---|---|---|---|
| 8 | `calib_out` | refit readout only | 2d+2 ≈ 202 |
| 9 | `sup_proc` | Procrustes with true correspondence | ~d(K−1) ≈ 700 |
| 10 | `refit` | full refit | 2d+2 ≈ 202 |

**GRAY** (uses *target identity*, not velocity — reported **separately**)
| # | name | form | params |
|---|---|---|---|
| 11 | `centroid_proc` | Procrustes on condition centroids | ~d(K−1) ≈ 700 |

**DEFERRED → `28`** (heavy; session subset + LR/epoch sweep)
| # | name | form | params |
|---|---|---|---|
| 12 | `deep_unlabeled` | autoencoder / OT / adversarial / NoMAD adapter | ~2dh ≈ 15–50 k |
| 13 | `deep_finetune` | fine-tune MLP decoder on the fit pool | ~10–30 k |

### 4.4 Notes
- **Unlabeled rows are reported at max-N** (all data) as their headline, plus the curve.
- Every row is tagged with **whether it touches velocity** — no silent leakage.
- `centroid_proc` uses direction labels derived from velocity (cued-target assumption) → gray, not clean.
- `calib_out` = for a *linear* decoder this equals `refit`; keep only if a nonlinear decoder is used.

## 5. Metrics / outputs
1. **R²(N) curves** per adapter (the "learning rate").
2. **Slope** per adapter (mid-range), in ΔR²/decade.
3. **N\* benchmarks:** to beat `frozen`; to reach `refit`.
4. **Saturation** point.
5. **Grouped summary:** best-unlabeled vs best-labeled, and % of refit headroom recovered.
6. Consistency: SD + fraction-of-sessions improving (as in `26`).

## 6. Decision rules (the point of the experiment)
- **Steep unlabeled curve, not saturated, above `mom_diag`** → give it all the data → **justifies `28`**.
- **Steep but saturates below `mom_diag`** → capacity isn't the issue; the **objective** is → stop.
- **Flat at any N** (like `cov_lowrank`) → genuinely capped → stop.
- **Labeled curves** tell us the achievable ceiling and the calibration cost to approach it.

## 7. Cost
- Rows 1–11 × 4 N-values × 8 windows × 53 sessions → all linear/algebraic, **~minutes** on `cpu_short`
  (compare `25` = 131 s). `--mem` **4 G** (16 G is hugely over-provisioned).
- `28` (deep) is the expensive part (≈2 500+ fits) → session subset (15–20) + fewer N-values, separate job.

## 8. Risks / caveats
- **Eval set shrinks** if a window is short; require a minimum window size and skip otherwise.
- **Moment estimates get noisy at small N** → the `mom_diag` curve will dip at f=0.10, which is *expected*
  and is itself informative (it sets the minimum data for the incumbent).
- **Causal split ≠ deployment**: it removes the two-pass concern, but latency within a window remains
  (you must observe the fit pool before decoding the eval set).
- **`reg_ref` / `cov_zca` are already O(d²)** — "bigger" does not require deep.
- Deep rows need **LR/epoch tuning**, else unfairly judged → handled in `28`.

## 9. Implementation sketch (when built)
- `mvp/scripts/27_adapter_curves.py` + `.sbatch` (`id_ac`), `--mem=4G`, `--time=00:40:00`.
- Reuse `load/exp_filt/r2/procrustes` from `25`; add a dict of `fit(Zfit,...)` / `apply(Z,...)` adapters.
- Emit a tidy table: `adapter, N_frac, meanR2, sd, frac>frozen` + derived slopes/benchmarks.
- Findings → `findings/<date>_adapter_curves.md`.

## 10. What this plan replaces
- The earlier `27` design (adapters at a **single fixed** window) → superseded by the **N-sweep grid**.

