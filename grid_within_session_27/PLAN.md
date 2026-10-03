# PLAN — `grid_within_session_27`: decoder × objective grid with adaptation-data learning curves

**Status:** PLAN (authoritative). Supersedes `drafts/17_plan_27_adapter_curves.md` (kept as the
discussion trail).
**Scope:** **within-session** adaptation only. Cross-session / meta-learned adapters = Idea 11 (later).

---

## 0. The question (and the flaw it fixes)
Our earlier adapter comparison used **proxy objectives** (feature distribution matching) that are **not
aligned with the end-to-end decoder**, and it gave every adapter the **same fixed window** (penalizing
high-capacity ones). This experiment fixes both:

> **For each decoder, how does R² depend on the amount of adaptation data — and does a
> DECODER-ALIGNED objective beat a decoder-agnostic proxy?**

## 1. Where things live
**Repo:** `grid_within_session_27/` (drivers) + `adapters/` (library) + `findings/` (results).
**HPC scratch:** `$SCRATCH/invasive-drift/{repo,data,artifacts,results,logs,tmp}`.
Bulk/derived → scratch; small summaries + figures → git.

## 2. The grid (4 axes)
| axis | values |
|---|---|
| **Decoder** | ridge · wiener · **kf_posvel** · mlp · gru (all frozen at burn-in; cached) |
| **Objective** | O1 mom_diag · O2 zca · O3 reg_ref · O4 cca · **O5 out_mom** · **O6 out_mlp** · **O7 null_proj** · O8 dynamics · O9 cycle · (O10–12 deep → later) · L1 refit · L2 finetune · L3 out_affine · L4 sup_proc |
| **Label use** | unlabeled(clean) · labeled · **gray(target identity, reported separately)** |
| **Data budget N** | `f in {0.10, 0.25, 0.50, 1.00} x |fit pool|` |
| **Output** | velocity(2-D) · direction(circular) · direction(8-class) · both |

**Decoder-alignment flag** is a first-class column: `agnostic` (O1–O4, O9) vs `aligned` (O5–O8).

## 3. Per-session protocol
```
|<-- BURN-IN 20% -->|<---------------- ONLINE 80% ---------------->|
   frozen decoders          window: |<-- FIT POOL 80% -->|<- EVAL 20% (fixed) ->|
   5 cached models                  grow N = f*|pool|   (identical eval rows for all cells)
   + reference stats
```
- **Fitting on the earlier part, scoring the later part** → causal within the window.
- **Eval set is fixed** across every adapter and every N → all comparisons **paired**.

## 4. The three learning rates (all logged)
| id | meaning | applies to |
|---|---|---|
| **LR-1** | R² vs **N** (data-scaling slope) | every cell |
| **LR-2** | **training dynamics** (loss curve, grad norm, epochs, early-stop) | trainable (mlp, gru, trainable adapters) |
| **LR-3** | adapter-fit convergence | trainable adapters only |

## 5. Record schema (long format — one row per cell × window)
**keys/identity:** `session_id · window_idx · t_start_min · n_units · decoder · objective · form ·
label_use · aligned · causal · output · N_frac · N_samples · seed`
**metrics — velocity:** `r2_all · r2_vx · r2_vy · corr_vx/vy · mse · bias_vx/vy · slope_vx/vy ·
mse_bias2 · mse_var · lag_bins · r2_frozen · r2_refit`
**metrics — direction/speed (Option A, ~free):** `ang_err_mean_deg · ang_bias_deg · ang_abs_err_deg ·
ang_resultant_R · speed_ratio · speed_corr · dir_acc8`
**neural context (cheap, explains drift):** `rate_mean · rate_median · frac_silent · active_units ·
mean_pairwise_corr · pc1_var · eff_dim · subspace_angle_deg`
**task covariates (per window):** `speed_mean · moving_frac · dir_coverage` (n directions present)
**ceiling anchors:** `r2_refit` (optimistic oracle) · `reliability` (split-half of TRUE kinematics)
**overfit gap:** `r2_train · fit_time_s`
**temporal:** `lag_bins` (pred−true cross-corr lag) · `metric_slope_over_windows`
**cross-decoder agreement / uncertainty:** `pred_corr_mean` (pairwise across decoders on the same rows) ·
`ens_disagree` (std across decoders) · `err_corr_mean` (are the ERRORs aligned?) · `r2_consensus`
**KF self-uncertainty:** `kf_post_var_mean` · `kf_post_var_growth` (posterior covariance over the window)
**non-neural baselines (anchor the R² scale):** `r2_persist` (v_t ≈ v_{t−1}) · `r2_mean` (const) ·
`r2_target` (mean velocity for the cued direction)
**correction size:** `||T-I||_F · det(T) · median|scale-1| · rank(T) · n_params`
**diagnostics:** `obj_value · align_score · drift_energy_rowspace · drift_energy_nullspace`
**LR-2 (trainable only):** `epochs_run · loss_init · loss_final · loss_slope · grad_norm_mean/max ·
early_stopped · degenerate_fit · fit_time_s · r2_burnin`
**flags:** `degenerate · skipped · n_eval_samples · **eval_contiguous** (bool)`
**ROW-ALIGNMENT RULE (critical):** `wiener` (L=5), `mlp` (L=3) and `gru` (L=10) return **len(X) − L**
predictions. Every comparison must align targets to the **TAIL**: `y_eval = y_eval[len(pred)-len(y_eval):]`.
Get this wrong and predictions are silently compared to the wrong rows. (Caught by the L1b smoke test.)
**CI policy:** CIs are computed **at analysis time** by grouping the per-window rows (never collapse
in-pipeline).
**Storage:** Tier A table (~110k rows, few MB) — **keep all**; Tier B raw trajectories → leaders only.

> `lag_bins` is **only valid when `eval_contiguous == True`** (eval rows form one contiguous time block,
> as in the current design). Never interpret a lag from a shuffled/random eval split.

## 5b. Metric definitions (LOCKED — Block 1, first grid search)
Notation: `y` = true velocity `(n,2)`, `ŷ` = prediction, on the **eval rows only**; `moving` = rows with
speed above the 60th percentile; `Δθ` = angle wrap of `θ̂ − θ` into `[−180°, 180°)`.

**Velocity**
- `r2_all = 1 − Σ‖ŷ−y‖² / Σ‖y−ȳ‖²` (pooled); `r2_vx`, `r2_vy` = same per dimension.
- `corr_vx/vy` = Pearson(pred, true) per dimension.
- `mse = mean_rows Σ_d (ŷ−y)²`; `bias_vx/vy = mean(ŷ−y)`.
- `slope_vx/vy` = OLS slope of **true on predicted** (gain), per dimension.
- `mse_bias2 = ‖mean_rows(ŷ−y)‖²`; `mse_var = mean_rows ‖(ŷ−y) − mean(ŷ−y)‖²`
  → **identity: `mse = mse_bias2 + mse_var`** (verified in the selftest).
- `lag_bins = argmax_{l∈[−20,20]} corr(ŷ_t, y_{t+l})`; **positive ⇒ the output lags the truth**.
- `r2_frozen`, `r2_refit` — computed by the harness on the **same** eval rows.

**Direction / speed** (on `moving` rows)
- `θ = atan2(y_y, y_x)`, `θ̂ = atan2(ŷ_y, ŷ_x)`.
- `ang_bias_deg = deg( atan2(mean sin Δθ, mean cos Δθ) )` — the **systematic rotation** (drift signature).
- `ang_err_mean_deg = mean |Δθ|`; `ang_abs_err_deg = median |Δθ|` (robust).
- `ang_resultant_R = |mean e^{iΔθ}|` (concentration); `circ_std_deg = deg( sqrt(−2 ln R) )`.
- `speed_ratio = median( |ŷ| / (|y|+ε) )`; `speed_corr = Pearson(|ŷ|, |y|)`.
- `dir_acc8` = accuracy of binning `θ̂` into 8 target bins vs `θ`'s bin (**velocity-derived**
  correspondence → flagged).

**Neural context** (per window, from `cache.py`)
- `rate_mean`, `rate_median` (Hz over units), `frac_silent` (rate < 0.5 Hz), `active_units`.
- `mean_pairwise_corr` = mean off-diagonal pairwise correlation of unit rate vectors
  (**subsample 1500 fit-pool rows, fixed seed**).
- `pc1_var` = variance fraction of PC1; `eff_dim = (Σλ)² / Σλ²` (participation ratio).
- `subspace_angle_deg` = mean principal angle between the window's top-k PCA subspace and the reference.
- `speed_mean`, `moving_frac`, `dir_coverage` = # direction bins with ≥ 20 samples.

**Ceiling / overfit**
- `r2_refit` = decoder fit on the **eval** rows (optimistic oracle).
- `reliability` = **unit split-half** decoding ceiling: split units randomly in half, fit ridge per half on
  the fit pool, predict eval, correlate the two predictions (mean over vx,vy).
- `r2_train` = R² of the frozen decoder on the **fit pool**; `fit_time_s`.

**Agreement / uncertainty** (computed per **adapter × window**, **across the 5 decoders**, then broadcast)
- `pred_corr_mean` = mean over decoder pairs of corr(pred_A, pred_B) on eval.
- `ens_disagree` = mean over rows of std across decoders (per dimension, then averaged).
- `err_corr_mean` = mean over decoder pairs of corr(err_A, err_B), `err = pred − true`
  (**high ⇒ the models fail the same way = common cause**).
- `r2_consensus` = R² of the mean-across-decoders prediction.
- `kf_post_var_mean` = mean_t trace(P_t); `kf_post_var_growth` = slope of trace(P_t) over the window.

**Non-neural baselines** (need only true velocity + `dirbin`)
- `r2_mean` (predict the fit-pool mean) — the 0 floor.
- `r2_persist_lag1` (ŷ_t = y_{t−1}) **and** `r2_persist_lag12` (ŷ_t = y_{t−12}, ≈240 ms) — **both**.
- `r2_target` = R² of predicting the **reference mean velocity of the row's direction bin**;
  **flagged `target_src="velocity_bin"`** (mildly self-referential; not a trial target).

## 6. Aggregate OUT of that table (never collapse in-pipeline)
median+IQR · mean+SD · **p10 (worst window)** · late-session R² · slope over windows ·
frac-of-sessions-improving · **fail fraction** · bimodality check.

## 7. Execution order (tracer bullet first) — with checkpoint **BLOCKS**
> **Blocks** are discussion gates: we stop, look, and agree before proceeding.

| step | deliverable | script | gate |
|---|---|---|---|
| **Block 1** | lock metric definitions (§5b) + grid axes | — | ✅ done |
| **P0** | skeleton, `config.json` | — | ✅ done |
| **P1** | `adapters/` library + `selftest.py` (ALL PASS) | `adapters/*.py` | ✅ done |
| **P2-tracer** | cache 1 session | `cache.py --n 1` | ✅ done |
| **Step 0.5** | `metrics.py` + `selftest_metrics.py` + per-window context in `cache.py` + KF `trace(P)` + smoke update → re-run smoke | new | **Block 2** |
| **Step 0** | non-neural baselines (`r2_persist*`, `r2_mean`, `r2_target`) vs ridge, 53 sessions | `baselines.py` | **Block 3** |
| **P2-scale** | cache all 53 sessions | `cache.py --n 0` | — |
| **P3** | sufficient stats → transforms | `fit.py` | — |
| **P4** | apply + full metric row (incl. agreement, KF, direction) | `evaluate.py` | — |
| **P5** | trainable adapters + LR-2 | `trainable.py` | — |
| **P6** | per-decoder cards, curves, racing, CIs | `analyze.py` | **Block 4** |
| **P7** | row-space + alignment diagnostics | `fit.py` / `analyze.py` | — |

**Block 3 decides the headline metric:** if `r2_persist*` ≫ decoder R², we switch from raw R² to
**skill-over-persistence** (`Δskill`, `R²_innovation`) **before** building P3/P4.

**Widening:** 1 → 3 → 10 → 53 sessions; ridge → +linear → +MLP/KF → +GRU; N=1 → full sweep.

## 8. Smoke tests (5 levels)
- **L0 unit (synthetic):** adapters undo a known transform; identity ~ frozen; no-drift ⇒ no harm.
- **L1 contract:** shape/dtype asserts; **eval set identical across N and cells**; determinism by seed.
- **L2 tracer:** `--n 1 --nwin 2 --decoders ridge --N 1.0` finishes in seconds, emits a valid row.
- **L3 sanity band:** `frozen <= adapter <= refit` (±slack); flag NaN/Inf; no silent blow-ups.
- **L4 leakage/controls:** deliberate leak probe; **`shuffled_ref` negative control** (permanent cell);
  **`identity` no-op cell** must equal `frozen`.
- **L5 scale:** 1→53 with per-stage wall time + **peak RSS** printed (no `squeue` polling).

## 9. Factorization (why this is cheap)
1. **Fit-then-evaluate:** agnostic transforms fit **once**, evaluated against **all 5 decoders** for free.
2. **Sufficient statistics:** running `Sz, Szz^T, Szy^T, Sf(z), Sf(z)f(z)^T ...` → **all N-prefixes free**.
3. **Decoders cached once/session** (GRU = 53 trainings, not ×windows×N).
4. **Shared eval rows** → paired statistics for free.
Naive ≈ 13.8k fits → factorized ≈ **~1.7k cheap fits + 53 GRU trainings**.

## 10. Racing (intelligent search)
S0 screen (alignment diagnostic) → S1 all cells on ~15 sessions @max-N → S2 top-third on all 53 →
S3 full N-curves + paired tests on survivors.

## 11. Parallelization
Job array, **1 task/session**, `results/raw/<jobid>_<session>.csv`, `--mem=4G`, `cpu_short`;
single cheap merge/analyze job; **file sentinels, never `squeue` loops**.

## 12. Decision rules
- **aligned >> agnostic** (esp. O5/O7) → the misalignment was the flaw → pursue.
- steep unlabeled curve never saturating → give it all the data → justify deep / `28`.
- flat at all N (like `cov_lowrank`) → the objective is the cap → stop.
- **labeled column** = ceiling + calibration budget.

## 13. Deliverables
Per-decoder **cards** (LR-1 curves, LR-2 traces, headroom bars, distribution panels) · cross-decoder
summary · `findings/<date>_adapter_grid.md` · committed summary tables + figures.

## 14. Open items / deferred
- O10–O12 (deep/OT/adversarial adapters) → `grid_deep_28`.
- Idea 11 (cross-session **meta-learned adapter**) → later, needs unit matching.
- Gray row (`centroid_proc`, target identity) → reported separately, not in clean unlabeled.

