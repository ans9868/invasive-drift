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
**correction size:** `||T-I||_F · det(T) · median|scale-1| · rank(T) · n_params`
**diagnostics:** `obj_value · align_score · drift_energy_rowspace · drift_energy_nullspace`
**LR-2 (trainable only):** `epochs_run · loss_init · loss_final · loss_slope · grad_norm_mean/max ·
early_stopped · degenerate_fit · fit_time_s · r2_burnin`
**flags:** `degenerate · skipped · n_eval_samples`
**Storage:** Tier A table (~110k rows, few MB) — **keep all**; Tier B raw trajectories → leaders only.

## 6. Aggregate OUT of that table (never collapse in-pipeline)
median+IQR · mean+SD · **p10 (worst window)** · late-session R² · slope over windows ·
frac-of-sessions-improving · **fail fraction** · bimodality check.

## 7. Execution order (tracer bullet first)
| phase | deliverable | script |
|---|---|---|
| **P0** | skeleton, `config.yaml`, shared utils | — |
| **P1** | `adapters/` library + unit tests | `adapters/*.py` |
| **P2** | cached decoders + reference + splits | `cache.py` |
| **P3** | sufficient stats → transforms | `fit.py` |
| **P4** | apply + R² → long row | `evaluate.py` |
| **P5** | trainable adapters + LR-2 | `trainable.py` |
| **P6** | per-decoder cards, curves, racing | `analyze.py` |
| **P7** | row-space + alignment diagnostics | `fit.py` / `analyze.py` |

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

