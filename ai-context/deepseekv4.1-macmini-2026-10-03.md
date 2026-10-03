# AI session context — invasive-drift (exp 27 grid)

**File:** `deepseekv4.1-macmini-2026-10-03.md`
**Date:** 2026-10-03
**Model/host:** assistant on a macOS "Mac mini" host, driving a Torch (HPC) login node over SSH.
**Purpose:** a complete handoff of this working session — what the project is, every result, every design
decision, every bug caught, and exactly what to do next — so a fresh session (or a human) can pick up
with no loss.

---

## 0. One-paragraph summary
We are studying **cross-session / within-session decoder degradation** in intracortical BCI. Using the
Perich & Miller macaque M1 dataset (DANDI `000688`, sub-C, 53 centre-out sessions), we established that
within-session decoder decay is **representational drift** (a slow, low-dimensional, non-rigid
*deformation* of the population code) — real, but **largely unpredictable**. We are now building an
engineering-rigorous **adapter grid** (`grid_within_session_27`) to test one specific, self-inflicted
flaw: our earlier adapters optimised **proxy objectives** (feature-distribution matching) that are **not
aligned with the end-to-end decoder** (a linear decoder only "sees" a ~2-D subspace of feature space).

## 1. The project in one page
- **Object of study:** a *decoder* (spikes → cursor velocity) whose ideal weights drift over time, so a
  frozen decoder degrades. Two timescales:
  - **Within-session** (~hours): the focus of all work so far.
  - **Cross-session** (days/weeks): deferred (Idea 11), needs unit matching.
- **Goal of the MVP:** (a) fix the decoder, (b) build a health(t) axis, (c) forecast degradation
  **honestly** — i.e. only claim skill over a strong baseline.
- **Key finding to date:** within-session decay is **representational** (tuning rotation), **not**
  gain, **not** unit loss, **not** behaviour. And it is **not forecastable** by anything we have tried.

## 2. The intellectual arc
1. Build a decoder zoo → nonlinear decoders win.
2. Confirm within-session decay → yes (6/7 sessions).
3. Diagnose the cause → **representational (tuning rotation)**, r≈0.53.
4. Probe waveforms → stable within-session (so not hardware); **match across days** (0.996) → identity.
5. Try to **forecast** the decay → failed at scale; only a thin level gain over hist-slope.
6. Ask **what the drift geometrically is** → smooth, low-D, **not extrapolatable**; **non-rigid**.
7. Ask **is the drift axis stable across days** → **NO** (random).
8. Ask **is the drift low-rank** → **YES** (PC1+2 = 0.80; confirmed with an MLP).
9. Ask **can we correct it** → landmark rotation fails; **label-free per-unit moment re-matching gives a
   small real win (+10% of oracle headroom)**.
10. **New flaw identified (this session):** those adapters optimise **proxy** objectives. We are building
    a proper grid that (a) crosses adapters × decoders × data-budget, (b) tests **decoder-aligned**
    objectives, and (c) reports **learning curves**, not single numbers.

---

## 3. Results so far (dataset = Perich `000688` sub-C, 53 sessions unless noted)

### 3.1 Decoder zoo (`mvp/scripts/09_decode_zoo.py`)
Intra-session R² (train first 80% → test last 20%), mean over 53 sessions:
**GRU 0.607 · MLP 0.576 · Wiener(L5) 0.404 · KF-posvel 0.393 · KF-vel 0.383 · Ridge 0.357.**
Nonlinear ≫ linear; history helps (Wiener > Ridge); Gilja-2012's position-in-state trick reproduces
(KF-posvel ≥ KF-vel). **Decoder of record = GRU (or MLP).**

### 3.2 Within-session drift is representational (`05`,`06`,`15`,`16`)
- Frozen decoder **decays** over the session: **6/7** scanned sessions; e.g. R² 0.12 → 0.05 over ~1 h;
  scan mean 0.238 → 0.131, slope ≈ **−0.0022 / min**.
- **corr(decay, functional tuning rotation) = +0.526**; **corr(decay, weight-drift) = +0.525** — the two
  agree, so it is **NOT a ridge null-space artefact**.
- **Not** gain (**firing rate flat**), **not** unit loss (**waveforms stable**), **not** behaviour
  (**r = −0.433, wrong sign**).
- Recalibration helps only a little on average (0.317 → 0.332) but a lot in some sessions
  (CO-20160914: 0.06 → 0.40).

### 3.3 Waveform probes (`12`,`14`)
- **Within a session:** waveform stability **0.997–0.999**; waveform drift median **0.000** / max 0.016;
  alive units 72–74; amplitude **417 → 388 µV (−7 %)**.
- **Across sessions:** waveform matching **median corr ≈ 0.996**, **63–100 %** of units matched →
  **Perich waveforms unlock cross-session unit identity.**

### 3.4 Forecasting the decay — honest negative (`10`,`13`) + method validation (`11`)
- At **6 sessions / 45 rows** it looked great: **rate Δ+0.50**, accel Δ+0.25.
- **Injection test (`11`):** the real bar is the **historical slope** (it often beats const-accel:
  0.597 vs 0.510); **acceleration adds nothing** and **contaminates the slope** (bias −0.056);
  **steps smear the model** (−0.079 vs −0.676); slope/accel intervals **over-cover** (100 %).
- At **53 sessions / 446 rows** (`13`) the small-n result **EVAPORATED**:
  next-block **LEVEL**: persist 0.000 · hist_slope +0.146 · model +0.185 (modest **+0.04** over the bar);
  next-block **RATE**: **no skill**. → **rate of degradation is NOT forecastable.**


### 3.5 What the drift geometrically is
- **`17` drift dynamics:** **smooth** (consecutive weight-cosine **0.701**), **low-dim** (**PC1 0.35**),
  total drift cosine 0.665, **but NOT extrapolatable** — extrapolation **0.503** vs persistence **0.697**
  → a **smooth random walk on a low-D manifold**.
- **`18` predicting the next decoder:** **rot_lr 0.721**, **shrink 0.701** beat **persistence 0.661**;
  rot 0.663; trend 0.464; state 0.464; var 0.352. Functionally **none** beat persistence
  (median R² ≈ 0.93 all; a 2–3 % tail of catastrophic blocks).
- **`19` the drift axis is NOT stable across sessions** (unit-free, velocity space): end-to-end cos
  **+0.050**, trajectory PC1 cos **+0.033** ≈ random. Within a session it **is** low-D (**PC1 0.51**,
  step-changes 0.56) but that mode is **random across sessions** (+0.053). Drift **rate**: **duration_min
  rho = −0.471 (p<0.001)**; n_units −0.014; session_index −0.018; start_hour +0.234 (n.s.).
  `|driftV|` vs duration **+0.316** → **drift SATURATES**.
- **`20` manifold motion:** **NOT rigid** (rigid 0.88 ≈ identity 0.89) → **deformation**; *reach* subspace
  rotates **56°**, *rest* **0°**; corr(rigid-residual, R²) **−0.376**. ⚠️ affine (0.00) **degenerate**.
- **`21` "piano" (repeat the same movement):** chord drift **0.72**; a **shared mode ≈45 %**; persistence
  **0.809** beats rotation 0.765; landmark alignment **HURTS** (0.289 → **0.058**; refit 0.408).
- **`22` pairwise gaps:** only **15 %** monotone; only **37 %** of those widen; median change **−0.03**;
  extrapolation skill **−1.750**; lag: 22 % monotone, median |Δlag| 2.9 bins → **random wobble**.
- **`23` LOW-RANK GATE — PASSES:** velocity-field steps: **PC1 0.56, PC1+2 0.80, rank80 2.5/16**;
  weight-space PC1 0.44. Volatility: vs block +0.17, **lag-1 autocorr +0.04 (memoryless)**;
  corr(vol,R²) **−0.239**, corr(vol,R²_next) **−0.162** (weaker → **no lead**), corr(vol,ΔR²) +0.034.
  → **entropy gives no early warning.**
- **`24` MLP confirms the gate:** ridge PC1 0.55 / PC1+2 0.79 vs **MLP PC1 0.61 / PC1+2 0.82** → the
  low-rank structure is **NOT a linear artefact**. (Caveat: per-block MLP is unstable/degenerate.)

### 3.6 The adapter "win" — small, real, mechanistically humble (`25`,`26`)
- **Ladder (mean test R²):** frozen **0.288** · full-rank landmark **−375** · lr2 0.283 / lr3 0.275 /
  lr5 0.262 / lr7 0.252 (all **negative**) · **A1 per-unit moment match 0.303** · A2 low-rank cov 0.288 ·
  **refit 0.446 (in-sample oracle)**.
- → **landmark rotation FAILS** (drift is a deformation, not a rotation). **Label-free per-unit moment
  re-matching WORKS a little.**
- **`26` perspective:** A1 − frozen: **mean +0.0153, sd 0.0159, median +0.0101, 94 % of sessions,
  p = 5.5e-12**. **std_only −13 843** (unstable); **unit-SHUFFLED reference = 0.303 — identical!** →
  the win does **not** use per-unit identity; it is a **generic per-unit variance normalisation
  (gain whitening)**. Global scalar 0.281; behaviour rho −0.167.
- **Verdict:** real and consistent but **tiny** (~+0.015 R² ≈ 5 % relative; 10 % of *oracle* headroom)
  and it corrects per-unit **gain**, **not** the drift's direction.


---

## 4. Where things live

**Repo** (`github.com/ans9868/invasive-drift`, public; local root `/Volumes/CrucialX6/Home/projects/invasive-drift`):
```
drafts/          thinking notes 00–17 + ideas.md (Ideas 1–15)
findings/        16 dated result files (one per finding) + README index
mvp/             the ORIGINAL pipeline (scripts 00–26, decoders.py, requirements.txt) — frozen history
adapters/        NEW library: adapter classes + selftest.py
grid_within_session_27/  NEW: PLAN.md (authoritative), README.md, config.json, cache.py,
                         metrics.py, selftest_metrics.py, selftest_cache.py, smoke.sbatch, summaries/
ai-context/      this handoff document
WORKFLOW.md      operational rules (no `rm`, SLURM etiquette, scratch layout)
```

**Torch scratch** (`$SCRATCH/invasive-drift/`): `repo/` or the clone root · `data/perich/sub-C/*.nwb` (read-only)
· `artifacts/perich_subC/` (derived, regenerable) · `results/` · `logs/` · `tmp/` · **`trash/`** ·
**`temp-trash/`** (where old caches are *moved*, never deleted). Repo clone on Torch = `$SCRATCH/invasive-drift`.

## 5. The grid experiment (`grid_within_session_27`) — design

### 5.1 The flaw being fixed
Our earlier adapters (`25`,`26`) minimised **proxy objectives** (feature-distribution matching) that are
**not aligned with the end-to-end decoder**, and were given a **fixed window** (unfair to high-capacity
adapters). Reminder: a **linear** decoder has only **2 outputs**, so only a **~2-D subspace** of feature
space reaches it — anything an adapter does outside that row space is invisible or harmful.

### 5.2 The grid axes
| axis | values |
|---|---|
| **Decoder** | ridge · wiener · kf_posvel · mlp · gru (all frozen at burn-in, cached to disk) |
| **Objective / adapter** | O1 mom_diag · O2 zca · O3 (subsumed by zca) · O4 subspace(SA) · **O5 out_mom** · O6 out_mlp · **O7 null_proj** · O8 dynamics · O9 cycle · L1 refit · L2 finetune · L3 out_affine · L4 sup_proc (+ controls: `identity`, `shuffled_ref`) |
| **Label use** | unlabeled (clean) · labeled · **gray** (`centroid_proc`, target identity — reported separately) |
| **Data budget N** | `f ∈ {0.10, 0.25, 0.50, 1.00} × fit-pool` (PREFIX of the fit pool → causal) |
| **Output** | velocity(2-D) · direction(circular) · direction(8-class) · both (last two are Ideas 12/13) |
| **Decoder-alignment flag** | `agnostic` (O1–O4, O9) vs `aligned` (O5–O8) ← **the headline contrast** |

### 5.3 Per-session protocol
```
|<-- BURN-IN 20% -->|<---------------- ONLINE 80% ----------------|
   frozen decoders        window (FIXED ~3 min): |<-- FIT POOL 80% -->|<- EVAL 20% (fixed) ->|
   + reference stats                              adapter is fit here     scored here
```
- Adapter fit on the **earlier** part, scored on the **later** part → **causal within the window**.
- **Eval set is identical** for every adapter and every N → all comparisons are **paired**.
- **Windows are fixed-DURATION (~3 min), not fixed-count** → short and long sessions comparable;
  `W = online_len / 3 min` (≈4–10 per session).

### 5.4 The three learning rates (all logged)
**LR-1** data-scaling (R² vs N) · **LR-2** training dynamics (loss curve, grad norm, epochs, early-stop)
for trainable models · **LR-3** adapter-fit convergence.

### 5.5 The STALENESS instrument (added this session)
- **Rolling** (HEADLINE, fair by construction): fixed training size `w` blocks, varying **gap** in minutes
  → *"how fast does a decoder go stale?"*
- **Expanding** (calibration-budget curve): train on `blocks[0..j-1]`, test `j`, x-axis = calibration
  minutes, **floor at j ≥ 2** → *"how much calibration do you need?"*
- **Forward-only** (drop the backwards cells — not deployable). Scope: frozen + 1–2 adapters only.
- **≥5 blocks** required for this instrument (the grid keeps short sessions).
- **TG matrix** (fit on block i, test block j for all i,j) is the generalisation; we use the two schemes
  above rather than the full matrix.

### 5.6 Session-length handling (no second pipeline!)
Because rows are stored **per cell × window**, length is a `GROUP BY`, not a new pipeline. Tags written
into every row: **`session_minutes` · `n_windows` · `short_recording` (`n_windows < 5`)**.
Three analysis-time views: all sessions · `n_windows ≥ 5` · **length buckets (<15 / 15–25 / >25 min)**
(the confound check). Plus a **support curve** (N at each x-value).
Length is a **real moderator** — `19` found drift rate ≈ 1/duration (rho = −0.47; drift saturates).


---

## 6. Metrics (locked — full definitions in `PLAN.md` §5b)

**Velocity:** `r2_all · r2_vx · r2_vy · corr_vx/vy · mse · bias_vx/vy · slope_vx/vy` (gain) ·
`mse_bias2 · mse_var` (**identity: mse = mse_bias2 + mse_var**) · **`lag_bins`** (+ve = output lags).

**Direction/speed** (Option A — *free*, derived from the same velocity prediction):
`ang_bias_deg` (circular mean of the error = **the drift signature**) · `ang_err_mean_deg` ·
`ang_abs_err_deg` (median) · `ang_resultant_R` · `circ_std_deg` · `speed_ratio` · `speed_corr` · `dir_acc8`.
*(Direction **heads** — circular regression / 8-class — are deferred: Ideas 12 & 13.)*

**Neural context (per window, from `cache.py`):** `rate_mean · rate_median · frac_silent · active_units ·
mean_pairwise_corr (1500-row subsample) · pc1_var · eff_dim · subspace_angle_deg · speed_mean ·
moving_frac · dir_coverage`.

**Decoder calibration context:** `r2_burnin_in` (fit on burn-in, scored on burn-in — **INFLATED**) ·
**`r2_burnin_out`** (fit on the **first half of burn-in**, scored on the **second half** — time split,
honest) · their difference = the **overfitting gap**.

**Ceiling / overfit:** `r2_refit_oracle` (fit on eval, scored on eval — a BOUND, not achievable) ·
**`r2_refit_out`** (fit on the window's fit pool, scored on the eval tail — realistic recalibration) ·
`reliability` (**unit split-half** decoding ceiling) · `r2_train` · `fit_time_s`.
*(Note: `25`'s "0.446" was `r2_refit_oracle`, so its "headroom" was an **oracle** headroom.)*

**Agreement / uncertainty** (per adapter × window, across the 5 decoders):
`pred_corr_mean · ens_disagree · **err_corr_mean**` (high ⇒ the models fail the same way = common cause) ·
`r2_consensus` · **`kf_post_var_mean` / `kf_post_var_growth`** (KF posterior covariance).

**Non-neural baselines:** `r2_mean` (floor) · **`r2_persist_lag1`** and **`r2_persist_lag12` (~240 ms)** ·
`r2_target` (reference mean velocity of the row's direction bin; flagged `target_src="velocity_bin"`).

**Flags:** `degenerate · skipped · noop · n_eval_samples · eval_contiguous`.

⚠️ **ROW-ALIGNMENT RULE (critical):** `wiener` (L=5), `mlp` (L=3), `gru` (L=10) return **len(X) − L**
predictions → **always** align targets to the TAIL: `y = y[len(pred)-len(y):]`.

## 7. Record schema (long format — one row per cell × window)
Keys: `session_id · window_idx · t_start_min · session_minutes · n_windows · short_recording · n_units ·
decoder · objective · form · label_use · aligned · causal · output · N_frac · N_samples · seed`
then all metric groups above + `scheme · gap_min · calib_min` (for staleness) + correction size
(`||T−I||_F · det(T) · rank(T) · n_params`) + diagnostics (`obj_value · align_score ·
drift_energy_rowspace/nullspace`).
**CI policy:** CIs are computed **at analysis time** by grouping the per-window rows — never collapse in-pipeline.

## 8. Bugs the smoke tests caught (this is the value of the discipline)
1. **`insert_line` splice silently emptied `MomDiagSelf`** (kept only `name`) — syntactically valid,
   semantically broken. Caught by `mom_diag_self_is_non_causal` + `matches_mean`.
2. **`cca` was ill-posed** — CCA needs **paired** samples; unpaired window-vs-reference provides none
   (it degenerates to ZCA). → replaced by **subspace alignment (`subspace`)**; SA verified
   49.3° → 17.1°.
3. **`cache.py` unpacked the *left* singular vectors** (`U`) instead of the right (`Vh`) in the subspace
   -angle block → a (5305×k) @ (71×k) mismatch.
4. A selftest put the test direction **exactly on a bin edge** (45°) → `dir_acc8 = 0.56` instead of 0.
   Fixed by using 22.5°.
5. **Lagged decoders drop L rows** → predictions were compared to the **wrong target rows** (would have
   silently corrupted every grid cell). Fixed with `align_tail()`.
6. A selftest **asserted `in ≥ out`** — not a mathematical invariant (the two scores use *different* test
   rows, and `1 − MSE/Var` is variance-sensitive). Replaced by reporting the gaps.


---

## 9. Decisions log (everything we settled this session)
1. **Metric definitions LOCKED** (PLAN §5b) before any code — incl. the circular-safe direction stats.
2. **Add 8+ metric groups** (direction/speed, error decomposition, temporal, neural context, covariates,
   ceiling, overfit, agreement/KF, non-neural baselines). All ≈free because rows are per-window.
3. **`r2_burnin_in` AND `r2_burnin_out`** (time-split held-out half) — the user asked for both.
4. **`r2_refit_oracle` AND `r2_refit_out`** — because `25`'s headroom used an in-sample oracle.
5. **Merge P3+P4** into one per-session `run_grid.py` (fit and apply in the same process).
6. **Cache the frozen decoders to disk** — **pickle** (`<session>.decoders.pkl`) + **config-hash guard** +
   `cache_version` in the hash key. Round-trip **proven byte-identical** for all 5 decoders.
7. **No racing** — compute is free; run the full grid.
8. **Flag-and-keep** rows (no-op / too-small), never silently drop.
9. **Gray adapters** (`centroid_proc`) excluded from "clean unlabeled" aggregates, own panel.
10. **`reg_ref` subsumed by `zca`**; **`cca` → `subspace` (SA)** — both documented in code.
11. **Windows become fixed-DURATION (~3 min)**, not fixed-count 8 — for cross-session comparability.
12. **Staleness = rolling (headline) + expanding (calibration budget), forward-only**, ≥5 blocks,
    scope = frozen + 1–2 adapters only.
13. **Session length = 3 tags + 3 analysis-time views + a support curve** — no second pipeline.
14. **N derived from duration** (not hard-coded 5); sensitivity check at 2 / 3 / 4.5 min planned.
15. **Ideas 6–15 added to `drafts/ideas.md`** (waveform cause layer; moving manifolds; rotation
    regularizer; cross-day axis; piano; meta-learned adapter; direction head; multi-task; uncertainty;
    non-neural baselines).
16. **Old caches are MOVED to `temp-trash/`** (repo root, gitignored) — never deleted.

## 10. Current status (where we stopped)
**DONE:** Block 1 (definitions) · P0 (skeleton/config) · **P1** (`adapters/` library, 12 closed-form
adapters, ALL PASS) · **P2-tracer** (cache 1 session) · **Step 0.5** (`metrics.py` + selftests + per-window
context + KF `trace(P)`) · **Step 0.6/0.6b** (decoder pickle cache + `r2_burnin_in/out` + hash bump).
All smokes green. `id_inspect` cancelled. Torch synced. `temp-trash/` created and old cache moved there.

**Verified cache numbers** (`CO-20131003`, 71 units, T=66321, 11.1 min, adapted ctx):
`rate_mean 3.47 · rate_median 1.38 · frac_silent 0.169 · active_units 59 · mean_pairwise_corr 0.083 ·
pc1_var 0.19 · eff_dim 18.4 · subspace_angle_deg 47.7 · speed_mean 4.31 · moving_frac 0.33 · dir_coverage 8/8`.
Decoder caching (one session, ~24 s): `ridge in 0.452/out −0.229 · wiener 0.513/−0.629 ·
kf_posvel 0.475/−0.017 · mlp 0.996/0.084 · gru 0.557/0.267` → **in-sample is wildly optimistic**.

## 11. Next steps (the agreed sequence)
```
A+B   cache.py window_min=3 (+win_mask, win_t0/t1, session tags, cache_version=2)
      + common.py (shared evaluate_cell: align_tail, adapter stage, metrics)
      -> smoke: cache --n 1 --with-decoders ; selftest_common.py
C     run_grid.py (merged P3+P4, adds r2_refit_oracle/out)
      -> smoke: tracer bullet, one complete row, every column populated
D     staleness.py (rolling + expanding, forward-only, >=5 blocks)
      -> smoke: both curves on 1 session, sanity-checked
Then  full cache (53 sessions, job array) -> full grid -> Block 4/5 interpretation
      -> P5 trainable adapters (+LR-2) -> P6/P7 analyze (cards, curves, CIs) -> Block 6
```
**Also outstanding:** the **80/20 sanity check** ("bug vs setting" for the negative out-of-sample R²),
and the 2/3/4.5-min **window sensitivity check**.

**Two questions asked of the user and not yet answered:** (a) run the 80/20 sanity check now? (b) switch
`r2_burnin_out` to a k-fold version, or keep the half-split?

## 12. Operational rules (WORKFLOW.md)
- **NEVER `rm`** — move to `trash/` or `temp-trash/`. All work inside `$SCRATCH/invasive-drift/`.
- **NEVER poll `squeue`/`sacct` in a loop** — file sentinels + `tail`; `scancel <id>` only.
- `cpu_short`, **`--mem=4G`** is plenty (measured ≤1.3 GB). Queue backing up is fine.
- Login `/tmp` is often FULL → use `$SCRATCH/invasive-drift/tmp/`.
- **Push a backup before starting new implementation**; sync local → Torch with `git pull --ff-only`.
- Scripts print peak RSS + timings themselves.
- **Blocks are discussion gates**: stop, look, agree, then proceed.

