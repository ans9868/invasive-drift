# plan.md — MVP next steps

Non-destructive running plan (nothing here is deleted; options stay until done).

## Torch workflow rules (hard)

- **NEVER poll `squeue` (or `sacct`) in a loop** — it spams the SLURM controller; admins email / can
  rate-limit. Use **passive file sentinels + `tail` of logs** instead. `squeue`/`sacct` only at
  milestones (one-shot).
- Each job writes a **sentinel line** to its `.out` (e.g. `DEG_DONE`, `ZOO_DONE`); check for that line,
  not the scheduler.
- Sync via **git**; never `scancel -u` (only specific job IDs).

## Current status

- **Decoder fixed.** FALCON M1-A = **EMG** task (was wrongly classifying `tgt_loc`). Perich `000688` =
  **cursor-velocity** decoder (and it has **waveforms** + kinematics — richer than FALCON).
- **Within-session drift confirmed** across sessions: `06_session_scan` → 6/7 sessions decay
  (mean R² 0.238 → 0.131, slope ≈ −0.0022/min). Units fixed → no matching needed.
- **Forecast ladder (bias-guarded)** `07_within_forecast`: **trend 0.585 > persistence 0.555 >
  panel 0.494** → the **reduced panel does NOT beat the trend**. Honest negative for the *thin* feature
  set. (Without the ladder we'd have over-claimed.)

## Options (do NOT delete — mark done as we go)

1. **Enrich the features** — add decode MSE/entropy, per-unit rate stability (L1 change), mean pairwise
   correlation (+change), rate CV, and (Perich) waveform-SNR; re-run the ladder. *(doing now)*
2. **More data** — download all sub-C (53 sessions) + other subjects; more rows → less overfit.
3. **Accept the negative** — "reduced spike-only features are insufficient; raw/QC features required"
   — which *justifies* the IBL-raw arm (`drafts/07`).
4. **Cross-session** — implement waveform-based **unit matching** (UnitMatch-style) → cross-session
   `health(t)` + the same ladder; this is also the decomposition's **H3** instrument.

## Log

- **INJECTION/CONFIRM TEST DONE** (`11_injection_test.py`, job 19104802) — synthetic const-accel smoother,
  n=15, σ=0.05:
  - coverage: slope/accel **over-cover (100% → CIs too wide)**; with acceleration present the **slope is
    biased (−0.056) and misses** (slope↔accel trade-off).
  - walk-forward 1-block skill (1−MSE/MSE_persist): `hist_slope` beats the const-accel `model`
    (slope−0.05: 0.597 vs 0.510); **accel adds nothing** (accel case 0.465 vs 0.475); **steps smear the
    const-accel model** (step case: hist_slope −0.079 vs model −0.676); on flat, slope predictors hurt
    (correctly — no false skill).
  - **CONCLUSIONS:** (a) the bar is the **historical-slope** baseline, not persistence; (b) **acceleration
    is NOT forecastable** (our earlier Δaccel=+0.25 was over-fit) → drop/flag it; (c) **steps need a
    separate change-point layer** (BOCPD/PELT); (d) report **Δ over hist-slope**.
- **WAVEFORM PROBE DONE** (`12_waveform_degradation.py`, job 19105071, CO-20150309, 74 units, 57.7 min):
  within-session waveform **stability ≈ 0.997–0.999**, **waveform drift (1−corr 1st/2nd half) median 0.000
  / max 0.016**, alive units 72–74 stable; amplitude **417→388 µV (−7%)**.
  - **⇒ within-session unit waveforms are essentially unchanged → the within-session decoder drift is
    NOT unit-loss/waveform-driven; it is representational (tuning) or gain.** The −7% amplitude is a
    candidate gain/SNR effect.
  - **⇒ waveforms are the wrong probe for *within*-session drift, but the RIGHT probe *cross*-session**
    (unit death/appearance + identity). Perich has waveforms → unlocks **unit matching** (Track-A / H3
    blocker) + cross-session amplitude/stability trends = direct unit-loss/isolation/gain evidence.
- **NEXT:** (a) `12_degradation_v2` (rate vs hist-slope, σ_i, step layer, drop accel); (b) cross-session
  waveform unit-matching + per-unit amplitude/stability over days (decomposition mechanisms).

- **#1 DONE** (`08_within_forecast.py`, job 19097685). Rich features → **panel 0.570 > trend 0.515 >
  persistence 0.482**; ΔR² over trend **+0.055** (was −0.091). Caveats: corr *worse* than trend
  (0.757 vs 0.779), detrended fails, n=52 → *tentative*.
- **#2 DONE.** Downloaded **all 53 sub-C CO sessions** (~2.6 GB).
- **Decoder zoo DONE** (`09_decode_zoo.py`, job 19100558, N=53, train-80%/test-last-20%):
  `gru 0.607 | mlp 0.576 | wiener 0.404 | kf_posvel 0.393 | kf_vel 0.383 | ridge 0.357` (mean R²).
  - nonlinear ≫ linear; `kf_posvel ≥ kf_vel` (Gilja #2) in ~all sessions; Wiener > ridge.
  - range 0.03–0.88 (session-quality driven). In the published M1 ballpark.
  - **Decoder of record = `gru` (or `mlp`)** for the degradation-prediction phase.
- **NEXT PHASE — predict degradation.** Two target framings:
  - **rate of degradation**: regress the within-session `health(t)` slope (and/or cross-session slope);
  - **large degradations**: predict "crash" events (a block/boundary where R² falls sharply).
  Keep the bias ladder (`drafts/13`): chance → persistence → **trend** → panel, on detrended targets.
- **DEGRADATION FORECAST DONE** (`10_degradation_forecast.py`, job 19104112) — MLP decoder, denoised
  `health(t)` (const-accel Kalman smoother → level/slope/accel), 6 long sessions / **45 rows**:
  | target | persistence | trend | panel | Δpanel−trend |
  |---|---|---|---|---|
  | level | +0.813 | +0.909 | +0.916 | **+0.007** (tie — trivial) |
  | **rate** | +0.277 | +0.285 | +0.785 | **+0.500** ✅ |
  | **accel** | +0.350 | +0.377 | +0.627 | **+0.251** ✅ |
  | D1 (5-min drop) | +0.790 | +0.790 | +0.501 | −0.288 |
  | D3 (15-min drop) | −1.269 | −1.264 | −0.189 | +1.075 (unstable) |
  - **Headline: rate & acceleration of degradation ARE forecastable beyond the trend.** Level is
    trivial; raw fractional drop `D_h` is numerically fragile → prefer derivative targets.
  - **Caveat:** only 6 sessions / 45 rows (the `<5 blocks` guard skipped all sessions < 25 min).
    → Need more rows to trust +0.50.
- **NEXT:** more sessions/rows — relax guard (accept ≥4 blocks), shorter blocks (e.g. 2–3 min), and/or
  add other subjects (sub-M has 28, sub-T 12). Then reformulate `D_h` (or drop it).
