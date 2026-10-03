# plan.md — MVP next steps

Non-destructive running plan (nothing here is deleted; options stay until done).

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
- (#3 accept-negative, #4 cross-session matching) remain open.
