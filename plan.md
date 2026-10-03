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

- (#1) in progress: `mvp/scripts/08_within_forecast.py` (rich features) — see runs below.
