# findings/ — index

Dated finding notes from the invasive-drift MVP. One file per topic; newest date wins.

- `2026-10-03_decoder_zoo.md` — decoders: GRU/MLP ≫ linear; Gilja's position-state KF trick reproduces.
- `2026-10-03_within_session_drift.md` — within-session drift = **representational (tuning rotation)**,
  r≈0.53; not behavior/gain/loss; partially recalibratable.
- `2026-10-03_drift_dynamics.md` — **what representational drift is**; within-session it's a *smooth,
  low-D (PC1≈0.35) rotation* but **NOT extrapolatable** (extrap 0.50 < persist 0.70) → smooth random
  walk: structured, not predictable.
- `2026-10-03_predicting_the_drift.md` — predicting the next decoder: **low-rank rotation beats
  persistence (cos 0.721 vs 0.661)**; trend & state input do NOT (0.464); functionally none beat
  persistence robustly.
- `2026-10-03_drift_axis_consistency.md` — **unit-free test:** cross-session drift-direction cosine
  ≈ **+0.05** (random) → **no stable drift axis**; within-session mode is low-D (PC1≈0.5) but random
  across sessions; drift **saturates** (rate ∝ 1/duration; no long-term trend).
- `2026-10-03_waveform_probe.md` — waveforms stable within-session; **cross-session matcher works**
  (corr≈0.996) → unit-identity instrument.
- `2026-10-03_degradation_forecast.md` — rate/accel NOT forecastable at 53 sessions; only a modest
  level gain over the historical-slope baseline.
- `2026-10-03_injection_and_method_validation.md` — injection/coverage; the historical-slope bar;
  accel out; steps need their own layer.
- `2026-10-03_data_and_infra.md` — datasets (Perich 000688, FALCON 000941), env, Torch compute, workflow.

Background/thinking docs live in `../drafts/` and `../plan.md`.
