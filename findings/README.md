# findings/ — index

Dated finding notes from the invasive-drift MVP. One file per topic; newest date wins.

- `2026-10-03_decoder_zoo.md` — decoders: GRU/MLP ≫ linear; Gilja's position-state KF trick reproduces.
- `2026-10-03_within_session_drift.md` — within-session drift = **representational (tuning rotation)**,
  r≈0.53; not behavior/gain/loss; partially recalibratable.
- `2026-10-03_waveform_probe.md` — waveforms stable within-session; **cross-session matcher works**
  (corr≈0.996) → unit-identity instrument.
- `2026-10-03_degradation_forecast.md` — rate/accel NOT forecastable at 53 sessions; only a modest
  level gain over the historical-slope baseline.
- `2026-10-03_injection_and_method_validation.md` — injection/coverage; the historical-slope bar;
  accel out; steps need their own layer.
- `2026-10-03_data_and_infra.md` — datasets (Perich 000688, FALCON 000941), env, Torch compute, workflow.

Background/thinking docs live in `../drafts/` and `../plan.md`.
