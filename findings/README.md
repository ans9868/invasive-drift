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
- `2026-10-03_manifold_motion_and_repeat.md` — **not a rigid moving object** → the drift is a
  **deformation** tracking the R² drop; the "piano" test shows a **shared wobble (PC1≈0.45)** but
  persistence beats rotation and naive landmark alignment **hurts**. (+ 2 concrete fixes)
- `2026-10-03_pairwise_relations.md` — pairwise "gaps" between units **don't widen consistently**
  (only 15% monotone; 37% of those widen; median change ≈ 0) and **extrapolation fails** (skill −1.75).
  → within-session drift is a random walk at *every* level tested (state/axis/manifold/relations).
- `2026-10-03_lowrank_gate_and_volatility.md` — **low-rank gate PASSES** (drift PC1+2 = 0.80, ~2-D →
  a low-rank normalizer has a real shot); **volatility fails** (lag-1 autocorr 0.04 → memoryless, and
  does **not** lead the R² drop) → the entropy/early-warning route is closed.
- `2026-10-03_normalizer_perspective.md` — the "win" objectively: **real & consistent** (+0.015 R²,
  94% of sessions, p≈5e-12) but **small** (~5% relative) and mechanistically **per-unit gain whitening**
  (unit-shuffled control changes nothing) — **not** drift realignment.
- `2026-10-03_lowrank_normalizer.md` — the **normalizer**: label-free **moment re-matching helps a little
  (+10% of headroom)**; the **landmark low-rank rotation fails** (negative at all k) → drift is a
  deformation, not a rotation.
- `2026-10-03_mlp_confirms_gate.md` — **nonlinear MLP confirms the low-rank gate** (PC1+2 = 0.82 vs ridge
  0.79) → not a linear artifact; volatility still memoryless & non-leading. (Caveat: per-block MLP unstable.)
- `2026-10-03_waveform_probe.md` — waveforms stable within-session; **cross-session matcher works**
  (corr≈0.996) → unit-identity instrument.
- `2026-10-03_degradation_forecast.md` — rate/accel NOT forecastable at 53 sessions; only a modest
  level gain over the historical-slope baseline.
- `2026-10-03_injection_and_method_validation.md` — injection/coverage; the historical-slope bar;
  accel out; steps need their own layer.
- `2026-10-03_data_and_infra.md` — datasets (Perich 000688, FALCON 000941), env, Torch compute, workflow.

Background/thinking docs live in `../drafts/` and `../plan.md`.
