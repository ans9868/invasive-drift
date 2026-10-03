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
- `2026-10-03_adapter_grid_framing.md` — ⭐ **the reframe, written BEFORE the grid ran.** Three registers
  (classical raw R² / our skill scores / **ceiling** quantities like `r2_persist` that are properties of the
  behaviour, not decoders); the **pre-registered prediction P1–P4**: the adapter grid is a **ladder in moment
  order** (`mom_global` → `mom_diag` → `cov_lowrank` → `zca`) vs the **direction-only** family
  (`subspace`/`centroid_proc`); claim = *the class of alignment method that works is determined by which
  moment order the drift lives in* (NoMAD ≈ `mom_diag`+`zca`, Aligned FA ≈ direction-only). Also locks the
  measurement conventions (no per-bin CIs; across-session std / across-pair IQR + signed-rank) and the
  "do-not-claim" list.
- `2026-10-03_adapter_grid_results.md` — ⭐ **THE GRID RESULTS (53 sessions × 5 decoders × 12 objectives × 4 N
  = 12,720 rows).** **P1 FAILS** (moment ladder not monotone: `mom_global` −0.0009 → `mom_diag` +0.0065 →
  `cov_lowrank` +0.0039 → `zca` −0.0015; but per-unit DOES beat global, +0.0074, p≈0). **P2 CONFIRMED**
  (`subspace` **−0.1711**, only 3.8% of cells improved, harmful in every decoder; verified real via
  `corr_rel`=0.89 full-rank, not a blow-up). **P3 REFUTED** — the negative control `shuffled_ref` scores
  **+0.0082** ≈ the real `mom_diag_self` **+0.0085**, and `mom_diag` is *significantly worse than its own
  shuffled control* (p=6.5e-07) → the gain is **generic re-standardisation, not identity-specific**.
  Only **`out_affine`** (+0.0146, 90%) clearly helps — and it is **supervised**. Unsupervised within-session
  adaptation ≈ null (+2.7% relative, 3–6% of headroom).
- `2026-10-03_prior_art_falcon_nomad.md` — **read of the two defining papers.** FALCON = the benchmark
  (continuous/causal eval, **no trial labels**, R² = **variance-weighted** multi-output, held-in/held-out
  splits, published baseline table incl. **zero-shot WF 0.34±0.06** on M1-A); NoMAD = the SOTA (LFADS +
  KL-aligned feedforward net, **per-channel z-scoring** = our normalizer finding, **half-life = ln2/B after
  SNR = −10log10(1−R²)** = our staleness metric). ⚠️ **Their own within-session half-lives are 3.2 min
  (static) → 11.7 h (NoMAD+RTI)** → we must cite them and *not* claim within-session drift is unstudied.
  Also settles the CI question (report across-session std / across-pair IQR + Wilcoxon signed-rank; never
  per-bin CIs).

Background/thinking docs live in `../drafts/` and `../plan.md`.
