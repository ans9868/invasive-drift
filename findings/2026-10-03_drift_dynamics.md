# Finding — Representational drift is structured but NOT extrapolatable

**Date:** 2026-10-03
**Dataset:** Perich `000688` sub-C, 53 sessions
**Script:** `17_drift_dynamics.py` (job 19106285)

## What representational drift is
The mapping from neural activity → external variable **changes over time even when behavior is fixed**;
single-neuron tuning rotates (preferred direction/tuning curves drift) while the **population still
encodes the variable**. Picture: a stable low-D manifold whose **embedding/basis rotates**. Documented in
hippocampus (Ziv 2013), parietal cortex (Driscoll 2017), motor cortex (Gallego 2020 — stable dynamics,
drifting single-unit tuning; Rule 2019). Proposed mechanisms: synaptic turnover, network
reconfiguration, neuromodulation/excitability, representational degeneracy.

## What we measured (within a session, ephys)
Per-block encoder (spike→velocity) weight vectors, 53 sessions:
```
PC1 variance fraction (low-D?)       = 0.35    (1.0 = 1-D drift; ~0.1 = isotropic random)
mean consecutive cosine (smoothness) = 0.701   (1.0 = white noise; lower = smoother)
mean drift cos(first,last)           = 0.665   (1.0 = no drift)
extrapolation cos(next)              = 0.503
persistence cos(next)                = 0.697
```

## Result
- **Structured:** the drift is **smooth** (consecutive weight-cosine 0.70) and **moderately
  low-dimensional** (PC1 = 35% of trajectory variance) — a slowly rotating, low-D process, not noise.
- **Not predictable:** **linear extrapolation (0.50) is WORSE than persistence (0.70)** — the next
  block's decoder is best predicted by *reusing* the current one, not by continuing the trend.

## Interpretation
> Within-session representational drift ≈ a **smooth random walk on a low-dimensional manifold**:
> it stays in a subspace and moves slowly (structure) but its **step direction is unpredictable**
> (no extrapolation skill). This explains why it's partially recalibratable (re-fit tracks the new
> location) and why the "rate of degradation" is not forecastable (you can't extrapolate a random walk).

## Caveats
- Weight-trajectory PCA on 4–30 blocks/session is itself noisy.
- "Not linearly extrapolatable" ≠ "never predictable" — a rotation *model* or state-input could do
  better; but naive trend-continuation fails.
