# Finding — Drift axis is NOT consistent across sessions (unit-free test)

**Date:** 2026-10-03
**Dataset:** Perich `000688` sub-C, 53 sessions (≥3 blocks)
**Script:** `19_drift_axis.py` (jobs 19108048, 19108159, 19108264)

## Method (why this is unit-free)
Per-block decoders are fit in the **session-standardised** feature basis, then each block's decoder
probes a **fixed set of per-direction mean states** → a per-direction **velocity field** `V_b` (a 2·8 =
16-vector). Because `V` lives in **velocity space**, it is comparable across sessions **without unit
matching**. Drift = how `V_b` moves over blocks.

## Results
```
(1) AXIS CONSISTENCY   (2K=16 dims -> random mean_cos ~ 0, single-pair sd ~0.25)
    end-to-end drift dir      mean_cos = +0.050  (sd 0.41)
    PC1 of V trajectory       mean_cos = +0.033  (sd 0.45)
(2) SHARED MODE
    PC1 variance fraction     trajectory 0.51 | step-changes 0.56
    cross-session cos of step-PC1 direction  mean_cos = +0.053  (sd 0.42)
(3) DRIFT RATE vs PRE-REGISTERED covariates (Spearman, n=53)
    n_units        rho = -0.014  p = 0.92
    duration_min   rho = -0.471  p < 0.001   <-- only real effect
    session_index  rho = -0.018  p = 0.90
    start_hour     rho = +0.234  p = 0.09    (n.s. after correction)
    [diag] |driftV| vs duration  rho = +0.316 ; vs n_units rho = +0.182
[resources] loop 49 s, peak RSS = 1.14 GB
```

## Interpretation
1. **The within-session drift direction is essentially RANDOM across sessions** (mean cos ≈ +0.05; two
   independent estimates agree ~0.03–0.05). ⇒ **there is no fixed "drift axis" of the array.** The turn
   is a *different* turn every session.
2. **Within a session the drift IS low-dimensional** (one mode explains ~0.51–0.56 of the change), but
   that mode's **direction does not transfer** across sessions (+0.05). → a *per-session* dominant mode.
3. **Drift rate depends only on session length** (`rho = −0.47`); **no long-term trend** (session_index
   ≈ 0), **no unit-count effect**, start-hour marginal/n.s.
4. **The drift SATURATES:** `|driftV|` grows only weakly with duration (+0.32) while rate falls as
   ~1/duration → total drift accumulates **sub-linearly in time** (a bounded/saturating process, not a
   constant-rate random walk).

## Consequences for the ideas
- **Idea 9 (is the rotation axis stable across days?) → NO** (answered, negative). Axis is random per
  session.
- **Idea 8 (rotation-axis regularizer) → weakened:** a *learned, transferable* axis doesn't exist;
  only *within-session* estimation could be used (and `18` showed that gain is small, 0.72 vs 0.66).
- **Idea 7 (rigid vs deformable) → still open and now better motivated:** within a session there *is* a
  dominant low-D mode (PC1 ≈ 0.5); the open question is whether it is a *rigid rotation* or a
  deformation — and whether it's tied to reach-vs-rest.

## Caveats
- `V` uses 8 direction bins from a 60th-percentile speed cut; empty bins → 0 (≥4 required).
- Axis comparison is in 16-D; the null sd per pair is 0.25, so with 53 sessions the mean-cos standard
  error is ~0.01 — the ~0.05 estimate is small but **clearly not a consistent axis**.
