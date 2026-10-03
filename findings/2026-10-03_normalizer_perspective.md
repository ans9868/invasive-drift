# Finding — Putting the normalizer "win" in perspective (objective controls)

**Date:** 2026-10-03
**Dataset:** Perich `000688` sub-C, 53 sessions · **Script:** `26_perspective_normalizer.py` (job 19111071, 51 s)

## The measured win
```
(a) A1 - frozen per session: mean=+0.0153  sd=0.0159  median=+0.0101
                            frac>0 = 0.94    signTest p = 5.5e-12     (95% CI ~ [+0.011, +0.019])
```
→ **real and highly consistent** (94% of sessions improve), but **small** (+0.015 R² ≈ 5% relative;
10% of the refit headroom). Between-session R² spread is ~0.15–0.20 → the win is tiny next to how much
sessions differ.

## Controls — and what they imply
```
(b) frozen=0.288  mean_only=0.295  std_only=-13843  A1(both)=0.303
(c) unit-SHUFFLED reference = 0.303   (IDENTICAL to A1)
(d) global scalar moments   = 0.281   (worse than frozen)
(e) behavior control: spearman(gain, speed delta) = -0.167  (weak)
```
- **(c) is decisive:** permuting *which* burn-in mean/std is paired with which unit changes **nothing**.
  So the gain does **not** come from matching units to *their own* reference.
- **(d):** a single global scalar **hurts** → the correction must be **per-unit**.
- **(b):** naive **std-only is catastrophic** (−13843) → the normalization must be paired with re-centering.

## Objective interpretation
> The effect's core is **per-window, per-unit variance normalization (whitening)** — dividing each unit by
> its own within-window σ. It **neutralizes per-unit *gain* drift**, which is a legitimate component of
> drift, but it is **not** a correction of the drift's *direction/geometry*.

So the win is best filed as: **a robust, label-free pre-decoder GAIN/WHITENING step**, not a realignment.
It is **not behavior-driven** (ρ = −0.17), and it is **fragile** (needs the re-centering; the refit bound
is self-fit and therefore optimistic).

## Bottom line
- ✅ **Real, consistent, label-free, cheap** (+0.015 R², 94% of sessions, p ≈ 5e-12).
- ❌ **Small** (~5% relative) and **mechanistically humble** (gain whitening, not drift direction).
- ℹ️ Consistent with everything else: the drift's **direction/geometry remains uncorrectable**, so the
  only recoverable part is the **per-unit magnitude**.

## Recommended framing for the write-up
> "A label-free per-unit variance-normalization layer recovers ~10% of the recalibration headroom. It
> corrects per-unit **gain** drift only; the drift's **direction** is not linearly recoverable."
