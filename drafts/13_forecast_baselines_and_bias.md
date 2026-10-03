# Forecast baselines & the "it always goes down" bias

**Status:** methodology note. Guards the MVP forecast (`drafts/10`) and the spine (`drafts/09`).

## The trap

If `health(t)` (decoder R² over time) drifts **monotonically downward**, then a **trivial predictor**
that just says *"the future will be lower than now"* beats **persistence** without learning anything —
no mechanism, no lead time, no *when*. A "down-trend bias" masquerades as skill.

This is the same failure mode as the persistence-baseline problem (`drafts/04`), one level deeper:
persistence is too weak a null for a **trending** series.

## The fix — a baseline ladder

Every forecast result must beat the rung **above** it:

| # | Baseline | Definition | Beats |
|---|----------|------------|-------|
| 0 | **chance** | predict the mean / majority | — |
| 1 | **persistence** | `health(t+h) = health(t)` | chance |
| 2 | **trend / AR** | `health(t+h) = health(t) + fitted mean slope·h` (slope fit on **train** sessions only) | persistence |
| 3 | **our panel** | reduced-panel features at `t` | trend |

The headline = **incremental skill over rung 2** (the trend), not raw accuracy.

## Three defenses

1. **Detrend the target.** Predict the residual `health(t+h) − trend(t+h)` — i.e. *deviations* from the
   expected decay. The nontrivial question: *can features forecast when the drop is faster/slower than
   expected, or where a crash exceeds the trend?*
2. **Fit the trend on training sessions only** (never the test session); use **session-out / subject-out**
   splits and **permutation nulls** (shuffle features↔target).
3. **Report incremental metrics** (ΔR², ΔAUROC) + calibration over the trend baseline; state the trend
   baseline's own score so a reader sees how much is trivial.

## Why this matters here

- **Within-session** (units fixed): the decay 0.12→0.05 (job 19097276) *is* a trend — so the within-session
  forecast **must** clear rung 2, or it is just re-learning "sessions decay."
- **Cross-session**: the FALCON collapse (0.49→negative) is a cliff, not a trend; but the few-shot series
  still trends, so the same ladder applies.

## Decision rule (pre-registered)

- **GREEN:** panel beats the **trend** baseline (rung 2) on detrended targets, outside the CI.
- **YELLOW:** panel beats persistence but **not** trend → the "finding" is just the trend; reframe.
- **RED:** panel ≤ persistence → no forecast.
