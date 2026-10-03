# 14 — Designing the "degradation" target (gradient, not a hard rule)

Discussion note (2026). How to turn "predict degradation" into a principled, gradient target.

## Why not a hard threshold
`crash = R² < τ` is (1) arbitrary (τ, functional form), (2) scale-dependent (0.85 vs 0.30 sessions),
(3) magnitude-blind (how bad, and is it accelerating?). → use a **gradient**; tiers are only a
discretization of it.

## The unifying target family
> `D_h(t) = ( ℓ(t) − ℓ(t+h) ) / ℓ(t)`  — **fractional** decoder-health lost over the next `h`.

- **session-normalized** (fixes scale),
- subsumes **rate** (small h), "how much will drop soon" (h = window), and "large degradation" (large `D_h`),
- works at **multiple timescales**: `h ∈ {1, 5, 15, 30 min}` (fast clock); `{next-day, 1-wk, 1-mo}` (slow).

## Derivative hierarchy (level → rate → acceleration)
| order | symbol | meaning | note |
|---|---|---|---|
| 0 | `ℓ(t)` | current health | ≈ persistence baseline |
| 1 | `ℓ̇(t)` | **rate** of degradation | the target |
| 2 | `ℓ̈(t)` | acceleration | **crash signature** — a step fault (channel death) = accel spike |

Ramp vs step (from the forecast-note taxonomy): smooth ramp ⇒ `ℓ̈≈0`; sudden fault ⇒ accel spike.
**Second derivative is the most interesting and the noisiest** — needs denoising.

## Denoising (avoid finite-differencing noisy R²)
- Fit a **constant-acceleration state-space (Kalman) smoother** to `health(t)` → latent
  `[level, slope, accel]` **with uncertainty**. (This is the forecast-note "clocks + derivatives" idea.)
- Alternatively wider blocks / rolling smoothing; report CIs and propagate into `D_h`.

## Tiers = discretization of the gradient
- continuous severity `s(t)` ∈ {`D_h`, `ℓ̇`, `ℓ̈`};
- **tiers by cross-session quantiles** (terciles) → **small / medium / large** (data-driven);
- also offer **semantic** tiers (usable / marginal / broken) for interpretability;
- report **both** regression (continuous) and ordinal classification (terciles).

## Open questions
1. Target quantity: relative `D_h` (lean) vs slope vs acceleration — or forecast all three?
2. Tier boundaries: quantiles vs semantic thresholds?
3. Which derivative is actually forecastable? (level trivial; rate = goal; accel = may be too noisy.)
4. Denoising: block size vs state-space; propagate CI into `D_h`?
5. **Mechanism-linked rates**: forecast the rate of *decoder health*, or of the *mechanisms*
   (unit-loss / drift / gain rates) via the decomposition? → most informative target.
6. What is "degradation" *defined on* — decoder R², cross-session decodability, or unit-match survival?

## Bias ladder still applies
chance → persistence → **trend** → panel, on **detrended** targets; session-out CV; report **Δ over trend**
(`drafts/13`).
