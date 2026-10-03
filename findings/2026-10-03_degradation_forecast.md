# Finding — Degradation forecast: rate is NOT forecastable at scale

**Date:** 2026-10-03
**Scripts:** `07/08_within_forecast.py`, `10_degradation_forecast.py`, `13_degradation_v2.py`
(jobs 19097527, 19097685, 19104112, 19105335)

## The question
Can a cheap spike-derived **panel** (rate, active units, correlation, decoder-MSE, rate-CV, rate-change)
forecast the **next-block decoder health** — beating the trivial baselines?

## Baseline ladder (bias guard, `drafts/13`)
chance → **persistence** → **trend** → **historical-slope** → panel. Skill = `1 − MSE/MSE_base`.

## Results
### Small n (6 sessions / 45 rows) — looked promising
- level: panel ≈ trend; **rate: panel − trend = +0.50**; **accel: +0.25**.
### Injection/coverage test (`11`, job 19104802)
- **The real bar is `hist_slope`**, and it often **beats** the const-accel model
  (slope−0.05: hist_slope 0.597 vs model 0.510).
- **Acceleration adds nothing** (0.465 vs 0.475) and **contaminates the slope** (bias −0.056).
- **Steps smear the const-accel model** (step: hist_slope −0.079 vs model −0.676).
### Large n (53 sessions / 446 rows, `13`) — the honest answer
```
next-block LEVEL: persist 0.000 | hist_slope +0.146 | model +0.185   → modest +0.04 over the true bar
next-block RATE : persist 0.000 | hist_slope  0.000                   → NO skill
steps/session: 0
```

## Interpretation
> **With 10× more data, the earlier "rate Δ+0.50" EVAPORATED.** What survives is a **modest
> improvement over the historical-slope baseline on the next-block *level* (~+0.04)** — and
> **the rate of degradation is not forecastable** from these features. Acceleration is not either.

## Caveats / notes
- One `13` number (`model` for slope-target) is a **bug** (advanced the slope target with the level
  formula); the meaningful reading is `hist_slope` skill = 0 for rate.
- CUSUM found no steps (threshold or genuinely no steps).
- Methods to try next per `drafts/15` addendum: floor-crossing probability, per-block σᵢ, BOCPD/PELT step
  layer.
