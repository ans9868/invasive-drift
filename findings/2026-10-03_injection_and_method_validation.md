# Finding — Method validation: injection/coverage + the historical-slope bar

**Date:** 2026-10-03
**Script:** `11_injection_test.py` (job 19104802)
**Setup:** synthetic `h(t) = μ(t) + ε`, `ε ~ N(0, σ²)`, n=15 blocks, σ=0.05; const-accel Kalman smoother
with **known** σ.

## Coverage (target 95%)
```
config             slope_cov  slope_bias  accel_cov  accel_bias
flat                    1.00     -0.0002      1.00    -0.0000
slope=-0.02/min         1.00     -0.0002      1.00    -0.0000
slope=-0.05/min         1.00     -0.0002      1.00    -0.0000
accel=-0.002/min2       0.00     -0.0562      1.00    -0.0000   ← slope contaminated by accel
step=-0.30              1.00     +0.0016      1.00    +0.0022
```
- Slope/accel intervals **over-cover (100%) → too wide to sharply identify** at n=15.
- **When acceleration is present the slope estimate is biased and its interval misses.**

## Walk-forward 1-block-ahead skill (1 − MSE/MSE_persistence)
```
config               hist_slope     model
flat                   -0.227      -0.493    ← slope predictors HURT on flat (no false skill)
slope=-0.02/min        +0.077      -0.124
slope=-0.05/min        +0.597      +0.510    ← slope helps; hist_slope beats the model
accel=-0.002/min2      +0.465      +0.475    ← accel adds nothing
step=-0.30             -0.079      -0.676    ← const-accel SMEARS the step
```

## Conclusions
1. **The bar is the historical-slope baseline**, not persistence.
2. **Acceleration is not forecastable** (and contaminates slope).
3. **Steps (channel death) break the const-accel model** → need a separate change-point layer
   (BOCPD/PELT).
4. On flat series, slope-based prediction **hurts** — the check behaves honestly (no false skill).
