# Finding — LOW-RANK GATE **PASSES**; volatility/entropy does **not** lead

**Date:** 2026-10-03
**Dataset:** Perich `000688` sub-C, 53 sessions
**Script:** `23_lowrank_volatility.py` (job 19109422)

## Method
Unit-free: per-block decoders in the session-standardised basis, probed on a fixed set of per-direction
mean states → **velocity field `V_b`** (2K = 16-D). Drift step `dV_b = V_{b+1} − V_b`.

## Results
```
(A) LOW-RANK?  velocity-field steps (16-D): PC1=0.56  PC1+2=0.80  rank80=2.5/16
               weight-space steps:          PC1=0.44
(B) VOLATILITY: vs block index rho=+0.17 (weak trend) ; lag-1 autocorr=+0.04 (NOT predictable)
(C) LEAD/LAG: corr(vol_b, R2_b)   = -0.162
             corr(vol_b, R2_{b+1}) = -0.162   (no stronger lead)
             corr(vol_b, dR2)    = +0.034    (null)
```

## Interpretation
### (A) The drift IS low-rank → **GATE PASSES** ✅
- **PC1+2 explains 80%** of the block-to-block change; **PC1 = 0.56** (weight-space PC1 = 0.44).
- I.e. the drift is, to first order, a **~2-dimensional transformation** — a small, estimable object.
- This is exactly what `21`'s 45% shared mode predicted, and it **rehabilitates the normalizer idea**:
  `21`'s landmark alignment failed because it was estimated in *full* dimension (under-determined);
  a **low-rank (k≈2–5) alignment is well-determined** and should be estimable from a few landmarks.
- **⚠️ Caveat:** `rank80 = 2.5` is partly bounded by the number of blocks (≤ nb−1 ≈ 3–27 rows), so treat
  `rank80` as indicative; the **`PC1+2 = 0.80`** number is the meaningful one (it does not depend on nb).

### (B) Volatility (entropy proxy) is NOT predictable ❌
- Lag-1 autocorrelation = **+0.04** → the volatility of the drift carries **no memory**: knowing this
  block's volatility tells you nothing about the next. Only a weak upward trend (+0.17) over the session.

### (C) Volatility does NOT lead the R² drop ❌
- `corr(vol_b, R²_b)` = **−0.162** (weak, contemporaneous) and `corr(vol_b, R²_{b+1})` = **−0.162** —
  identical, i.e. **no lead**. The R²-*change* correlation is +0.03 (null).
- So the "pass a signal that the entropy is going to change to the decoder" idea **does not work here**:
  the entropy isn't predictable and doesn't precede the drop.

## Bottom line
> **Structural gate PASSES, informational signal FAILS.**
> The drift is **low-rank (2-D)** → a **low-rank normalizer has a real shot**.
> But the drift's **volatility is memoryless and does not lead** → entropy gives no early warning.

## Consequences
- **Idea 8 is revived**: build a **low-rank (k≈2–5) online normalizer** rather than the full-rank one.
- The **entropy/early-warning** route (via drift volatility) is **closed**.

## Resources
60 s, 1.14 GB → **4 G** ample.
