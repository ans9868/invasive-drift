# Finding — Can the drift be predicted? A low-rank rotation beats persistence (a little)

**Date:** 2026-10-03
**Dataset:** Perich `000688` sub-C, 53 sessions, 393 block-predictions (120 s blocks)
**Script:** `18_predict_drift.py` (jobs 19106837, 19106977, 19107231, 19107504)

## Question
Given the per-block encoder weights `W[0..b]`, can we predict the **next block's decoder** `W[b+1]`
better than simply reusing the current one (**persistence**)? Tested **rotation** vs **trend** vs
**state input** vs **shrink-to-mean**.

## Models
`persist`=W[b] · `trend`=2W[b]−W[b−1] · `rot`=W[b]·R (orthogonal Procrustes on history) ·
`rot_lr`=same in a 5-PC subspace · `var`=VAR operator in PC space · `shrink`=0.7W[b]+0.3·mean ·
`state`=predict the drift *step* from [time, mean rate, rate sd, |speed|].

## Result (weight-space cosine to the true next decoder)
| model | cos(next w) | mean R² | median R² | frac \|R²\|>5 |
|---|---|---|---|---|
| **rot_lr** | **0.721** | −25.6 | 0.939 | 0.031 |
| **shrink** | **0.701** | −0.06 | 0.941 | 0.010 |
| persist | 0.661 | 0.93 | 0.943 | 0.000 |
| rot | 0.663 | −26.1 | 0.937 | 0.025 |
| trend | 0.464 | −7.6 | 0.929 | 0.020 |
| state | 0.464 | −3.8 | 0.928 | 0.025 |
| var | 0.352 | −24.9 | 0.913 | 0.033 |

## Interpretation
- **A low-rank rotation does beat persistence** (cos 0.721 vs 0.661), and shrink-to-mean nearly does
  (0.701). This means the drift's **direction is modestly predictable** when modeled as a **rotation in
  a few dominant dimensions** — exactly consistent with `17` (smooth, low-D ≈ PC1 0.35).
- **Trend-continuation does NOT help** (0.464) — confirming `17`.
- **A state input does NOT help** (0.464) — behavior/rate state carries no extra predictive information
  about the drift step.
- **Full-space VAR is worse** (0.352) — over-specified in 110 dims.
- **Functionally none beat persistence:** median R² ≈ 0.93 for every model (non-discriminative), and
  mean R² is destroyed by a **~2–3% tail of catastrophic blocks** (persist has 0% such blocks). So
  predicted decoders are not robust enough to improve decoding out-of-the-box.

## Caveats
- **Weight-space cosine is the primary metric here**; the functional R² is fragile (near-zero-variance
  blocks + a catastrophic tail) — both were made shape-robust / direction-normalized in the script.
- 4–30 blocks/session; the PC subspace uses k=5.

## Bottom line
> Rotation ≫ trend. The drift is **weakly predictable in direction** (low-rank rotation: 0.72 vs 0.66
> persistence) but that gain is **not yet usable for decoding**, and naive extrapolation/state inputs
> add nothing.
