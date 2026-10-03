# Finding — Pairwise "gaps" do NOT widen consistently, and can't be extrapolated

**Date:** 2026-10-03
**Dataset:** Perich `000688` sub-C, 53 sessions
**Script:** `22_pairwise_relations.py` (job 19108913)

## The question (user's "piano" follow-up)
Units = keys; a movement direction = a piece; a time window = one performance. For each pair of units,
take the **gap** = distance between their **direction profiles** (their mean firing across the K=8
directions). *Is the gap getting wider the more you play, consistently? And can we extrapolate it?*

## Results (K=8 dirs, NW=6 performances, |rho|≥0.8 = "monotone")
```
(1) monotone pairs (|rho|>=0.8)      = 0.15
    of monotone, fraction WIDENING    = 0.37   (<0.5 => no directional bias)
    median rel. change gap(w1->wNW)   = -0.03  (essentially zero)
(2) extrapolate next gap: skill vs persistence = -1.750  (corr +0.65)
(3) temporal lag: monotone pairs      = 0.22 ; median |lag change| = 2.9 bins (~58 ms)
```

## Interpretation — **"different", not "consistent"**
1. **Only ~15% of unit pairs change monotonically** → for **85%** the gap's motion is **non-monotone**
   (up-down wobble). It is **not** a consistent directional change.
2. **Of the monotone pairs, only 37% widen** (< 50%) → **there is no general widening**; if anything
   slightly more pairs *narrow*. The **median relative change is −0.03** → the typical gap barely moves.
3. **Extrapolation fails badly:** skill = **−1.75** (≈ 2.75× worse than persistence). The extrapolated
   gap *correlates* with truth (+0.65 — so gaps are stable), but its **direction of change is not
   predictable**, so pushing the trend forward blows the error up.
4. **Temporal lag** behaves the same: only **22%** monotone, median |lag change| ≈ 2.9 bins (~58 ms).

## Bottom line
> **The gaps don't widen consistently, and you can't extrapolate them to the next performance.** The
> change is a *random wobble around a roughly fixed value*, not a directional widening — the same
> "smooth random walk" verdict we keep hitting, now at the level of **unit-pair relationships**.

**So the answer to "is it a consistent change or a different one?" → different (random). And "can we
corner it?" → not by extrapolation.**

## Caveats
- `|rho|≥0.8` over only 6 windows is a fairly strict monotonicity bar; loosening it raises the count but
  cannot create a *signature* direction (the widening fraction is already < 0.5).
- Gaps use 8 direction bins; `MINBIN=15` samples/bin; units must be valid in all windows → 38–71
  units/session.

## Consequence
Together with `17`–`21`, the within-session picture is now **closed and consistent**:
representational drift that is **low-dimensional, non-rigid (deforming), random-walk in direction,
and not extrapolatable** — at the level of the *whole state*, the *axis*, the *manifold*, and now
*pairwise relations*. The one structural positive remains the **shared mode (~45%, `21`)** → which is
why the only remaining within-session hope is a **low-rank alignment** (fix, not forecast).

## Resources
91 s, 1.03 GB → **4 G** is ample.
