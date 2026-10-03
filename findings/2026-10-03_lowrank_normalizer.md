# Finding — The low-rank NORMALIZER: moment-matching helps a little; landmark rotation does not

**Date:** 2026-10-03
**Dataset:** Perich `000688` sub-C, 53 sessions
**Script:** `25_lowrank_normalizer.py` (job 19110782)

## Protocol (honest, BCI-style, within-session)
**Burn-in = first 20%** → fixes the scaler `(m0,s0)`, the **frozen decoder**, and the reference landmarks
`C0` (mean state per movement direction) + reference moments. **Online = 8 sliding windows** over the
remaining 80%; each window's normalizer is estimated **label-free** from that window's structure, applied
to the features, then the **frozen** decoder decodes. (Direction labels for landmarks come from velocity —
i.e. the *cued-target* assumption, true for center-out.)

## Ladder (mean test R², 53 sessions)
```
method        meanR2   headroomRecov
frozen         0.288        0.00      <- the thing to beat
full_rank   -375.408    -3400.35      <- 21's failure mode (d-dim Procrustes on 8 points)
lr2            0.283       -0.04
lr3            0.275       -0.08
lr5            0.262       -0.19
lr7            0.252       -0.27
A1_moment      0.303       +0.10      <- per-unit z-score re-match  (LABEL-FREE, best)
A2_lowcov      0.288       -0.01      <- low-rank covariance match
refit          0.446        1.00      <- UPPER BOUND (headroom = 0.29 -> 0.45)
LATENCY (landmarks from first 25% of window, decode last 75%): frozen 0.288 | lr5 0.220
```

## Interpretation
1. **There IS headroom** (frozen 0.288 → refit 0.446).
2. **The landmark idea — "normalize using the first few reps of each piece" — FAILS**, even low-rank:
   `lr2…lr7` are all **negative**, and the damage grows with k (−0.04 → −0.27). Full-rank is catastrophic.
   Reason: the drift is a **deformation**, not a rotation (`20`), and a Procrustes rotation fit to 8 noisy
   centroids adds noise rather than removing drift.
3. **Moment matching (A1) is the only positive**: re-matching each unit's mean/std to the burn-in
   reference recovers **+10% of the headroom** (0.303 vs 0.288). It is **entirely label-free** and needs
   **no landmarks**. (A2's low-rank covariance version adds nothing.)
4. **Latency:** the low-rank landmark variant still under-performs the frozen decoder even using only the
   first 25% of a window for landmarks.

## Bottom line
> **The low-latency normalizer works — but only in its simplest, label-free form:** per-unit **moment
> re-matching** gives a **small, real gain (+10% of headroom)**. The **landmark-based low-rank rotation
> does not help** (it hurts), because the drift is a deformation, not a rotation.

This is the **first positive intervention** in the within-session programme — small, but real and cheap.

## Caveats
- Gains are small (+0.015 R² absolute). The bulk of the headroom is *not* recovered by any linear/unsupervised map.
- Landmark directions come from velocity (cued-target assumption); an unsupervised-clustering variant is untested.
- 8 windows/session; window length varies (session-length dependent).

## Resources
131 s, 1.26 GB → 4 G ample.
