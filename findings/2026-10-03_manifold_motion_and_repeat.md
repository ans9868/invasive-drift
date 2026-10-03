# Finding — Manifold motion (not rigid) & repeated-movement drift ("piano")

**Date:** 2026-10-03
**Dataset:** Perich `000688` sub-C, 53 sessions
**Scripts:** `20_manifold_motion.py` (job 19108528), `21_repeat_drift.py` (job 19108529)

---
## `20` — Is the condition manifold a rigid "moving object"? **NO — it is non-rigid.**
Per block: the condition manifold = the (8 × d) matrix of mean standardised firing per movement-direction
bin. Motion block b−1 → b fit as identity | **rigid** (orthogonal Procrustes) | **affine** (general
linear). Residuals normalised to the spread of the target set.

```
motion residuals (mean over blocks): identity=0.89  rigid=0.88  affine=0.00
  -> rigid explains 1% of identity error; affine adds 100% over rigid
subspace drift vs block0 (deg): reach=56.1  rest=0.0
LEAD/LAG: corr(rigid-residual, decoder R2) rho = -0.376
```
- **Rigid ≈ identity (0.88 vs 0.89)** → the manifold does **not** move as a rigid body. A rotation +
  translation explains **essentially none** of the change → **the drift is a DEFORMATION.**
- **⚠️ The affine number is degenerate:** with only **8 points** in d-dim space, a general linear map
  (d²+d params) **fits any 8 points exactly** (0.00 is guaranteed, not evidence). So this test can only
  establish **"not rigid"**; it cannot yet separate *"affine"* from *"arbitrary non-rigid"*.
- **Reach subspace rotates ~56° from block 0; rest subspace 0°** → the *condition* code rotates while
  the rest/baseline subspace stays put. (Caveat: `rest=0.0` exactly in many sessions → rest basis may be
  degenerate; needs a check.)
- **The change magnitude tracks the R² drop** (rho = −0.376 for rigid-residual vs R²) — consistent with
  the deformation *being* the degradation.

**REDESIGN NEEDED** to make rigid-vs-deformable non-degenerate: use **many points** (per-trial states, or
direction × movement-phase, or subsampled time points ≫ d) so affine is *not* trivially perfect; and/or
compare against a **low-rank deformation**.

---
## `21` — Repeated-movement drift ("piano"): big drift, a shared wobble, but NOT usable
Units = keys; a movement **direction** = a small piece; a **time window** (6 per session) = one
performance. Chord = (8 × d) mean per direction per window.

```
(1) chord drift per window (rel.)      = 0.72
(2) SHARED mode: PC1 var fraction      = 0.45   (>> 1/8 => a genuine shared wobble)
(3) predict next performance:  persistence cos=0.809  rotation cos=0.765
(4) decode: unaligned R2=0.289  landmark-aligned=0.058  refit(upper)=0.408
```
- **(1)** The same piece sounds **quite different** by the end (~0.72 relative move) — the chord drifts a lot.
- **(2)** **A shared mode explains ~45%** of the chord change (vs 12.5% for 8 independent directions) →
  **there IS a common "wobble" that all directions share.** This is the most encouraging result: it means
  the drift is *partly* a global, low-dimensional mode (an "alignment-friendly" component).
- **(3)** But **persistence (0.809) BEATS rotation (0.765)** → the *step* is still not predictable
  (same random-walk conclusion as `17`/`18`/`19`).
- **(4)** **Unsupervised landmark alignment HURTS**: aligning later windows onto window 0 by Procrustes
  over the 8 directions drops R² from 0.289 → **0.058** (vs refit upper bound 0.408).
  → as in `20`, **Procrustes from 8 landmarks in high-dim is under-determined**; the min-norm rotation is
  noise-dominated and damages the data.

## Bottom line
- **Not a rigid moving object** — the drift is a **deformation**, and it co-varies with the R² drop. ✔
- **There is a shared low-D wobble (~45%)**, so *some* structure exists to exploit. ✔
- **But it is not directly usable**: the step isn't predictable, and naive landmark alignment fails. ✘
- **Two concrete fixes to try next:**
  (a) make rigid-vs-deformable non-degenerate with **≫ d points** (per-trial states / direction × phase);
  (b) do alignment **in a low-rank subspace** (estimate the rotation in the top-k PCs, not in d dims).

## Resources
`20`: 58 s, 1.08 GB · `21`: 80 s, 1.24 GB → both fit comfortably in **4 G**.
