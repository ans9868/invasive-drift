# ideas.md — directions worth pursuing

## Idea 1 — Cross-session waveform instrument (STRONG)
**What:** metrics are always **upstream** and **mechanistic**, not the downstream/noisy decoder:
match units across days by **waveform similarity** (works: median corr ≈ 0.996, 63–100% matched),
then track per unit over time:
- **amplitude / SNR over days** → channel health (impedance/gliosis) → **gain**.
- **waveform stability over days** → unit drifting off the electrode → **isolation/loss**.
- **fraction of units lost per session** → **unit loss**.
Use those as the **degradation signal + forecast features** instead of within-session R².
This directly measures the decomposition's mechanisms (unit loss / drift / gain) and unblocks the
cross-session axis (Perich has waveforms, so the "unicorn" dataset isn't needed).
**Status:** waveform matcher implemented (`14_waveform_crosssession.py`), works.

## Idea 2 — Within-session mechanism decomposition (frozen vs refit)
**What:** the within-session decoder decay (R² 0.12→0.05) is real but **not waveform-driven**
(waveforms stable within a session). So *what* decays? Separate:
- **Refit** decoder (trained on recent data) vs **Frozen** decoder (day-1). If **refit stays high while
  frozen decays → representational drift (recalibratable tuning change)**. If **both decay → intrinsic**
  (unit degradation / noise).
- **Tuning-weight drift** (encoder weights vs block 0) → representational drift magnitude.
- **Mean firing rate** trend → gain.
**Status:** building (`15_within_session_mechanism.py`).

## Idea 3 — Waveform-based unit matching as the H3/Track-A instrument
Match units across sessions → "units available to the decoder" becomes a **measured** quantity →
the decomposition's **unit loss** bucket becomes identifiable (death vs isolation vs matcher).
Overlaps Idea 1; downstream of it.

## Idea 4 — Hierarchical / partial pooling across days
Per-session estimates on ~10 blocks are noisy; a hierarchical model (random slopes per session/day)
would give honest cross-session trends (and matches the forecast-note's hierarchical-Bayes panel).

## Idea 5 — Gain vs drift vs loss, measured directly
Within a session: gain = rate scale; drift = tuning-weight change; loss = units dropping. Test which
tracks the decoder decay (Idea 2 does this).

## Idea 6 — Raw-waveform "cause layer" under the R² tracker (from reviewer note) **[revisit]**
**Principle:** raw voltage does **not** tell you the decoder got worse — it tells you **why** (channel
died / shorted / saturated / lost amplitude), which is the usual reason a block's R² drops and often
moves *before* the population R². → a **cause layer UNDER h(t)**, not a replacement for the R² tracker.
**Short-series limit does not apply inside a block:** 2–5 min @ 1–30 kHz = 10⁵–10⁷ samples, so
spectral/distributional features are identified; there are still only 4–30 blocks, so each feature is
**summarized once per block** and fed into the *same* Kalman / drawdown / Page–Hinkley stack as h(t).
**Do NOT** run Lyapunov / DFA / GARCH on raw voltage.

**Per channel, per block (the 8):**
1. **Spike-band noise floor** — high-pass 300 Hz → MAD or RMS. Dead channel collapses; railed front-end
   rails. *Single most useful raw feature.*
2. **Kurtosis of the high-passed trace** — spikes heavy-tailed; Gaussian electronic noise ≈ 3. Fall
   toward 3 = unit loss (not a firing-rate change).
3. **Threshold-crossing rate + median peak-to-peak amplitude** — no sorting; amplitude decay =
   encapsulation/micromotion; rate collapse = lost unit / meaningless threshold.
4. **Clipping fraction / rail hits** — saturation looks like a huge R² drop and is not neural.
5. **Line-noise ratio** (50/60 Hz + harmonics over neighbouring bins) — rising = reference/shielding
   failure, which a decoder will happily fit.
6. **Aperiodic exponent on the LFP (specparam)** — encapsulation/distance changes the 1/f slope;
   broadband power drop with *stable* exponent = gain, not biology.
7. **High-gamma power (≈70–150 Hz)** — for ECoG/speech cortex this is the feature speech decoders
   actually use → closer to decoder degradation than spike-band RMS.
8. **Pairwise correlation** — one number/channel (median corr with others): a short → ≈1 across a bank;
   a disconnected channel → ≈0. Impedance @ 1 kHz joins the same vector if available.

**Algorithms legit only with the waveform:** per-block feature + **const-accel smoother per feature**
(a feature slope that *leads* the R² slope = early warning); **specparam** not a raw periodogram;
coherence/CCA as a short detector; **SPC / Page–Hinkley** with limits from the first clean sessions
(Frontiers 2022 corrupted-channel paper); **SpikeInterface quality metrics** (presence ratio, amplitude
cutoff, SNR, drift) only if still sorting. **Skip:** Hurst, DFA, Lyapunov, RQA, Hawkes, GARCH.

**The whole experiment (one question):** compute the features, smooth with the *same* filter as h(t),
and ask — **does a feature's slope/change-point LEAD the R² drop on held-out sessions?**
- If spike-band RMS & kurtosis fall *before* R² → early warning + a channel mask.
- If R² falls while every raw feature is stable → behavioural / nonstationary-tuning / decoder, and **no
  waveform algorithm sees it coming** — **which is exactly our current within-session situation.**
A second forecasting stack on the voltage itself will **not** beat this.

**Feasibility, as we looked (IMPORTANT):** Perich `000688` NWB has **spike waveforms (48 samples/spike)
but NO continuous raw voltage/LFP** (the drafts' "no waveforms" note is *outdated* — waveforms do
exist; raw does not). So:
- On **Perich**: only **waveform-derived proxies** — per-unit **amplitude**, **waveform kurtosis/skew**,
  **waveform stability**, **per-channel yield** → a *partial* cause layer.
- The **full 8-feature stack needs raw AP/LFP → IBL (Prong 2)**.
- Decision: do the waveform-proxy subset on Perich now; wire the full 8-feature stack to IBL.
**Status:** not started; `12`/`14` already give amplitude + stability + kurtosis-ready waveforms.

## Idea 7 — Manifolds as "moving objects": degradation geometry of reach vs rest **[revisit]**
**Question:** treat the population as a **low-D manifold** (PCA / CCA / jPCA / GPFA) and ask whether the
manifold is **moving / deforming** over time — and whether that motion **explains or predicts** decoder
degradation. The intuition: after dimensionality reduction a 3-D object can *translate/rotate* in a
plane (rigid) or *deform* (non-rigid) — which is it here?
**Concrete tests, per block:**
- Fit manifolds **separately for reach vs rest** (and/or per target/condition). Measure:
  (i) **subspace similarity** across blocks (principal/subspace angle, Procrustes alignment);
  (ii) **geometry** — manifold volume / covariance-spectrum change;
  (iii) **rigid vs non-rigid motion** — fit translation+rotation vs affine vs deformable models to the
  per-block manifolds; *which wins?* (rigid → cheap realignment; deformable → genuine drift);
  (iv) trajectory of the manifold **centre / basis** over time.
- **Reach vs rest:** does the *condition subspace* rotate/degrade differently than rest? Does the reach
  manifold drift while rest stays put (a task-specific degradation signature)?
- **Link to `17`–`18`:** the low-D **rotation axis** we already found (PC1≈0.35; "smooth random walk on
  a low-D manifold") **is** manifold motion — test whether that rotation axis *is* the manifold's
  principal drift direction.
- **Prediction:** does manifold motion (subspace / volume change) **lead** the R² drop on held-out sessions?
**Caveats:** needs enough units/blocks; per-session `n` is small → hierarchical pooling (Idea 4).
**Status: PARTLY ANSWERED (2026-10-03, `20`).** Motion is **NOT rigid** (rigid residual 0.88 ≈ identity
0.89) → the drift is a **deformation** that tracks the R² drop (rho≈−0.38). ⚠️ affine test is
**degenerate** at 8 points → redesign with ≫ d points. See
`findings/2026-10-03_manifold_motion_and_repeat.md`.

## Idea 8 — Rotation-axis as a recalibration **regularizer** (from `18`) **[next step a]**
`18` gave a **directional** gain (low-rank rotation cos **0.721** vs persistence **0.661**) that is not
yet usable functionally. Instead of **refitting** a decoder per block (needs labels/long windows),
**align** the current decoder to the **learned rotation axis**: penalize deviation from the estimated
drift trajectory, or rotate the read-out by the estimated per-block rotation. Test whether this **beats
both refit and persistence** on held-out blocks, and how it behaves when few blocks have been seen.
**Status: TESTED (2026-10-03, `25`) — MIXED.** The low-rank **landmark** normalizer **does NOT work**
(lr2…lr7 all negative; full-rank catastrophic; drift is a deformation, not a rotation). But the simple
**label-free moment re-match DOES help a little** (+10% of headroom: 0.303 vs frozen 0.288; refit bound
0.446). → **the normalizer's value is moment-matching, not landmark rotation.** See
`findings/2026-10-03_lowrank_normalizer.md`.

## Idea 9 — Is the **rotation axis stable across days**? **[next step b]**
Re-estimate the within-session drift rotation per session and measure **axis consistency** across days
(principal angle between axes). **Stable axis → a persistent "drift direction" you can pre-empt**;
random axis → per-session-only correction. Use the **waveform unit matcher (Idea 1/3)** to keep units
comparable across days, so the axes live in the same coordinate frame. This is the more BCI-relevant
question: *"does the array drift the same way every day?"*
**Status: ANSWERED — NEGATIVE (2026-10-03, `19`).** Unit-free test in velocity space: cross-session
drift-direction cosine = **+0.05** (≈ random) → **no stable axis across days.**
See `findings/2026-10-03_drift_axis_consistency.md`.

## Idea 10 — Repeated-movement drift (the "piano") + landmark realignment **[new, from `21`]**
Units = **keys**; a movement **direction** = a small **piece**; a **time window** = one **performance**.
Play the *same* piece repeatedly and watch the "chord" (population state for that direction) move.
Questions: is the move a **shared wobble** (all keys together) or per-key? Can the **1st predict the 5th**?
Can we **re-align** later performances to the 1st using the directions as landmarks (unsupervised)?
**Result (`21`, 53 sessions):** chord drift is large (0.72); **a shared mode explains ~45%** (vs 12.5%
chance) → *there is* a global wobble; but **persistence beats rotation** (0.809 vs 0.765) and **naive
Procrustes landmark alignment HURTS** (R² 0.289 → 0.058; refit upper bound 0.408).
**Why it fails & the fix:** Procrustes from only **8 landmarks in high-dim is under-determined** →
(a) use ≫ d landmarks (per-trial / direction × phase), and (b) do the alignment **in a low-rank
subspace** (estimate the rotation in the top-k PCs). This is the most promising remaining lead because
the shared ~45% mode is real.
**Status:** first pass done (`21`); **pairwise-gap follow-up (`22`) → NEGATIVE**: only 15% of unit pairs
change monotonically, only 37% of *those* widen, median gap change ≈ 0, and extrapolation skill = −1.75.
The change is **random, not a consistent widening** → not cornerable by extrapolation. See
`findings/2026-10-03_pairwise_relations.md`. Remaining hope = **low-rank alignment** (fix, not forecast).

---
*Reviewer note (raw-waveform cause layer) folded into Idea 6; the two `18` follow-ons are Ideas 8–9;
the "moving manifold" idea is Idea 7; the "piano" is Idea 10. Revisit order: 7-redesign (≫d points) →
10-redesign (low-rank alignment) → 6 (Perich proxy) → 8. **9 is closed (negative).***

