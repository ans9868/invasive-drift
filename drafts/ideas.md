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

## Idea 11 — Cross-session **meta-learned adapter** (SS2) **[later — cross-session track]**
Learn the *adapter* on **other sessions** (labels allowed there — they're calibration sessions), then apply
it to a new session with **zero labels at test time**. This is the strongest realistic label-*free-at-test*
solution and the **proper home for a "deep / FALCON-class" model** (the adapter, not the decoder).
- Distinguish clearly: **test-time adaptation** (`27`) vs **transfer/meta-learned adaptation** (this idea).
- Needs the cross-session machinery: a **common unit set** or the **waveform unit matcher** (Idea 1/3).
- Sits next to the alignment literature (Degenhart 2020, NoMAD) but with the *forecasting/failure* twist.
- **Compute plan:** train on N−1 sessions, evaluate on the held-out one (leave-one-session-out).
**Status:** deferred until the cross-session track; recorded here so it is not lost.

## Idea 12 — Direction-decoding **head** (circular / 8-class) **[Option B — later]**
Decode **movement direction** explicitly, not only velocity:
- **Circular regression**: output `(cos θ, sin θ)`.
- **8-class**: softmax over center-out targets → accuracy + confusion.

**Why:** a population **rotation** (`20`, ~56°) shows up as a **systematic angular bias** — a cleaner,
more interpretable degradation axis than R², and a direct readout of representational drift. Track the
**signed mean angle** (systematic) separately from the **circular variance / mean resultant length**
(scatter). Caveat: 8-class may **saturate** (targets separable) → prefer continuous angular error.
**Status:** later; depends on the metric columns of Option A (added now).

## Idea 13 — **Multi-task** decoder (velocity + direction) **[Option C — later]**
One network predicting velocity *and* direction jointly (shared trunk, two heads). Tests whether the two
targets share drift structure, and whether the direction head acts as a regulariser for velocity.
**Status:** later; needs a trainable decoder family + the direction head from Idea 12.

## Idea 14 — Cross-decoder **uncertainty / agreement** **[from discussion]**
We already run all 5 decoders on the *same* eval rows → agreement is nearly free.
- **Epistemic:** `pred_corr_mean` (pairwise prediction correlation), `ens_disagree` (std across decoders),
  `r2_consensus` (does the ensemble beat any single decoder?).
- **Are the ERRORS aligned?** `err_corr_mean` — high ⇒ the models fail the *same way* (a common cause =
  the drift); low ⇒ idiosyncratic model noise.
- **KF posterior covariance is native** → does the filter's *own* uncertainty grow with drift? A classic
  early-warning signal we have never looked at.
**Hypothesis:** disagreement grows as drift accumulates and may **lead** the R² drop → an **unsupervised**
early warning (no labels).
**Cost:** ~free (predictions already computed). A within-decoder bootstrap ensemble would cost ~B×.
**Status:** metrics recorded as columns; the *test* (trend + lead/lag) is an analysis step.

## Idea 15 — **Non-neural baselines** / skill-over-persistence **[from discussion — DO EARLY]**
- `r2_mean` (constant → the 0 floor) · `r2_persist` (`v_t ≈ v_{t-1}`) · `r2_target` (mean velocity for the
  cued direction — task structure only).
- **Risk this exposes:** velocity is binned at **20 ms** and smoothed, so it is *highly autocorrelated* —
  persistence may already reach **R² ≈ 0.9+**, in which case our decoders' raw R² (0.36 / 0.61) may be
  **worse than doing nothing**, and **raw R² is a misleading metric**. This is the same trap we hit in
  `13` (the historical-slope was the real baseline and ate the apparent skill).
- **Fix if true:** re-express everything as **skill over persistence** — `R²_innovation` (R² on the
  residual after persistence) or `Δskill = R²_ours − R²_persist`. The **margin over persistence** may also
  be a *better* drift signal than raw R².
- **Caveat:** offline persistence uses the **true** previous velocity (a closed-loop BCI only has the
  *decoded* previous output) → an **optimistic anchor**, not a fair competitor.
**Status:** compute **Stage 0, before P3/P4** — it anchors the interpretation of every other number.

## Idea 16 — **80/20 sanity check** (settle "bug vs setting") **[deferred check]**
The decoder cache's honest out-of-sample scores came out **negative** (`ridge out = −0.229`,
`wiener −0.629`, `kf −0.017`, `mlp 0.084`, `gru 0.267`) because `r2_burnin_out` trains on only ~2.2 min.
Before trusting that, run the **zoo-style** number through the **same pipeline** on the same session
(`fit first 80% of session → test last 20%`) — it should land ≈ **0.35** for ridge. If it does, the
pipeline is fine and the negative is purely the (much harsher) setting; if it doesn't, there is a bug.
~1 min. **Re-anchors the reading of every R² we report.**

## Idea 17 — `r2_burnin_out`: half-split vs **k-fold within burn-in** **[design question]**
The half-of-burn-in split is the noisiest number we have (tiny training set, variance-sensitive R²).
A **k-fold within burn-in** (e.g. fit 5/6, test 1/6) keeps the honest out-of-sample meaning while being
far more stable. *Lean: k-fold.* Decide before P2-scale, since it changes the cached payload.

## Idea 18 — **window-length sensitivity check** (2 / 3 / 4.5 min) **[deferred check]**
Block duration trades per-cell reliability against gap resolution. Run the staleness instrument on ~5
sessions at **2 / 3 / 4.5 min** and confirm the *shape* of the curve is stable. *Lean: do it AFTER the
first real curve* — don't stall the build for it.

## Idea 19 — **Why we REMOVED windows from the grid** (2026-10-03) **[decision — done]**
The grid originally cut the online period into fixed 2–3-min windows, each split 80/20 into a fit pool and
an eval. **We removed the windows** after the cache distribution showed the median session yields only
4–6 blocks, and after re-examining what windows actually buy:

| windows bought | verdict |
|---|---|
| **temporal locality** (adapter trained adjacent to the eval) | **❌ not needed** — a single 80/20 split of the *online* period still ends the fit pool exactly where the eval begins, so the adapter is still trained on the data immediately preceding the eval |
| **repeated refitting** ("does re-adapting help?") | **❌ buys little** — drift is ≈ **−0.002 R²/min**, i.e. ~0.02 R² over a 10-min lookback — *smaller than the per-cell noise* |
| **a per-session trajectory** | **✅ but staleness already provides exactly this** |
| **more cells** | **❌ outweighed** — one cell/session × 53 sessions × adapters × N is plenty |

**What removing them GAINS:**
- adapter fit pool: 96 s → **~9.6 min** (4 800 → 28 800 bins) → covariance adapters (`zca`) become estimable at large `d`;
- eval set: 24 s → **~2.4 min** (1 200 → 7 200 bins) → R² noise ~±0.02 → **~±0.008**;
- simpler design, and the grid's numbers land on the **same 80/20 geometry as the decoder zoo**.

**What the grid becomes:** session-level **20/80** (*decoder*: burn-in → online) + **80/20** (*adapter*:
first 80 % of online → last 20 %), with **N** = a prefix of the adapter's fit pool (still causal).
**Windows survive only in the staleness instrument** (2-min blocks, `block_min`).

⚠️ **Not** zoo-comparable in absolute terms: the grid's decoder trains on the **20 % burn-in**, the zoo
trained on **80 %**. The zoo-comparable number is **Idea 16** (`sanity_8020.py`, which reproduced
ridge 0.357 / wiener 0.404 / kf_posvel 0.393 **exactly**).

---
*Reviewer note (raw-waveform cause layer) folded into Idea 6; the two `18` follow-ons are Ideas 8–9;
the "moving manifold" idea is Idea 7; the "piano" is Idea 10; the meta-learned adapter is Idea 11;
direction head is Idea 12; multi-task is Idea 13; uncertainty is Idea 14; non-neural baselines is Idea 15;
deferred checks are Ideas 16–18; window removal is Idea 19. Revisit order: 7-redesign (≫d points) →
10-redesign (low-rank alignment) → 6 (Perich proxy) → 8. **9 is closed (negative).***

