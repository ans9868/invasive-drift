# Invasive-drift decomposition — break-down, holes, build-up, and a paper "dummy execution"

**Status:** thinking notes (mostly prose, deliberately no code). v0.1 draft.

**Source document (full path):**
`/Volumes/CrucialX6/Home/Zotero/storage/2X4EU83K/invasive_drift_decomposition_proposal.pdf`
Local copy in this folder: `drafts/invasive_drift_decomposition_proposal.pdf`
(Local copy was made because the Zotero full-text cache `.zotero-ft-cache` is truncated mid-document;
`pdftotext -layout` on the PDF gives the complete 4 pages.)

Related lab context that this project leans on:
- `/Volumes/CrucialX6/Home/projects/kalman-test/dev-notes/` — the SLN + DREDge drift-correction work, the 33-session dataset, FINDINGS.md.
- `/Volumes/CrucialX6/Home/Downloads/jl_spike_sorting_idea_note.pdf` — the "pre-detection vs post-detection" spike-processing note (relevant to the before/after-spike gap below).
- `/Volumes/CrucialX6/Home/Downloads/bci_drift_forecast_note.pdf` — physical taxonomy of drift (motion / channel death / state / code warp), useful for splitting "unit loss".

---

## 1. Break it down

### 1.1 What the framework is

Four conditions, each one accuracy number:

| Cond | Train | Test | Unit set | Normalization |
|------|-------|------|----------|---------------|
| C1  | A held-in  | A held-out | **all** units in A | Session A stats |
| C3  | A held-in  | A held-out | only units **matched to B** | Session A stats |
| C2b | A (full)   | B (full)   | only units matched A↔B | **per-session** z-score (B uses B's own mean/sd) |
| C2a | A (full)   | B (full)   | only units matched A↔B | **source** stats (A's mean/sd applied to B) |

### 1.2 The three claimed components

- **Unit loss** `= C1 − C3`
  Both are intra-session on A; only the *unit set* changes (all vs matched). Claim: isolate the
  accuracy carried by units that "fail to track" across sessions.
- **Representational drift** `= C3 − C2b`
  Both use matched units; C3 is within A, C2b is A→B *with per-session normalization*. Claim:
  per-session norm removes gain, so what is left is changed tuning.
- **Gain shift** `= C2b − C2a`
  Both A→B on matched units; only normalization differs. Claim: applying A's stats to B
  re-introduces global gain, so the gap is the gain contribution.

### 1.3 Closure

`C1 − C2a = (C1 − C3) + (C3 − C2b) + (C2b − C2a)`  — called "exact, no residual."

### 1.4 The mental model the proposal sells

One accuracy number (aggregate cross-session drop) is opaque; splitting it into three
mechanisms that each map to a different fix (tracking ↔ loss, latent alignment ↔ drift,
normalization ↔ gain) lets the field grade drift-mitigation methods by *which mechanism they
address*. The pitch is "a measurement framework," not a new method.

### 1.5 The empirical anchor

From Hayashi et al. 2026 (mouse V1, calcium imaging): drift 59.5%, neuron loss 31.6%, gain 8.9%.
Predictions: ephys unit loss higher (40–60%), motor drift lower (30–50%), gain comparable-or-higher
(10–25%).

---

## 2. The holes

Ordered roughly by how badly they bite. Severity: **[fatal]** = reviewers kill it on this,
**[real]** = needs a fix before results are believable, **[minor]** = framing/caveat.

### H1. "Exact, no residual" is a tautology, and the split is path-dependent. **[fatal]**
Any chain of differences telescopes, so closure is *always* true — it carries no information about
whether the three mechanisms are separable. The *attribution* is defined along one fixed ordering
(C1 to C3 to C2b to C2a). If mechanisms interact — and unit loss and drift must, because changing
which units survive changes what "drift" is even measured on — then a different ordering yields
different numbers. There is no unique answer; there is one of several arbitrary ones. The proposal
presents the arbitrary one as *the* answer.

### H2. `C1 - C3` (unit loss) is confounded with dimensionality reduction. **[real]**
C3 is a *subset* of C1, so dropping units shrinks the decoder's input dimension. Part of the gap is
just "fewer channels," not "loss of units that carried unique information." Worse, matched units are
biased toward high-SNR / high-rate units, so C3 can be better *per unit* than C1. Without a control
that drops the same number of units at random, the component is uninterpretable.

### H3. "Unit loss" bundles biology with analysis artifacts — and is non-identifiable. **[fatal]**
The proposal's own description lists "died, drifted beyond the recording radius, became poorly
isolated, or were not matched by the matching algorithm" as one bucket. Those are three different
causes:
  - physical death / tissue response (biology),
  - sorting failure ("poorly isolated"),
  - matcher false-negative (algorithm).

The four conditions only see "in the matched set / not in the matched set." They *cannot* separate
these. So the headline percent is really "units unavailable to the cross-session decoder," which is
an **operational** quantity, not a biological one. This is the "before/after spike" gap (see §3.3).

### H4. `C3 - C2b` is labeled "drift in the strict sense" but is actually a residual. **[real]**
It absorbs everything cross-session not captured by unit-set or by per-unit affine gain: changed
tuning, changed noise correlations, behavioral/task non-stationarity, different trial sampling,
day-of-recording nuisance. Calling that whole thing "representational drift" overclaims. It is the
right *operational* knob; it is the wrong *name*.

### H5. The gain estimate assumes a per-unit affine gain model, and can eat real signal. **[real]**
`C2b - C2a` = "per-session z-score helps vs source-stats." That only equals "gain shift" if the true
shift is a per-unit, per-session, linear rescale of firing rate. Real shifts can be per-*condition*,
multiplicative across conditions, or nonlinear (Poisson variance coupling). Also, a rate/baseline
change that *carries task information* (arousal-state encoding) is real signal, not nuisance — and
per-session z-scoring throws it away, so the "gain" number can double-count signal as degradation.

### H6. No "no-degradation floor" / ceiling reference. **[minor]**
C1 (intra-session, held-out) is treated as 100%. But C1 already contains a within-session
generalization gap. Percent-of-drop is measured from C1 without acknowledging C1's own ceiling.

### H7. The comparability claim is too strong. **[real]**
"Percentages ... comparable across datasets, decoders, and time gaps" is unsafe: percent-of-drop is
sensitive to the decoder's operating point. A near-ceiling task compresses all three components; a
linear ridge on rates may manufacture a "drift" component that a nonlinear/latent decoder absorbs.

### H8. Selection / circularity risk between matching and drift. **[real]**
Units are matched using information from A and B; drift is then estimated on those same units across
A and B. If the matcher ever uses activity/tuning stationarity, drift is partly defined away. The
matcher described (waveform, drift trajectory, ISI) avoids this — but it must be *pre-registered* as
tuning-blind, because the whole drift component rides on it.

### H9. Statistical inference is underspecified. **[real]**
The four accuracies are correlated (nested unit sets, shared trials), so their differences have
dependent sampling distributions. Independent permutation nulls are wrong; you need paired/blocked
resampling over trials and units, with session-blocked nested CV to avoid leakage (matching vs the
held-out B trials used to score C2a/C2b).

### H10. Calcium-imaging to ephys metric non-equivalence. **[minor/real]**
dF/F trial-averaged z-scoring and Poisson-ish spike counts have different noise/normalization
structure. The 59.5/31.6/8.9 magnitudes do not transfer; only directions of the predictions are
defensible.

### H11. "No current method effectively addresses unit loss" overstates. **[minor]**
Drift correction / registration (DREDge, and the lab's own SLN) *does* address physical unit loss by
improving tracking; channel-dropout robustness helps too. Reframe to "no method explicitly optimizes
for unit-identity recovery across sessions."

### H12. Gain vs signal is philosophically muddy. **[minor]**
The proposal says gain = nuisance, fixed by normalization. But global rate changes are often the
brain coding state. The operational line (normalization) is fine; the interpretation ("just
normalize it away") ignores that some gain changes are the code.


---

## 3. Build up from the holes (patches)

The point of "building up" is: the framework is *repairable*; each hole has a concrete patch that
keeps the original spirit (structured decomposition) but makes the attribution honest. Patch Pn
addresses Hn.

### 3.1 P1 — Replace the fixed path with an attribution that is unique. (fixes H1)
Instead of one telescoping chain, define a **value function** `v(S)` = cross-session accuracy when
only mechanism-subset `S` is injected, for all `2^3 = 8` subsets. Two consequences:
  - The **interaction term** `v({all}) - Σ v({single})` becomes an explicit reported number. If it is
    near zero, the additive story is fine; if not, the proposal's "no residual" claim is false and
    now visible.
  - Use **Shapley values** over the lattice for the headline split — it is the unique attribution
    that satisfies efficiency/symmetry/dummy (it averages over all orderings, so it is
    path-independent by construction).
In practice you cannot ablate biology on real data, so `v(S)` is approximated by *scenario
counterfactuals*: drop a random matched subset (loss), shuffle B trial labels within unit (drift),
reapply A stats (gain). Report both the fixed-path split and the Shapley split; if they disagree,
that disagreement *is* the interaction result.

### 3.2 P2 — Add dimensionality controls for "unit loss". (fixes H2)
Two extra conditions, same unit count as C3:
  - `C3_rand`: drop the same number of **random** alive units;
  - `C3_snr`: drop the lowest-SNR units (an SNR-matched, worst-case control).
Redefine: **selective loss = (C1 - C3) - (C1 - C3_rand)**, i.e. the *excess* over what random
dimensionality loss alone produces. Report all three so the reader sees how much is "just fewer
channels."

### 3.3 P3 — Make the pipeline stage a first-class factor ("before/after spike"). (fixes H3)
This is the gap flagged: the framework has **no condition that separates biological loss from the
preprocessing/sorting/matching stage**. Fix by turning stage into a real factor:
  - Define the unit set at **two stages**: (i) raw sorted units, (ii) drift-corrected localizations
    from the lab's SLN/DREDge pipeline. This is the "before" vs "after" spike-processing axis.
  - A 2x2 (stage x session-pair) shows how much "unit loss" moves when you only change the *analysis*
    stage. If `C1 - C3` barely changes between stages while `C1 - C2a` does, the loss is
    preprocessing-bound, not biological.
  - Rename the component **"units unavailable to the cross-session decoder."** Split it only where
    independent evidence allows: histology/SNR for physical death, matcher FPR/FNR on a labeled
    subset for algorithmic loss.
  - Crucially: report the **null** as a finding. "Before vs after spike processing shows no
    difference in C1 - C3" means the biological-loss reading is unsupported — that is a result, not
    a failure.

### 3.4 P4 — Rename and bound the residual. (fixes H4)
Call `C3 - C2b` **"cross-session drift (residual)."** Add a within-B reference decoder and a
cross-validated tuning-curve-stability measure to bound how much of it is real tuning change vs
behavior/sampling nuisance.

### 3.5 P5 — Stress the gain estimator. (fixes H5, H12)
Recompute the gain component under alternative normalization models: per-condition z-score,
variance-stabilizing (sqrt / Anscombe) transform, median/MAD, rate-only (no variance) rescaling.
If the gain number is stable across these, it is real; if it swings, it was a normalization artifact.
Separately, test whether "gain" carries task information by decoding *state* from the discarded
component — if it does, it is signal, not nuisance.

### 3.6 P6/P7 — Ceiling and comparability. (fixes H6, H7)
Always report **raw Δaccuracy** alongside percentages. Add a ceiling reference (best achievable
intra-session R² across the pair) and show the split is invariant across >=2 decoder families before
claiming comparability across decoders/time gaps.

### 3.7 P8/P9 — Pre-register and infer correctly. (fixes H8, H9)
Pre-register matching as **tuning/activity-blind**; report matcher FPR/FNR on a hand-labeled subset.
Use paired/blocked bootstrap over trials and units; session-blocked nested CV.

### 3.8 P10/P11 — Softer claims. (fixes H10, H11)
Frame magnitudes as directional, not transferred. Reframe Prediction 4 to "no method *optimizes for*
unit-identity recovery."

### 3.9 Summary table

| Hole | Patch | One-line move |
|------|-------|---------------|
| H1 path dependence | P1 | value lattice + Shapley + explicit interaction term |
| H2 dimensionality | P2 | random-drop & SNR-drop controls; report excess |
| H3 non-identifiable loss | P3 | stage as a factor; rename; report the null |
| H4 residual as drift | P4 | rename; bound with within-B reference |
| H5 gain model | P5 | stress across normalizations |
| H6 ceiling | P6 | report raw + ceiling reference |
| H7 comparability | P7 | raw Δ + multi-decoder invariance |
| H8 circularity | P8 | pre-register tuning-blind matching |
| H9 inference | P9 | blocked bootstrap + nested CV |
| H10 metric | P10 | directional claims only |
| H11 overclaim | P11 | soften Prediction 4 |
| H12 gain vs signal | P5 | state-decode the discarded component |


---

## 4. Paper "dummy execution"

No code — a tabletop walk-through of running the four conditions on **one** hypothetical session
pair, to see the *shape* of the output and where it breaks. Numbers are **ILLUSTRATIVE placeholders**
(made up to show the arithmetic), not measured. A real run would fill them in.

### 4.1 The hypothetical run
Metric = cross-validated R² of a linear ridge decoding reach direction ([cos, sin]) from spike
counts. One animal, session A vs session B one month apart.

| Condition | R² (illustrative) |
|-----------|-------------------|
| C1  (all units, intra-A)          | 0.90 |
| C3  (matched units, intra-A)      | 0.78 |
| C2b (matched, A→B, per-session)   | 0.55 |
| C2a (matched, A→B, source stats)  | 0.50 |

### 4.2 Proposal arithmetic (what the paper would report)

- Unit loss        = 0.90 - 0.78 = **0.12**
- Representational drift = 0.78 - 0.55 = **0.23**
- Gain shift       = 0.55 - 0.50 = **0.05**
- Total            = 0.90 - 0.50 = **0.40**
- Closure check    = 0.12 + 0.23 + 0.05 = 0.40  ✓ (trivially true — see H1)

As percentages of the 0.40 total: unit loss **30%**, drift **57.5%**, gain **12.5%**.
This is exactly the kind of sentence the proposal wants to publish. Now stress it.

### 4.3 Where the dummy run breaks

**(a) Dimensionality control (H2/P2).** Suppose dropping the same *number* of random alive units
gives `C1 - C3_rand = 0.09`. Then most of the 0.12 "unit loss" (75%) is just fewer inputs. The
*honest* selective-loss number is `0.12 - 0.09 = 0.03` — i.e. 7.5% of the total, not 30%. The
headline flips categories: this task's degradation is dominated by drift (to the extent the
dimensionality is controlled).

**(b) Before/after spike (H3/P3).** Take the 0.12 "unit loss." Did those 22% of units die, or did the
matcher just miss them? Run the same decomposition with (i) raw sorted units and (ii) SLN/DREDge
drift-corrected localizations. If `C1 - C3` is ~0.12 in both, then the "loss" is identical whether
units physically died or were merely untracked — the estimator **cannot tell the two apart**. So
"unit loss = 30%" cannot be read as "30% of degradation is biological death." It is "30% of
degradation is units unavailable *to this decoder at this pipeline stage*." *(This is the gap
flagged: no condition distinguishes before- vs after-spike processing, so the biology reading is
unsupported by the design.)*

**(c) Path dependence / Shapley (H1/P1).** Suppose single-mechanism ablations give
loss-only 0.10, drift-only 0.19, gain-only 0.03 → sum 0.32, but the combined drop is 0.40. Then the
**interaction term is +0.08** (super-additive), and the proposal's "no residual" is false. Shapley
(which averages over orderings) would land somewhere between, e.g. loss 0.115 / drift 0.235 /
gain 0.05 = 0.40 — a *different* split from 0.12/0.23/0.05. Two defensible rules, two different
stories. You must pick and justify one.

**(d) Gain model (H5).** Recompute gain under a variance-stabilizing (sqrt) normalization: suppose it
gives 0.05 -> 0.015. The "gain shift" was mostly a z-score artifact, not impedance/arousal. Or
suppose decoding *state* from the discarded component succeeds: then the "gain" you normalized away
was real signal.

### 4.4 What a trustworthy table looks like

| Component | Proposal (path) | + dim control | Shapley | stable across normalization? |
|-----------|-----------------|---------------|---------|------------------------------|
| units unavailable (not "loss") | 0.12 | 0.03 selective (+0.09 dim) | 0.115 | n/a (stage test) |
| cross-session drift (residual) | 0.23 | 0.23 | 0.235 | yes/no |
| gain shift | 0.05 | 0.05 | 0.05 | 0.05 vs 0.015 |
| interaction residual | — | — | +0.08 | — |
| pipeline-stage delta (before-after) | — | 0.12 vs 0.12 (null) | — | — |

### 4.5 The test as a checklist (dry run order)

1. Pick session pair + define held-in/held-out trials (A held-out must never touch matching).
2. Build unit sets at **two stages**: raw sorted vs SLN/DREDge-corrected. (stage factor, H3)
3. Tuning-blind A↔B match; record per-unit confidence; hand-label a subset for FPR/FNR. (H8)
4. Compute C1, C3, C2b, C2a → proposal components + closure. (as-is)
5. Add `C3_rand` / `C3_snr` → selective-loss excess. (H2)
6. Build value lattice via counterfactual scenarios → Shapley + interaction term. (H1)
7. Stress gain across normalizations; decode state from residuals. (H5/H12)
8. Report raw Δ and %; check invariance across >=2 decoder families. (H6/H7)
9. Blocked bootstrap CIs on every component; nested CV. (H9)
10. Report the before/after-spike **null** explicitly. (H3)

### 4.6 Bottom line of the dummy execution
The framework is a good *scaffold* and the three knobs are the right knobs, but as specified:
- the closure "exactness" is vacuous and the split is arbitrary (H1);
- the largest bucket ("unit loss") is partly a dimensionality artifact (H2) and is not
  biologically identifiable at all (H3) — the before/after-spike null makes that concrete;
- the "drift" bucket is a residual wearing a mechanism's name (H4);
- the "gain" bucket is only as good as the z-score model (H5).
With patches P1–P3 it becomes defensible; without them, reviewers will (correctly) say the
percentages are model-dependent and the mechanism labels are not identified.


---

## 5. Open thoughts / threads to pull

- **The strongest version of the paper is P1 + P3, not the current C1/C3/C2b/C2a story.** The
  "value lattice + Shapley + explicit interaction term" turns "exact, no residual" from a weakness
  into the actual contribution: *quantifying the interaction* between drift, loss, and gain is more
  interesting than assuming it away.
- **P3 (before/after spike) may be the real headline.** The lab already owns SLN/DREDge. Framing the
  decomposition as "how much apparent unit loss is biological vs a processing artifact" is a cleaner,
  more falsifiable claim than "loss is 40-60%." It also connects directly to the JL note's
  pre-detection vs post-detection question.
- **Naming matters for the whole paper.** Rename all three buckets to be operational:
  "units unavailable to the cross-session decoder" / "cross-session drift (residual)" /
  "gain (per-session affine)." The mechanistic names are what invite the fatal critiques.
- **The Ca-imaging anchor (59.5/31.6/8.9) is doing a lot of rhetorical work.** Since the metric is not
  transferable (H10), consider demoting it from "target numbers" to "one prior decomposition, with a
  different metric."
- **What would make this ML-venue vs neuro-venue?** The interaction-term/attribution machinery (P1)
  is ML-flavored; the before/after-spike biological identifiability (P3) is neuro-flavored. The
  proposal defers this to Phase C — but the choice actually determines the core contribution, so
  decide earlier.
- **Cheap first experiment.** Even without new recordings, the SLN repo (kalman-test) already has
  multi-session predictions; a "before vs after drift correction" C1-C3 contrast on that data would
  immediately test whether the unit-loss bucket moves with pipeline stage (H3) — no new data needed.

## 6. One-line question to the PI
"Does 'unit loss' mean *biological death* or *units unavailable to the decoder*, and does the design
have any condition that could ever tell those apart?"

