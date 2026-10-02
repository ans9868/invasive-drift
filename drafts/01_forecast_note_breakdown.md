# Forecast-note break-down, holes, build-up, and a paper "dummy execution"

**Status:** thinking notes (no code). Companion to `00_breakdown_holes_buildup.md`.

**Source (full path):**
`/Volumes/CrucialX6/Home/Zotero PDFs/storage/bci_drift_forecast_note-2.pdf`
(byte-identical to `/Volumes/CrucialX6/Home/Downloads/bci_drift_forecast_note-2.pdf`).
Local copy: `drafts/bci_drift_forecast_note-2.pdf`

Title: *"Forecasting iBCI decoder failure — clocks, a linear Bayes panel, and a map G."*
Subtitle: *"Motor-first project sketch."*

---

## 1. Break it down

This note is **three deliverables stapled together**, sharing one thing (a physical *taxonomy* of drift):

### 1.1 Arm 0 — a physical / literature taxonomy of drift
Six classes, each with physical story, signature, timescale horizon, and "right move":

| Class | Story | Signature | Horizon | Right move |
|-------|-------|-----------|---------|------------|
| Motion / registration | probe creeps in tissue (DREDge) | coherent waveform shift along depth | s–min (onset), slow creep | remap sites/unit IDs |
| Channel death / isolation | gliosis, encapsulation, bad threshold | rate→0, SNR collapse, impedance, waveform gone | hr–wk (scar), sudden if mechanical | mask channel |
| Reference / artifact | cable, pump, common-mode | many channels jump together | seconds | discard segment |
| State (arousal/fatigue) | neuromod, behavior; geometry unchanged | LFP low-freq/beta, population rate scale | minutes | gate or add state feature |
| Code warp | same units, map to kinematics rotated/gained | factor-angle vs rest, preferred-dir walk, soft R² fade | min–days | adapter / stitch / map G |
| Decoder / task confound | inner speech, rest-vs-attempt, LM lock-in | features fine, posteriors collapse | utterance | fix the gate |

Framing: "chronic iBCI papers treat nonstationarity as a fact; they rarely type it." The taxonomy *is*
the first deliverable.

### 1.2 Arm 1 — three clocks on a typed sensor panel
Same sensors read at three timescales, each as **level + short difference**:
- vs yesterday / last-good day (hours–days; "warp and scar")
- vs 5–20 min ago (fatigue, gain walk, isolation getting messy)
- vs 10–30 s ago (motion/artifact/sleep-spin onset)

Plus a **horizon-plausibility table** (what physics allows to be forecast at 1–30 s, 1–20 min,
hours–days vs what is *not* plausible). Key admission: **step faults are not forecastable** (cable
yank, unit death at 14:00).

### 1.3 Arm 2 — fuse with a linear / GLM Bayes model, then policy
- Hierarchical (animal-and-day intercepts) + **regularized horseshoe**; predeclare a few interactions
  (LFP-slope × SNR-level, motion-velocity × unit-survival, angle × rest-only correlation).
- Target `y` = next-horizon **delta R²** (Gaussian) or **crash / severity** (logistic / ordered logit).
- Labels come from "sensors + injected lesions + lab notes"; severity from a frozen reference decoder.
- Explosion warning: "**do not use a decoder already updated on the future. That is the leak.**"
- Evaluation: session-out + animal-out; lead-time distribution; **false alarms per hour**; policy value
  (mask/stitch/adapter/pause vs always-recal vs never-recal); **MINDFUL as a nested model** = figure 1.
- Data recipe: public multi-day monkey reaching (natural drift) + Neuropixels/DREDge (motion w/ GT) +
  **replay with scheduled lesions** (stopwatch for lead time) + a second animal for generalization.

### 1.4 Arm 3 (optional) — a class-conditional transport map `G`
`G` on neural inputs so `f(G(x_B)) ≈` the decoder you'd have with labels at B, **conditioned on the
typed alarm** (e.g., motion → permutation/interpolation along the shank; dropout → mask/projection;
state → gain/style or a state bit; code warp → Procrustes/CCA/tiny adapter; artifact → drop). Explicit:
`G` is underdetermined from unpaired data; supervise in order (lesion/DREDge → labeled anchors →
unpaired-rest-only matching).

### 1.5 The pitch
"Instability metrics exist; they are **contemporaneous and untyped**. A physically typed panel on
three clocks, fused by a hierarchical Bayesian GLM, plus an optional class-conditional `G`, evaluated
as a **policy** against daily full recalibration, is the increment."

---

## 2. The holes

Severity: **[fatal]** = undermines the central claim, **[real]** = must fix, **[minor]** = caveat.

### F1. The six classes are defined by their *sensor signatures* — and then forecast from the same sensors. **[fatal]**
"Class comes from sensors + injected lesions + lab notes." But the panel features are also sensor
outputs (SNR, rate, impedance, DREDge, LFP state). So on real data, "predict class from the panel" is
close to re-encoding the label, not forecasting it. Only the **injected-lesion (replay)** arm has a
label source that is independent of the features. This must be handled explicitly (see G2), or the
whole "typed forecasting" claim collapses into "the sensors know what the sensors say."

### F2. State-classification and forward-forecasting are conflated. **[fatal]**
The pitch is *forecasting* (delta R² at horizon h, crash at h). But much of the panel is
*contemporaneous* state. Predicting "crash now" from "SNR collapsing now" is detection, not forecast.
The defensible contribution is the **forward** part: features at `t` predicting an event at `t+h`
that is *not yet present in the features*. The note gestures at this but the design doesn't force it.

### F3. The horizon the panel targets (10 s–30 min) is not the timescale where the only validated harm-link lives. **[real]**
MINDFUL correlates (r≈0.7–0.9) with closed-loop error **over weeks** — the hours–days clock. The panel
mostly hunts 10 s–30 min ramps. Nothing in the note shows the short-horizon "delta R²" proxy tracks
the closed-loop user harm that motivates the paper. Offline delta R² ≠ user harm (they say motor
"understates harm," but the deeper issue is offline-vs-closed-loop, not motor-vs-speech).

### F4. Power / decision analysis is missing for the policy claim. **[real]**
"Policy value vs always-recal vs never-recal" only *means* something if you know crash prevalence,
false-alarm cost, and the cost of always-recal. If recalibration is cheap and rare, forecasting adds
nothing. The note mentions a decision threshold but never does the marginal-value arithmetic that
decides whether the paper has a result.

### F5. By the note's own taxonomy, the worst failures are step faults — which it admits it cannot forecast. **[real]**
Channel death at 14:00, cable yank: "not plausible" to forecast; "right move" is mask/discard
(reactive). So "forecasting decoder failure" oversells the physically possible headline. The honest
title is "forecasting *ramping* failure + fast detection of step faults."

### F6. Class ground-truth is the linchpin and the weakest part. **[real]**
Six-way class labels from "lab notes" are noisy and un-audited; injected lesions exist only in
replay. If the class head trains on noisy labels, the whole "typed" payload inherits that noise. No
inter-rater / label-quality / sim-to-real gap discussion.

### F7. Uneven data support across the three clocks distorts "which clock carries lead time." **[real]**
The 1–30 s clock has enormous data; the hours–days clock has ~session count. A hierarchical model
will attribute lead time partly by data volume, not physics. The "which clock carries lead time" claim
needs explicit per-clock support accounting.

### F8. `G` is a second project wearing the same title. **[real]**
They themselves warn "G must not be a second foundation model," yet arm 3 doubles the deliverable and
its evaluation ("f∘G vs retrain vs nothing") is a *different* question from forecasting. Bundling
risks neither half being convincing.

### F9. Simulated lead-time and real lead-time are not kept apart. **[real]**
The "stopwatch" lives in lesion replay; generalization lives in real data. If results tables mix them,
reviewers will read simulation lead-time as real. Needs a hard separation in every table.

### F10. The increment over MINDFUL/NoMAD is interpretive, not obviously predictive. **[real]**
Positioning is sharp ("beat MINDFUL on specificity," "NoMAD is G without a forecast"). But if a
scalar already gets r≈0.8, the burden is to show *typing changes the policy outcome*, not just the
posterior table. The nested-model ladder (MINDFUL → +clocks → +typed panel) must show a real lift.

### F11. Vocabulary drift. **[minor]**
"Drift," "nonstationarity," and "decoder failure" are used for the latent process, its signature, and
the harm. The note mostly types things well, but the title uses them loosely.

### F12. "Linear/GLM not a net" + "tree on the residuals / second-stage class head" partially undercuts the interpretability rationale. **[minor]**
The escape hatch is fine, but it weakens "coefficients are interpretable" if the residual model
carries the signal.


---

## 3. Build up from the holes (patches)

### 3.1 G1 — Make the forecast target single and frozen. (fixes F2/F3)
One primary endpoint (delta R² of a **day-1 frozen decoder** on the next block), one horizon set,
pre-registered. Everything else secondary. Prove the proxy tracks harm: correlate short-horizon delta
R² against an independent closed-loop / behavioral harm measure on at least one cohort. If it doesn't,
downgrade the claim from "user harm" to "offline decoder health."

### 3.2 G2 — Force forward-ness and break the feature/label circularity. (fixes F1)
- Define **forecast-only** metrics: event at `t+h` that is *absent* from features at `t`. Report a
  "persistence baseline" (does `t` already predict `t+h` trivially?). Beat persistence, not just chance.
- Get at least one **label source independent of the panel** (behavioral error, closed-loop bits/s,
  experimenter log of a mechanical event). Show class prediction holds against the independent source.
- On real data, report class performance *only* where the label is instrument-derived, not
  sensor-derived.

### 3.3 G3 — Keep simulated and real lead-time in separate universes. (fixes F9)
Every lead-time figure is tagged {replay-lesion | natural drift}. The "6 hours early" headline is
banned unless it reproduces on natural drift. Lesion replay is the *calibration* set, and its
sim-to-real gap is stated.

### 3.4 G4 — Do the policy marginal-value arithmetic first. (fixes F4)
Before modeling: given crash prevalence p, false-alarm cost, miss cost, and always-recal cost, compute
the break-even detection quality. If no realistic panel clears break-even, the policy claim is dead on
arrival and the paper becomes a forecasting (not policy) paper.

### 3.5 G5 — Retitle around what is physically forecastable. (fixes F5)
"Forecasting *ramping* decoder failure and fast detection of step faults." Detection latency for step
faults becomes a first-class metric (time-to-alarm), not a footnote.

### 3.6 G6 — Audit the labels. (fixes F6)
Inter-rater agreement on class labels for a hand-annotated subset; treat injected lesions as ground
truth only in simulation; report class-head performance as a function of label noise. Publish the
label schema in the supplement.

### 3.7 G7 — Account for per-clock support. (fixes F7)
Report effective N per clock; use partial pooling explicitly to borrow strength for the sparse
hours–days clock; present "which clock carries lead time" with support-aware credible intervals (not
raw coefficient magnitude).

### 3.8 G8 — Demote / split `G`. (fixes F8)
Either (a) publish the panel+policy as paper 1 and `G` as paper 2, or (b) keep `G` as one figure with
a pre-declared "what G cannot do" list (can't resurrect dead channels; underdetermined from unpaired).
Do not let `G`'s evaluation share a results table with the forecast.

### 3.9 G9 — The nested-model ladder is the increment; make it figure 1. (fixes F10)
MINDFUL scalar → + clocks → + typed panel → + interactions, each with LOO / likelihood-ratio vs the
rung below. The increment claim = the lift of the last two rungs on the **policy** metric, not AUROC
alone.

### 3.10 G10 — Calibration discipline. (fixes F10/F11)
Reliability diagrams + PSIS-LOO; the note already flags "posteriors lie." Add precision/recall at the
operating threshold the policy would actually use.


---

## 4. Paper "dummy execution"

Tabletop run on one hypothetical multi-day motor animal. Numbers are **ILLUSTRATIVE placeholders**.

### 4.1 Setup
- Frozen day-1 decoder; panel read at t; horizon h = 30 min; event = "crash" (delta R² < 0.1).
- One animal, N sessions; one session held out for animal-out.

### 4.2 What a run reports (illustrative)

| Model | crash@30min AUROC | lead time (mean) | false alarms/hr |
|-------|-------------------|------------------|-----------------|
| chance | 0.50 | — | — |
| MINDFUL scalar (nested) | 0.80 | ~0 (contemporaneous) | — |
| + three clocks | 0.83 | minutes | 1.1 |
| + typed panel | 0.86 | minutes | 0.7 |
| **leak** (+ current decode entropy) | 0.96 | 30 min (fake) | 0.1 |

The **leak row** is the lesson (F1/F2): adding a contemporaneous feature fakes a huge jump. The honest
increment is 0.80 → 0.86 — real but modest, and it is the *typing* that buys the false-alarm drop.

### 4.3 Where the dummy run breaks

**(a) Persistence baseline (G2).** If a "predict crash at t+30 from features at t" model barely beats
a persistence baseline ("was it already bad at t?"), the panel is detecting, not forecasting.

**(b) Lead-time by source (G3/F9).** Lesion replay says lead time ≈ 6 min; natural drift says ≈ 0–1
min. The "forecast" story lives in simulation. If reviewers see only the replay number they will
over-credit the panel.

**(c) Policy break-even (G4/F4).** Suppose always-recal costs 1 unit, a missed crash costs 10, crash
prevalence = 2%. The forecast policy saves cost only if false-alarm rate < ~0.2/hr. If the panel runs
at 0.7/hr, **always-recal wins** and the policy arm has no result — even though AUROC looked good.

**(d) Class typing vs scalar (G9/F10).** If the typed panel's AUROC lift over the MINDFUL scalar is
within CI, then "typing" bought *interpretation* (which clock, which class) but not *prediction*.
Still publishable — but the headline changes from "we forecast better" to "we forecast *the same* and
explain *why*, which changes the policy."

**(e) Label noise (G6/F6).** If injected-lesion classes give 0.95 class accuracy but lab-note classes
give 0.6, the "typed" claim is simulation-only until labels are audited.

### 4.4 Bottom line of the dummy execution
The note's *taxonomy* and *leakage discipline* are its strongest assets. Its *forecast* and *policy*
claims are the fragile ones: they depend on (i) beating a persistence baseline, (ii) lead-time
reproducing on natural drift, (iii) a policy break-even that may not favor forecasting, and (iv)
independent class labels. Patches G1–G4 decide whether this is a forecasting+policy paper or a "typed
monitoring" paper.

