# Match-up: the decomposition proposal and the forecast note

**Status:** thinking notes (no code). This is the "do they match up?" doc.

Two source documents (both local copies in `drafts/`):

1. **Decomposition proposal** — `drafts/invasive_drift_decomposition_proposal.pdf`
   (orig: `/Volumes/CrucialX6/Home/Zotero/storage/2X4EU83K/invasive_drift_decomposition_proposal.pdf`)
   -> analyzed in `00_breakdown_holes_buildup.md`.
2. **Forecast note** — `drafts/bci_drift_forecast_note-2.pdf`
   (orig: `/Volumes/CrucialX6/Home/Zotero PDFs/storage/bci_drift_forecast_note-2.pdf`)
   -> analyzed in `01_forecast_note_breakdown.md`.

---

## 1. One line each

- **Decomposition:** *retrospective attribution* — take the accuracy drop between two sessions and
  split it into unit loss / drift / gain, offline.
- **Forecast:** *prospective typing + prediction* — type the drift physically, monitor it on three
  clocks, forecast decoder failure, and optionally learn a map `G`.

They are not competitors. They are the **same object seen from opposite ends of time**.

## 2. They are two arms of one program

| Axis | Decomposition | Forecast |
|------|---------------|----------|
| When | offline, retrospective | online, prospective |
| Question | "*what* did the drop come from?" | "*when* is it coming, what type, what to do?" |
| Output | 3 mechanism percentages | lead time + class + policy + map G |
| Timescale | hours–days (session↔session) | s / min / day (three clocks) |
| Needs labels? | no (label-light) | yes (class labels) |
| Main risk | arbitrary attribution (H1), non-identifiable loss (H3) | sensor/label circularity (F1), detection≠forecast (F2) |

The decomposition is literally *the endpoint of the forecast note's slowest clock* (vs yesterday /
last-good day). One is the derivative (forecast), the other is the integral (attribution).

## 3. The shared spine is the *typed drift taxonomy*

The forecast note's six classes are a physical typing of exactly the degradation the decomposition
tries to measure. They line up like this:

| Decomposition mechanism | Forecast taxonomy class(es) | Comment |
|-------------------------|-----------------------------|---------|
| **Unit loss** (C1−C3) | motion/registration + channel death/isolation | note splits it into two *causes* the decomp lumps — this is the fix for H3 |
| **Representational drift** (C3−C2b) | code warp (+ motion tail) | "code warp" = the preferred-direction walk / factor-angle change |
| **Gain shift** (C2b−C2a) | state (arousal/fatigue) + impedance/amplitude | note says "gate or add a state feature," i.e. do NOT normalize it away (contra decomp H12) |
| *(missing in decomp)* | reference/artifact | decomp has no artifact condition |
| *(missing in decomp)* | decoder/task confound | decomp can't see task confounds |

**Read:** the decomposition has 3 buckets; the note has 6 classes — two of which the decomposition
cannot represent at all (artifact, task confound), and two of which it *merges* (motion + channel
death). That merge is precisely the H3 hole.

---

## 4. They fix each other's weakest holes

This is the strongest argument that they "match up": each one's fatal hole is the other one's core
strength.

### 4.1 The forecast note supplies the missing identifiability for decomp H3.
Decomp's worst hole: "unit loss" bundles *dead + isolation-failure + matcher-failure* and the four
conditions can't tell them apart. The note's **panel features are exactly the covariates that can**:
fraction dead, channel SNR / rate / impedance, DREDge position & velocity, **unit-match survival**, and
the Kilosort / UnitMatch / Bombcell "death vs isolation" metrics. In other words, the note contains the
instrument that makes the decomposition's largest bucket biologically identifiable.
-> **Fix for H3 = import the note's panel as a decomposition covariate.**

### 4.2 The decomposition supplies the label-free ground truth for the forecast's typing.
Forecast's worst hole (F1/F2): classes are defined by sensor signatures and then "predicted" from the
same sensors — circular; and contemporaneous state is confused with forward forecast. The
decomposition is a **label-light, retrospective attribution** that needs *no* class labels. So it can
independently *check* the forecast's typing offline: did a session that the panel tagged "code warp"
actually show up as a dominant *drift* residual in the decomposition? If yes, the typing is validated;
if no, the panel mislabeled.
-> **Fix for F1/F2 = use the decomposition as an external consistency test on the class labels.**

### 4.3 Shared: both need the same leakage/time-symmetry discipline.
The note is explicit ("do not use a decoder already updated on the future. That is the leak"). The
decomposition is under-specified here (H9). The note's discipline should be inherited wholesale.

### 4.4 Shared: both ride on the same data.
Both name the same public multi-day benchmarks (brainsets / DANDI / Perich–Miller; Neuropixels +
DREDge; IBL). One dataset can serve both arms.

### 4.5 Tension: they disagree on what to do with gain.
Decomp treats gain as *nuisance* ("per-session normalization removes it"). The note treats state/gain
as *signal* ("gate or add a state feature; do not rebuild the head on tired"). This is decomp H12, and
the note already took the more sophisticated position. The merged program should side with the note.

## 5. The timescale bridge

- Decomp = the **vs-yesterday/last-good-day** clock, collapsed to a single number per session pair.
- Forecast = that clock *plus* the 5–20 min and 10–30 s clocks.
So the decomposition is a **special case** of the forecast program at the slowest clock and the longest
horizon. A merged paper would show the decomposition as the "days" rung and the panel as the
seconds/minutes rungs.

## 6. The unified 4-part program

Neither document names the whole thing, but together they are a clean four-step arc:

1. **TYPE** it (forecast note §0 taxonomy) — the shared vocabulary.
2. **ATTRIBUTE** it (decomposition) — offline, how much of the drop is which mechanism.
3. **FORECAST** it (panel + clocks) — online, when and what type, with lead time.
4. **CORRECT** it (map `G`, conditioned on the typed alarm) — what to do.

Where each doc sits today: decomposition = step 2; forecast note = steps 1 + 3 (+ step 4 sketched).
The pitch no one has written yet is the *integration*: a typed taxonomy that is measured offline
(step 2), predicted online (step 3), and corrected (step 4).

## 7. Ownership / positioning tension

Both docs can claim "typed drift." To avoid a collision:
- **Decomposition paper owns:** retrospective mechanistic attribution + the interaction/Shapley
  machinery + "what is / isn't identifiable."
- **Forecast paper owns:** prospective typed forecasting + policy + `G`.
- **Shared, cited-once:** the taxonomy (§0 of the note).
- **Danger:** bundling taxonomy + decomposition + panel + `G` into one paper repeats the note's own
  warning ("G must not be a second foundation model"). Keep them as siblings; the *synthesis* is a
  third, later paper (which also matches the decomposition proposal's stated "future synthesis paper"
  connecting EEG + BCI + connectomics).


---

## 8. Shared holes (fix once, use twice)

| Shared weakness | Appears in | Shared fix |
|-----------------|-----------|------------|
| Leakage / time-symmetry | decomp H9, forecast F1/F2 | session-blocked nested CV; frozen decoder; features strictly past-only |
| No power / decision analysis | decomp H7, forecast F4 | pre-register the endpoint; compute break-even before modeling |
| Metric ↔ harm gap | decomp H10, forecast F3 | report raw + proxy; show the proxy tracks an external harm measure |
| Rare-event statistics | decomp H9, forecast F4/F7 | effective-N per cell; support-aware CIs |

Doing these once across both papers is cheaper than doing them twice, and it makes the *synthesis*
paper credible.

## 9. Match-up "dummy execution": the same crash seen from both ends

One story, one animal, a crash on day 30. Walk both frameworks and check they agree. Numbers are
**ILLUSTRATIVE**.

**Offline (decomposition), day 1 -> day 30:**
C1 = 0.90, C3 = 0.78, C2b = 0.55, C2a = 0.50
-> units-unavailable 0.12, drift 0.23, gain 0.05; with the random-drop control, selective loss only
0.03 (the rest is dimensionality). Dominant mechanism = **drift**.

**Online (forecast), during day 30:** the panel fires ~20 min before the crash, class = **"code warp"**,
recommends an adapter (not recalibration).

**Consistency check (the point of the match-up):**
- Decomp says drift dominates; forecast says code warp. **Agree** -> typing validated.
- Now suppose the forecast had said "channel death" while decomp says drift dominates -> **conflict**;
  one framework is wrong, and that disagreement is itself a publishable finding.

**Where each end fails on this story:**
- Decomp can't say *when* within day 30 the decoder started dying (it's a single pair of numbers).
- Forecast can't say how much of day-30's drop was *biological* vs *processing* (it has no
  decomposition) — unless you import the panel as covariate (4.1) and the decomp arm as label check
  (4.2).
**Together** they say: drift dominated, it was a code warp, it began ~20 min before failure, and the
right move was an adapter. Neither doc produces that sentence alone.

**The minimum experiment that tests the match-up:** take one lab multi-session dataset (the
kalman-test 33-session set), run the decomposition offline per session pair, run the panel online, and
tabulate agreement between {dominant decomposition bucket} and {forecast class} session by session.
Agreement rate *is* the evidence that the two frameworks describe the same object.

## 10. Open thoughts / what I'd decide next

- **Decide ownership now, not at Phase C.** Decomposition = attribution paper; forecast = forecast+policy
  paper; taxonomy is shared; synthesis is a later paper. (Mirrors the decomp proposal's own deferred
  synthesis plan.)
- **The match-up is a selling point, not a merge.** Two siblings sharing a taxonomy and a leakage
  discipline reads as a program; one mega-paper reads as unfocused.
- **Import the panel as a decomposition covariate is the single highest-value, lowest-cost experiment**
  (fixes the decomp's fatal H3 and gives the forecast an external label check).
- **One-line question to the PI:** "Are these meant to be one paper or two siblings — and does the
  decomposition import the panel features (death vs isolation vs match-survival) to make 'unit loss'
  identifiable, or is it staying as the four binary conditions?"

