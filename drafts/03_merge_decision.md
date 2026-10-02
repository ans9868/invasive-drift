# Should the two be merged into one project? — decision note

**Status:** thinking notes (no code). Follows `02_matchup_decomposition_vs_forecast.md`.

The two were meant to be separate. The question: should they be *made together*?

---

## 0. Short answer

**Yes, it makes sense — but only as a specific *compositional* merge, not a concatenation.**

- **Bad merge:** one paper with a "decomposition section" and a "forecasting section." That is two
  papers stapled together: scope bloat, fuzzy contribution, and it repeats the forecast note's own
  warning ("G must not be a second foundation model"). Reviewers say "this is two papers."
- **Good merge:** wire the two together so each *feeds* the other:
  - the **decomposition produces the labels** the forecast needs (kills F1/F6 — no hand-typed classes),
  - the **panel produces the covariates** the decomposition needs (kills H3 — unit loss becomes
    identifiable).
  One thesis, one pipeline, one dataset.

The good merge has a name: **"forecast the *mechanisms*, not just the error."**

## 1. Why they were separate (and that was reasonable)

| | Decomposition | Forecast |
|--|---------------|----------|
| Endpoint | 3 mechanism percentages (offline) | lead time + class + policy (online) |
| Math | nested differences + Shapley | hierarchical Bayesian GLM |
| Community | ML / measurement | neuro / translational / clinical |
| Label need | label-light | label-heavy |
| Venue logic | NeurIPS/ICLR vs eLife/NatComm | clinical/neuro |

Different endpoints and different reviewer communities is a real reason to keep them apart. That
reason has not gone away.

## 2. Why merging is now attractive

Because (doc `02`): **each one's fatal hole is the other's core strength.**
- Decomposition's H3 (unit loss not identifiable) → solved by the forecast note's panel features.
- Forecast's F1/F2 (sensor-defined labels that are then "predicted" from the same sensors) → solved by
  the decomposition as a label-free, retrospective attribution.

That mutual-gap-filling is unusual and is the whole argument for combining.

---

## 3. Bad merge vs good merge

| | Bad merge ("do both") | Good merge ("wire them") |
|--|------------------------|--------------------------|
| Structure | Section 2 = decomposition, Section 5 = forecasting | One pipeline where stages hand outputs to each other |
| Labels | hand-typed classes (noisy, circular) | decomposition components = the forecast targets |
| Contribution | "we did two things" | one claim: forecast the mechanisms |
| Failure mode | scope rejection | harder labels, but honest |

## 4. The unified thesis

> **Type iBCI drift physically; measure its mechanisms offline with the decomposition; use the panel to
> forecast those mechanisms online; act with a typed correction.**

The key move is that **the decomposition is not a parallel analysis — it is the label generator for the
forecast, and the panel is the covariate set for the decomposition.** They are wired, not stacked.

## 5. The merged pipeline (4 stages)

- **Stage 0 — TYPE.** The forecast note's 6-class taxonomy. Shared vocabulary. Cheap.
- **Stage 1 — ATTRIBUTE (offline).** Run the decomposition on session pairs to produce, per pair, a
  vector (units-unavailable, drift, gain) — *with* the P1/P2/P3 fixes (Shapley, dim control, stage
  factor). Output: clean mechanism labels.
- **Stage 2 — FORECAST (online).** Panel features at `t` (the note's three clocks) predict **the
  Stage-1 mechanism components** (and the crash indicator) at horizon `h`. This replaces "predict a
  hand-typed class" with "predict a measured mechanism vector."
- **Stage 3 — CORRECT.** Class-conditional map `G` + policy (mask/stitch/adapter/pause), chosen from the
  forecast mechanism.

**Cross-wiring (this is the merge):**
- Stage-1 components ➜ Stage-2 **targets** (fixes F1/F2/F6).
- Stage-2 panel features ➜ Stage-1 **covariates** that split "unit loss" into death / isolation /
  matcher-failure (fixes H3).
- Stage-0 taxonomy names both the Stage-1 components and the Stage-2 classes (one vocabulary).

## 6. What the merge buys

1. **No hand-typed labels.** The decomposition generates the forecast's targets. The circular
   "sensor-defined class predicted from sensors" problem (F1) disappears — the target is a *measured,
   model-audited* quantity.
2. **Identifiable unit loss.** Panel covariates enter the decomposition; H3 (the fatal hole) closes.
3. **One dataset, one leak discipline, one taxonomy.** Cheaper than two projects, and the
   session-out/animal-out + past-only rules are shared.
4. **A single memorable claim.** "We forecast the *mechanisms* of decoder failure, not just a scalar
   error" — that is a headline neither doc has alone.


---

## 7. What it costs

1. **Experimental surface roughly doubles.** The roadmap becomes the union of both: decomposition
   (Phases A–D) + panel/forecast + policy + `G`.
2. **Label uncertainty propagates.** If the Stage-1 components are themselves model-dependent (doc
   `00`: path-dependence, dim confound), then forecasting them inherits that uncertainty. The merged
   paper *must* carry the Shapley/controls through into the targets, and report forecast skill against
   the *uncertainty* on the labels.
3. **Venue identity crisis.** Not a pure ML paper (forecasting/policy) and not a pure neuro paper
   (attribution machinery). Likely lands at a translational/interdisciplinary venue
   (Nat Neuro / Nat Biomed Eng / Nat Comm), not NeurIPS.
4. **Two infrastructures at once.** Unit-matching pipeline + lesion-replay/label instrumentation +
   policy evaluation. More engineering before any result.

## 8. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Scope rejection ("two papers") | one thesis + one pipeline; move `G` to a single figure or companion |
| Forecasting model-dependent labels | freeze the decomposition config; forecast the *components with CIs*, score against them |
| Ramp vs step confusion | adopt the note's honest split: forecast ramps, detect steps fast |
| Policy has no break-even | do the marginal-value arithmetic **before** building the merged paper |
| Two audiences | pick one primary venue; frame the other arm as a figure, not a co-headline |

## 9. Decision rule (pick one)

- **Safest (recommended if time-pressured):** keep **two sibling papers** sharing the taxonomy, each
  cross-citing the other; do the mutual gap-filling as a *joint figure* in both. Low risk, still tells
  the program story.
- **Middle:** submit as a **linked pair** (Paper A = attribution; Paper B = forecast), same data,
  reviewed as a package. The taxonomy is the explicit glue.
- **Flagship (if you have the runway):** **merge compositionally** — decomposition as label generator,
  panel as predictor. Highest ceiling, highest cost.

## 10. Minimal viable merged paper (if you merge)

1. **Fig 1** — taxonomy (Stage 0).
2. **Fig 2** — decomposition on one dataset, with P1/P2/P3 controls; the mechanism labels (Stage 1).
3. **Fig 3** — panel predicts the mechanism components at horizon h; beats persistence + MINDFUL
   scalar (Stage 2).
4. **Fig 4** — typed correction + policy break-even vs always-recal (Stage 3).
5. **Fig 5** — consistency: forecast class ↔ dominant decomposition bucket, session by session.

Everything above Fig 2 already exists as sketches in the two docs; the merge is mostly Fig 3.

## 11. One-sentence gut check

State the merged claim in one sentence **without the word "and."** If you can't, it is two papers.

- *Works:* "We type iBCI drift physically and forecast the mechanisms of decoder degradation before
  they manifest." (forecasting the *mechanisms* implies the measurement — no "and".)
- *Fails:* "We decompose the drop **and** forecast failure." → two papers.

---

## 12. My recommendation

Combine — but **compositionally, and stage the risk**: build Stage 1 (decomposition) first, use it to
generate labels, then attempt Stage 2 (predict the mechanisms). If Stage 2 clears its persistence and
policy break-even bars, you have the flagship merged paper. If it doesn't, you still have two clean
sibling papers (the decomposition stands alone; the forecast stands alone with hand labels as a
fallback). **The merge is an upside option, not a forced marriage.**

