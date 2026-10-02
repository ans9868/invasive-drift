# Prediction-first: a feasibility test of the "functional" (forecast) arm

**Status:** thinking notes / test spec (no code yet). Follows `01` (forecast note) and `03` (merge).

The call: **test whether we can *predict* first**, and only if that clears, use the working
predictor as the handle to **break down which mechanisms** it is tracking. This reverses the
"decomposition-first" order in `03 §12` — and that reversal is defensible (see §0).

---

## 0. Why prediction-first is the right de-risking move

- The **whole merged program is load-bearing on prediction.** If you cannot forecast degradation
  ahead of time, then generating decomposition labels for a forecast (the compositional merge in `03`)
  has nothing to attach to.
- **Prediction is the cheaper test.** It needs time-series features + a decoder metric. It does *not*
  need the matched-unit infrastructure, labeled matching subsets, or class labels that the
  decomposition and the forecast-note class heads both depend on.
- So: prediction is the **assumption**, and it's the **cheapest assumption to falsify**. Test it first.
- The decomposition still matters — but it becomes the *second* question: "given a prediction, which
  mechanism is it tracking?" (§6). That ordering matches the user's instinct exactly.

Caveat that must be stated up front: a **synthetic** feasibility test only validates the *harness*
(that the code/target/nulls are wired correctly). Whether *real* iBCI drift is predictable is an
empirical question that only real multi-session data can answer. Keep those two claims separate.

## 1. The one falsifiable question

> **Do features strictly before time `t` predict decoder health on `(t, t+h]` *better than a
> persistence baseline*?**

That single clause — **"better than persistence"** — is the crux. Without it, "prediction" is just
detection: "was the decoder already bad at `t`?" A persistence baseline (`health(t+h) ~ health(t)`)
must be beaten, at every horizon, or there is no forecast.

Secondary falsifiable clauses:
- Does any signal carry **lead time** (i.e., predict the crash before it appears in `health(t)`)?
- Is the lead time the **physics** allows (ramps), or is the model just re-detecting?

## 2. The feasibility ladder (cheapest first)

| Tier | Question | Data needed | Cost | What it proves |
|------|----------|-------------|------|----------------|
| **T0 — harness** | Does the pipeline run, split correctly, and reproduce a known signal? | synthetic | hours | the code/target/nulls are wired right (no science) |
| **T1 — session-scale** | Does session `N`'s feature panel predict session `N+1`'s decoder drop? | >= 3–5 sessions | days | the *slow clock* is forecastable at all |
| **T2 — within-session ramp** | Does the 5–20 min panel predict the next block's drop? | one long session, blocked | days | the *middle clock* carries usable lead time |
| **T3 — step-fault latency** | How fast can a step fault (channel death) be detected? | injected/known events | days | the *fast* arm (detection, not forecast) |

Recommendation: **run T0 (harness) then T1.** T1 is the cleanest go/no-go on the slow clock, and the
slow clock is the one the decomposition lives on (doc `02 §5`). T2/T3 are refinements.

---

## 3. Test design (the T1 spec, used as the template)

**Features (strictly past):** the forecast note's panel, at time `t`:
channel SNR, rate, impedance, fraction-dead, DREDge position/velocity, unit-match survival,
factor-angle vs baseline rest, rest-only noise correlations, LFP band power — each as **level +
short difference** (the three clocks: vs-yesterday, vs-5–20 min, vs-10–30 s).

**Target:** decoder health on `(t, t+h]` from a **frozen day-1 decoder** — primary endpoint = delta R²
(crash = delta R² below a pre-registered threshold).

**Horizons:** `h ∈ {next session}` (T1); later `{10 s, 2 min, 30 min}` (T2/T3).

**Splits:** **session-out** (and **animal-out** once >1 animal). Never let `t+h` trials enter any
fitting that produces features at `t`.

**Baselines (report all four — the paper lives or dies here):**
1. chance (AUROC 0.50 / majority class);
2. **persistence** — `health(t)` predicting `health(t+h)` (the honest bar);
3. MINDFUL-style scalar (nested, one feature);
4. full panel (+ optional interactions).

**Metric buckets:** discrimination (AUROC / PR-AUC at realistic prevalence), **calibration**
(reliability + PSIS-LOO), and **lead time** (distribution, given a crash happened).

**"Feasible" decision rule (pre-register before looking):**
- **GREEN:** panel beats persistence on discrimination **and** calibration is not worse, with the
  increment outside the CI, and lead time > 0 in the physics-allowed range → forecast is real; proceed
  to §6.
- **YELLOW:** beats chance but not persistence → it is **detection/monitoring**, not forecasting →
  reframe the claim (and the decomposition still works as offline attribution).
- **RED:** does not beat chance across horizons → the phenomenon (at this timescale/features) is not
  forecastable → the merged program is in serious trouble; fall back to pure offline decomposition.

## 4. Leakage checklist (each item is a known killer)

- [ ] No feature at `t` is derived from data in `(t, t+h]` (esp. current decode entropy — the fake
      0.96 AUROC in doc `01 §4.2`).
- [ ] Frozen reference decoder; never a decoder updated on the future.
- [ ] Feature normalization uses **past-only** statistics (no session-wide mean/std leaking future).
- [ ] Session/animal split; no trial-level random split across the horizon boundary.
- [ ] Class/severity labels derive from an **independent** source, not the panel features (doc `01` F1).
- [ ] Report the **persistence baseline** on the *same* split (not a global number).

## 5. What would falsify "prediction is feasible"

- Panel ≈ persistence at every horizon (YELLOW→RED).
- Lead time ≈ 0 when tested on **natural** drift (works only in lesion replay) (doc `01` F9).
- Prediction survives only when a contemporaneous feature is included (leak artifact).
- Calibration is poor at the operating threshold even when AUROC looks fine.

If any of the first three hold, the honest output is a **monitoring/attribution** paper, not a
forecast paper.


---

## 6. From prediction → break down the components (the second half)

Only if §5 does *not* falsify, and §3 is GREEN, do you move down the user's plan: use the working
predictor as the handle to attribute the mechanisms.

The bridge is a **two-step** move:

**Step A — is the prediction TYPED?** Ask whether the panel's predictive signal carries mechanism
information, by checking whether the forecast's discriminative features align with a decomposition of
the actual drop. Concretely: on held sessions, run the decomposition (doc `00`) to get
`(units-unavailable, drift, gain)` per session pair, and test whether panel features at `t` predict
**which component** will dominate — not just "a drop is coming." Output: a *mechanism forecast*, not a
scalar forecast.

**Step B — close the loop back into identifiability (H3).** The same panel covariates that predict are
the covariates that make "unit loss" identifiable (doc `02 §4.1`). So importing the panel into the
decomposition is simultaneously (i) the H3 fix and (ii) the "break-down" the user wants.

**Ordering, cleanly:**
1. Can we predict *that* a drop is coming? (§1–§5)  <- feasibility gate
2. If yes, can we predict *which mechanism* dominates? (Step A)
3. If yes, the panel is also the identifiability instrument for the decomposition (Step B) — the merge
   in `03` now has an empirically validated front end.

If step 1 fails, steps 2–3 are moot; stop and publish the offline decomposition instead. That is why
prediction-first is the right sequencing.

## 7. Dummy execution (illustrative numbers)

Tabletop T1 run: one animal, session-out, horizon = next session, crash = delta R² < 0.1.

| Model | AUROC (next-session crash) | calibration (PSIS-LOO) | lead time |
|-------|-----------------------------|-------------------------|-----------|
| chance | 0.50 | — | — |
| persistence `health(t)` | 0.74 | — | 0 (it IS `t`) |
| MINDFUL scalar | 0.71 | ok | ~0 |
| full panel (past-only) | 0.79 | ok | > 0 |
| **leak** (+ decode entropy at t) | 0.93 | ok | fake |

Reading:
- Panel beats persistence (0.79 > 0.74) and MINDFUL (0.79 > 0.71) → **GREEN**; prediction is feasible
  at the slow clock. Proceed.
- The leak row shows how easy it is to fake the result — the persistence baseline is what exposes it.
- Lead time > 0 but modest → the honest claim is "forecastable a bit ahead on the slow clock," not the
  note's more ambitious multi-clock pitch.

Then Step A: among held sessions, the decomposition says drift dominates in 70% of the crash sessions,
and the panel's forecast features load onto drift-type covariates (factor-angle, rate) in those → the
prediction is **typed** → proceed to the compositional merge.

Failure variant: if panel AUROC = 0.73 (below persistence 0.74) → **YELLOW** → the finding is
"the panel is contemporaneous, not predictive," and the program pivots to monitoring/attribution.

## 8. Open questions / next actions

- **Minimum real dataset for T1?** A multi-day set with a frozen decoder and compute-able R² per
  window (brainsets/DANDI/Perich–Miller). Do any lab sets exist? (kalman-test has session localization
  outputs but I found no per-session *task* decoder metric — that gap is the blocker.)
- **Is `health(t)` well-defined** for a low-D cursor at all, or is R² too flat to forecast? Check before
  building anything.
- **Do we have a within-session block structure** for T2? If not, T1 is the whole feasibility test.
- **Deliverable now:** the T0 harness (synthetic) can be built to validate target/split/null wiring
  before any real data — that is the only "test" runnable without real sessions.

## 9. One-line takeaway
Prediction is the load-bearing, cheapest-to-falsify assumption of the whole program. Test it first
against a **persistence baseline**; if it clears, forecast the *mechanisms* (Step A) and import the
panel as the decomposition's identifiability instrument (Step B). If it fails, everything downstream
is moot and the offline decomposition stands alone.

