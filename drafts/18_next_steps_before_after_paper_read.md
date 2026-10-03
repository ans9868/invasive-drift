# 18 — Next steps: BEFORE the paper read vs AFTER

**Purpose.** Freeze the plan as it stood immediately *before* the 2026-10-03 read of FALCON + NoMAD, next to
the revised plan, so the delta is explicit and reviewable.

**Scope.** This note is **only about what we do next**. The *content* of the two papers (numbers, quotes,
extraction) lives in `../findings/2026-10-03_prior_art_falcon_nomad.md` and is deliberately not repeated here.

---

## Part A — the plan BEFORE the read (as agreed, pre-read)

### A.0 State at the moment the read started

- ✅ grid plan, ✅ adapter library, ✅ metrics library, ✅ decoder pickle cache (round-trip verified),
  ✅ selftests green.
- ✅ cache built: **53 sessions, 52 artifacts, median 15.1 min**; fixed-duration **2-min blocks**.
- ✅ `run_grid.py` tracer: **2 rows × 61 cols in 0.2 s** on `CO-20131003`.
- ✅ `sanity_8020.py` exactly reproduced the zoo: **ridge 0.357 / wiener 0.404 / kf_posvel 0.393**.
- ✅ **Windows REMOVED** from the grid (Idea 19) → grid uses **one** split: 20/80 (decoder) + 80/20 (adapter).

### A.1 The six open items

1. **Persistence floor** — R²_persist = 0.995 on `CO-20131003` vs decoder 0.301 vs target 0.671. Wanted:
   (i) plain-language explanation confirmed, (ii) an LLM prompt, (iii) the assistant to read the papers on
   how the field handles this. *Partially delivered; paper half now done.*
2. **CIs are over-confident** — ρ₁ ≈ 0.9976 → effective N ≈ 9 per 2.4-min eval, not 10,600. Agreed to fix,
   **not yet implemented**.
3. **τ smoothing sweep** — {60, 120, 240} ms. Agreed to run, **not yet implemented**.
4. **`SS_free` + innovation R²** — agreed to add, **not yet implemented**.
5. **Reframe write-up** — user asked for it at the **top** of the writeup ("we can go back and fix it
   later"). **Not written.**
6. **FALCON** — assistant had not read it. **Now read.**

### A.2 The three agreed changes

| | Change | Notes at the time |
|---|---|---|
| a | Add **`SS_free`** (skill score vs state-only kinematic free-run) and **`r2_innovation`** (+ **`r2_recon`**) to `metrics.py` and the grid's schema | needs a code change, then the grid run |
| b | **Fix CIs** — block bootstrap over reaches, *or* subsample one bin per ~240 ms autocorrelation time | **analysis-time only — no rerun needed for aggregates** |
| c | **τ sweep** — rerun the decoder zoo at τ ∈ {60, 120, 240} ms | needs a decoder-zoo rerun |

### A.3 The planned sequence

1. Write the **reframe** at the top of `findings/<date>_adapter_grid.md` (raw R² = classical; SS + innovation
   R² = headline; persistence = predictability ceiling, never a competing decoder; CIs by block bootstrap).
2. **Full grid** — `run_grid.py` over all 53 sessions, all decoders × adapters × N.
3. **`staleness.py`** — rolling headline + expanding calibration budget.
4. **P5** — trainable adapters.
5. **P6/P7** — per-decoder cards, LR-1 curves, CIs, diagnostics.

### A.4 Known-safe, pre-read

- `r2_persist` and `r2_target` are **cell-invariant** → every adapter/decoder comparison, learning curve and
  53-session aggregate is **unchanged** by any reframing. Only absolute framing needed fixing.
- Our MSE-on-velocity objective == FALCON's own movement baselines (Wiener filter, RNN) and Glaser's LSTM.
  We were **not** methodologically off.
- The 240 ms smoothing acts on **spikes only**, not velocity → it does **not** inflate the persistence floor.
  τ sharpens the *decoder*; it does not lower the *floor*.

---

## Part B — the plan AFTER the read (recommendation)

Full paper content and citations: `../findings/2026-10-03_prior_art_falcon_nomad.md`. Only the **actions**
are restated here.

### B.0 The one reordering

**Write the reframe FIRST.** It was item ⑤ and the last step before the grid run; it is now **step 1**,
because the read gave it (i) an explicit citation backbone and (ii) a "do-not-claim" list. Writing it before
the grid run means the grid's output columns are chosen to serve the framing rather than the reverse.

### B.1 What changes, item by item

| # | Open item | Change after the read |
|---|---|---|
| 1 | Persistence floor | **Closed.** Frame as a property of the **behaviour at `bin_ms`**, reported *once* as the task's predictability ceiling — never as a rival decoder in the results table. Raw R² needs **no correction**: FALCON also normalises against the mean, so our headline R² *is* the FALCON number. |
| 2 | Over-confident CIs | **Approach replaced.** Do **not** patch the bins with a bootstrap. **Drop per-bin CIs entirely.** Adopt the field's two conventions: **mean ± std across sessions** (FALCON) and **[Q1, Q3] across session pairs + Wilcoxon signed-rank + failure counts** (NoMAD). If a within-session interval is ever genuinely required, block bootstrap with block length = the integrated autocorrelation time (~830 bins ≈ 16.6 s) — never i.i.d. |
| 3 | τ sweep | **Keep**, unchanged in substance ({60, 120, 240} ms). Better justified now: FALCON's WF history sweep (600 ms M1 / 140 ms M2 / 600 ms H1, chosen by elbow) is the same experiment, and NoMAD uses 20 ms bins throughout. **240 ms must carry the latency caveat** — NoMAD explicitly criticises ADAN/Aligned FA for bin sizes "which would incur latencies inappropriate for iBCI use". |
| 4 | `SS_free` + innovation R² | **Keep — now confirmed as our unique contribution.** Neither paper computes anything but R². Cite FALCON A.2 ("metrics alone do not necessarily capture all properties of a predicted output... we recommend visualizing predictions") as the justification. |
| 5 | Reframe | **Promoted to step 1.** Now citation-anchored. Constrained by the "do not claim" list below. |
| 6 | FALCON | **Done.** |

### B.2 Metric / schema additions (analysis-time — **no grid re-run needed**)

- `r2_vw` — **verified, not added.** `r2_all` (pooled) is *identically* sklearn's
  `r2_score(..., multioutput='variance_weighted')` — the FALCON/NoMAD metric — because
  `Σ_d w_d(1−num_d/w_d)/Σ_d w_d = 1 − Σ_d num_d/Σ_d w_d`. So our raw R² **was already** on FALCON's scale.
  `r2_vw` is now computed alongside purely as a cross-check (`|pooled − vw| = 1.4e-13`); a *uniform* mean
  would have given −0.32 where the correct value was +0.88.
- `snr = −10·log₁₀(1 − r2)` and `half_life_bins = ln2 / B`, where `B` comes from fitting `y = A·e^(−Bt)` to
  the SNR series — **NoMAD's stability currency, adopted verbatim**.
- `failure = r2 < 0` — per cell and per session, reported as a **count**. NoMAD's own convention for
  "decoding failure".
- **OR / ZS framing** — report the oracle (refit on the eval block) next to the zero-shot static number.
  The grid already has `refit_decoders` for this; it is presentation, not new computation.
- **Wilcoxon signed-rank across paired blocks** instead of t-tests on bins.

### B.3 `staleness.py` now has a published target shape

- **NoMAD's half-life** is the decay summary → `staleness.py` should emit half-life in the same units.
- **FALCON Table 3** (WF trained *from scratch* on ~1 min of held-out calibration: M1 0.24 / M2 0.14 /
  H1 0.11, vs oracle 0.53 / 0.26 / 0.21) is **exactly** the expanding-calibration-budget curve our
  `N_fracs` sweep produces, at finer granularity. Report it in FALCON's two-reference framing
  (**from-scratch vs oracle**) rather than in raw R².

### B.4 New lead to check before relying on it

NoMAD's *unloaded reaching* dataset is `Chewie_CO_2016` (Miller-lab, planar-manipulandum centre-out),
citing Perich *Neuron* 2018 / Gallego *Nat Neurosci* 2020 — i.e. the **same centre-out lineage** as our
Perich DANDI `000688` sub-C (`CO-*` session IDs). If our sessions share the animal/protocol, then NoMAD's
published **across-session** half-life (57.9 d) and our **within-session** curve sit on one data family, and
the "the across-session literature never plots the within-session axis" claim becomes checkable rather than
rhetorical. **Confirm lineage before using this framing.**

### B.5 Revised sequence

1. **Reframe write-up** (citation-anchored; constrains the schema).
2. **Analysis-time metric additions** (`r2_vw`, `snr`/`half_life`, failure counts, OR/ZS presentation,
   signed-rank). *No rerun.*
3. **τ sweep** {60, 120, 240} ms → then **full grid** (`run_grid.py`, 53 sessions × decoders × adapters × N).
4. **`staleness.py`** — rolling headline + expanding calibration budget, in FALCON Table-3 shape and
   NoMAD half-life units.
5. **P5** trainable adapters.
6. **P6/P7** per-decoder cards, LR-1 curves, CIs (across-block IQR, not per-bin), diagnostics.

### B.6 "Do not claim" (from the read)

- **Not** "within-session drift is unstudied" — NoMAD publishes within-session half-lives of **3.26 min
  (static) / 3.21 min (RTI) / 5.60 h (NoMAD) / 11.73 h (NoMAD+RTI)**.
- **Not** "nobody aligns latent spaces" — Degenhart / ADAN / NoMAD / CycleGAN do; NoMAD reaches a
  **208.7-day** half-life on isometric force.
- **Not** "our objective is wrong" — FALCON's own movement baselines are a Wiener filter and an RNN.
- **Do** claim: we plot and *forecast* the within-session decay curve that this literature only summarises
  with a post-hoc half-life, and we *decompose* it (unit loss / drift / gain).

---

## Part C — the delta, in one table

| | Before | After | Why |
|---|---|---|---|
| Ordering | reframe was the step *after* the three changes | reframe is **step 1**, before any code | now citation-anchored; it constrains the schema |
| CIs | block bootstrap / subsample the bins | **drop per-bin CIs**; across-session std, across-pair IQR + signed-rank | that *is* the field's convention; dissolves ρ₁→N_eff instead of patching it |
| R² | pooled | pooled **+ `r2_vw`** | exact FALCON/NoMAD comparability |
| Stability | (planned ad hoc) | `snr` + exponential fit + **half-life** | NoMAD's currency |
| Failures | (not planned) | **`r2 < 0` count** | NoMAD's convention |
| Oracle | `refit_decoders` existed but unpresented | **OR vs ZS framing** in every table | FALCON's readability convention |
| Tests | t-test on bins | **Wilcoxon signed-rank on paired blocks** | correct under autocorrelation |
| τ sweep | agreed | agreed **+ latency caveat at 240 ms** | NoMAD's own criticism of large bins |
| `SS_free`/innovation | agreed | agreed, **confirmed additive** | neither paper has it |
| Calibration curve | `staleness.py`, shape unspecified | **FALCON Table-3 shape** (from-scratch vs oracle) | published target |
| New | — | **check `Chewie_CO_2016` lineage** | may put within- and across-session on one data family |

## Part D — what did NOT change

- The **grid** itself: 20/80 (decoder) + 80/20 (adapter), no windows, 2-min blocks, `bin_ms = 20`, with
  `tau_ms = 240` as one point of the sweep.
- The **53-session cache** and the decoder pickle cache. **No rebuild.**
- **All rankings.** `r2_persist` and `r2_target` are cell-invariant, so every adapter/decoder comparison,
  learning curve and 53-session aggregate stands as computed. **Only the absolute framing changes.**
- **P5** and **P6/P7** in substance.
- The overall objective: MSE-on-velocity == FALCON's own movement baselines.
- The persistence floor's *value* (R²_persist ≈ 0.995 at `bin_ms = 20`) — only its **interpretation**
  changed: it is a property of the behaviour, not a competing decoder.

## Part E — unresolved / to decide

1. **Reframe location** — new framing doc, or the top of `findings/<date>_adapter_grid.md` (not yet
   written; the grid has not been run)?
2. **`SS_free` definition** — "state-only kinematic free-run" needs its exact reference trajectory pinned
   down before it can be coded (which free-run: decoder's own recurred state, or a kinematic prior?).
3. **τ sweep placement** — a separate decoder-zoo rerun, or folded into the full grid as a config axis?
4. **`Chewie_CO_2016` lineage** — confirm before using the same-data-family framing.
