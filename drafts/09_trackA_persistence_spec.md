# Track A, Step 3 — persistence-test spec (Perich `000688`)

**Status:** test spec / thinking. No downloads. Realizes `04_prediction_feasibility_test.md` on the
dataset chosen in `06` (Perich `000688`), inside the spine defined in `08`.

---

## 1. The pre-registered question

> **Do features measured strictly before session *N* predict session *N*+1's decoder health better than
> a persistence baseline?**

"Better than persistence" is the crux (`04` §1): without it, "prediction" is just detection.

## 2. Data and ordering

- **Perich `000688`**: NWB, spike times + cursor pos/vel + trials; sessions dated
  (`sub-<X>_ses-<TASK>-<YYYYMMDD>_behavior+ecephys.nwb`), task = center-out (`CO`) or random-target (`RT`).
- **Chains must be same-subject AND same-probe (chronic, not re-implanted).** Order sessions by date
  within subject; keep per-subject chains; **do not mix subjects**, and only chain sessions that share
  the **same physical chronically-implanted array** (Perich arrays stay in; units come/go, the device
  does not). This is the "same probe, not removed" requirement (`08`).
- **No raw, no waveforms** (`06` §1c) → features are spike-derived only.

## 3. Step 1 — reproduce a FALCON-style decoder (sanity)

- Decoder: **ridge regression**, binned spike counts (bins e.g. 50 ms) → cursor **velocity** `[vx, vy]`
  (or position). Fit `alpha` by cross-validation inside the *train* session only.
- Intra-session R² (train held-in trials, test held-out trials of the same session) = the ceiling `C1`.
- Cross-session R² (train session *i*, test session *j*) = the degradation `C2`.
- **Success criterion for Step 1:** reproduce a *nonzero, increasing-with-gap* cross-session drop —
  i.e. the phenomenon exists in the data we can actually load. If it doesn't, stop and re-pick.

## 4. Step 2 — build the target series

- **Frozen reference decoder:** train once on session 1 (or the "last-good" session) — never updated.
- **Health series:** `health(t)` = R² of the frozen decoder on session *t*.
- **Targets:**
  - regression: `ΔR²(t+1) = health(t+1) − health(t)`;
  - classification: `crash(t+1) = 1[health(t+1) < τ]` with a pre-registered `τ` (e.g. R² < 0.1).

## 5. Step 3 — the reduced-panel predictor

Features at session *t* (strictly past), per the spikes-only ceiling (`06` Part 3):

| Feature | Definition |
|---|---|
| firing rate | population mean/median spikes-per-bin |
| unit count / fraction-quiet | # units; # below a rate floor vs the previous session |
| match-survival | fraction of session *t* units matched to session *t−1* (spike-train/correlation matching) |
| factor-angle | angle between session-*t* and reference factor (PCA/FA) subspaces |
| noise correlations | mean off-diagonal pairwise spike-count correlation (rest/non-task windows) |
| decode entropy | entropy of the decoder's outputs at session *t* (past-only) |

Each as **level + difference vs the previous session** (the note's "clocks," collapsed to the
session clock).

## 6. Baselines (report all — the test lives or dies here)

1. chance (AUROC 0.50 / majority);
2. **persistence** — `health(t)` predicting `ΔR²(t+1)` / `crash(t+1)`;
3. MINDFUL-style scalar (one feature, nested);
4. reduced panel (above).

## 7. Splits + leakage discipline (`04` §4)

- session-out / subject-out; never random-split across the session boundary;
- frozen decoder; no feature derived from *t+1*; past-only normalization;
- report persistence **on the same split**.

## 8. Metrics + the pre-registered gate

- **Discrimination:** AUROC / PR-AUC (class) or correlation (regression).
- **Calibration:** reliability + a proper score (PSIS-LOO-style / Brier).
- **Gate (decide before looking):**
  - **GREEN** — panel beats persistence on discrimination **and** calibration not worse, increment
    outside the CI → forecastable; proceed to mechanism breakdown (`03`/`06`).
  - **YELLOW** — beats chance but not persistence → it's monitoring/detection, not forecasting →
    reframe.
  - **RED** — doesn't beat chance → phenomenon not forecastable with these features → spine in trouble.

## 9. What is needed to actually run it (later; nothing downloaded now)

- `000688` NWB sessions (DANDI), sorted into per-subject date chains.
- A spike binning + ridge/KF decoder (the FALCON recipe).
- A spike-train unit matcher across consecutive sessions.
- The reduced-panel feature extractor.
- Session-blocked CV harness + the persistence baseline.

## 10. Choices to fix before running — PROPOSED DEFAULTS (confirm or reject)

1. **Decoder:** **ridge** (velocity), FALCON-style. *(Alternative: Kalman — later.)*
2. **Horizon:** **`t+1` (next session)** primary. *(Multi-step `t+1..t+3` secondary.)*
3. **Crash threshold `τ`:** pre-register from a *blind* look at the marginal `health(t)` distribution
   only (e.g. `τ = 0.5 ×` median intra-session R²), never tuned per case.
4. **Unit matching:** spike-train correlation only (no waveforms). Report **match-survival as a noisy
   feature** and run a **no-matching ablation** (population-level features only) to check robustness.
5. **Task mix:** restrict to **center-out (`CO`)** chains first; add `RT` as a covariate later.
6. **Chain rule (from `08`):** chains are **same-subject + same-probe (chronic, not removed)**, ordered
   by date.

## 11. One-line takeaway
The spine's whole feasibility question reduces to: **does the reduced (spikes-only) panel beat
persistence on the next-session R² series of `000688`?** If yes, the panel matters and the program
proceeds; if no, the spine needs raw features (the companion's territory) to survive.

