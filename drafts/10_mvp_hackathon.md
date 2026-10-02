# MVP (hackathon) — "predict next-session decoder failure"

**Status:** plan / thinking. Realizes the smallest end-to-end slice of Track A (`09`).

## The MVP sentence (deliverable = this, nothing more)

> **"Given a chronic same-probe recording, a handful of cheap spike-derived features from one session
> predict the *next* session's decoder drop better than assuming 'it stays the same' (persistence)."**

If we can say that with one figure + one number, the MVP is done.

## Scope: IN vs OUT

**IN (must have):**
- one dataset, one subject, one **same-probe chain**;
- one decoder; the **health(t)** curve (does degradation exist?);
- 2–4 **past-only** features (rate, unit count, match-survival);
- **persistence baseline**; one metric (AUROC/PR-AUC or correlation);
- **one figure**.

**OUT (explicitly not in the MVP):**
- decomposition (C1/C3/C2b/C2a) — that's Phase 2 / stretch;
- LFADS / CEBRA / any deep model;
- raw LFP / DREDge / impedance (we don't have them — `06`);
- multiple subjects, multiple datasets, calibration curves, fancy stats;
- the whole forecast-note panel (3 clocks, Bayesian GLM).

## Data (one grab)

- **Primary:** a slice of **Perich `000688`** — one subject, ~20–40 dated sessions (same chronically
  implanted arrays). ~120 MB/session → a few GB, fine.
- **Optional sanity:** **FALCON M1-A** (`000941`) is ~300 MB total with dated sessions + a defined
  split — good for "does our decoder recipe reproduce a cross-session drop?"
- Load spikes + cursor kinematics via `pynwb`; bin spikes (50 ms) in task windows.

## The 5 steps (with a ~24h timebox)

| # | Step | Time | Output |
|---|---|---|---|
| 0 | Grab one subject's sessions, order by date; sanity-check trial counts | 2 h | a session list |
| 1 | Ridge decoder (spike bins → cursor velocity); train/test intra-session | 3 h | baseline R² works |
| 2 | **Frozen day-1 decoder** → evaluate on every later session | 2 h | **health(t) curve** |
| 3 | Features at session *t* (rate, unit count, spike-train match-survival); predict `ΔR²(t+1)`/crash | 6 h | predictor |
| 4 | **Persistence baseline** on the same split + one figure | 4 h | the headline |
| 5 | (stretch) decompose the drop (C1/C3/C2b/C2a) as a stacked bar | 8 h | Phase 2 |

**Checkpoint after step 2:** if `health(t)` shows *no* cross-session drop, **stop** — wrong
data/decoder, and no forecast is possible.

## The one figure (this IS the demo)

- **Panel A — the phenomenon:** `health(t)` (frozen decoder R²) vs session index/date. A downward
  curve = "decoders decay across sessions," reproduced.
- **Panel B — the clever bit:** predicted `ΔR²(t+1)` vs actual (scatter, fit line) **or** AUROC for
  `crash(t+1)`, with **persistence** drawn as the reference line and the reduced-panel predictor as the
  bar/curve.
- Optional **Panel C — mechanism (stretch):** stacked bar of the drop = units-unavailable / drift / gain.

## The number (say it in one breath)

> "Reduced spike-derived panel: AUROC **0.79** vs persistence **0.74** on next-session crash
> (n = N sessions, one subject)." *(or: "R = 0.6 for ΔR².")*

Even a **null** ("panel ≈ persistence") is a real hackathon result — it says forecasting needs the raw
features (turning the MVP into a *scope* result).

## Kill criteria (be honest fast)

1. No cross-session drop in `health(t)` → wrong data/decoder → stop.
2. Predictor ≤ persistence → **the honest finding is "not forecastable from spike-derived features"** →
   ship that, don't fake it.
3. Feature extraction eats the clock → cut to **one** feature (unit count) and still test persistence.

## Tech stack (boring on purpose)

- `pynwb` (read) + `numpy`/`scipy` (bin) + `scikit-learn` (`Ridge`, `LogisticRegression`, `roc_auc_score`).
- One notebook, seeded, cached intermediates. **No** deep learning, no GPU needed.

## Stretch goals (in priority order)

1. **Decompose** the drop into 3 mechanisms (C1/C3/C2b/C2a) — the "then break down" half.
2. A **second subject** → does the predictor generalize?
3. **Time-gap axis:** plot forecast skill vs gap (next-day vs 1-week vs 1-month).

## The demo slide (5 bullets)

1. Problem: iBCI decoders quietly decay across days.
2. Question: can we *see it coming* from cheap features?
3. Method: frozen decoder → health(t); reduced panel → predict next session.
4. Result: **[the number]**, vs persistence.
5. Next: decompose *what* decays (units vs drift vs gain).

## Why this is the right MVP

It tests the program's **load-bearing assumption** (forecastability) on the **cheapest data and
features**, with a built-in falsifier (**persistence**). Everything expensive (raw features,
decomposition, deep models, multi-subject) is deliberately deferred — so a negative result is still a
clean, publishable-looking scope finding rather than a failure.

