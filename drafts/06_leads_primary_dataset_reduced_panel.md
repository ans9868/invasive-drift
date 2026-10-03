# Lead-chase, primary-dataset decision, and the reduced panel

**Status:** thinking / scoping notes. **No downloads.** All facts from read-only lookups (DANDI REST
API, GitHub, IBL docs) + local repo inspection.

Does the three "next" items from `05_dataset_scoping.md`:
1. chase the §6 leads,
2. settle decision #1 (macaque vs human) -> one primary dataset + its Step-1 baseline paper,
3. map the "reduced panel" (what the forecast note's features become when data is spikes-only).

---

## Part 1 — Chasing the §6 leads

### 1a. Does a FALCON NWB carry raw or quality fields? (checked, no download)
DANDI asset metadata for `000941` (`sub-MonkeyL-held-in-calib_ses-20120924_behavior+ecephys.nwb`):
- `variableMeasured`: **Units, ElectrodeGroup, BehavioralTimeSeries**
- `measurementTechnique`: behavioral, **spike sorting**, surgical
- subject: `MonkeyL`, *Macaca mulatta*, age P7Y, session 2012-09-24
- **No `ElectricalSeries`, no LFP.** => FALCON is **spike-sorted units + behavior only**.

The `000941` dandiset **description** (fetched) confirms the *exact* payload: "multiunit spiking times
and behavioral data"; acquisition fields = `preprocessed_emg` (16 muscles, iEMG), `eval_mask`,
**`units` = spike times**, and `trials` metadata (`gocue_time`, `move_onset_time`, `contact_time`,
`reward_time`, `result`, `number`, `tgt_loc`, `tgt_obj`, `obj_id`, `condition_id`).
=> **`units` = spike times only; no waveforms, no quality metrics, no raw.** Task = center-out
reach-to-grasp (4 objects x 8 positions), M1 electrode arrays. License CC-BY-4.0.
Splits: `held-in-calib` (many trials, for calibration), `held-in-minival` (small format-check subset),
`held-out-calib` (separate sessions, few trials, for few-shot recalibration). One monkey per dandiset
(second monkey = `001209`).

### 1b. Is IBL "Reproducible Ephys" really same-probe-across-days? (checked)
It is the **2025 paper "Reproducibility of In Vivo Electrophysiological Measurements in Mice"**
(repo `int-brain-lab/paper-reproducible-ephys`, MIT, 888 commits). Findings:
- Figures include **`fig_data_quality`, `fig_ephysfeatures`, `fig_mtnn`, and `fig_decoding`** ->
  quality metrics **and** a decoding analysis exist.
- Insertions enumerated via `ONE` (`get_insertions(level=0, freeze='freeze_2024_03')`); public ONE
  password is `international`. It is a **repeated-measurement / reproducibility** dataset.
- Task = **IBL decision-making** (wheel + visual contrast), **not motor**.
- Bulk store (raw AP + LFP + spike sorting) is on the IBL server (HTTP/FTP/Globus) via Alyx metadata.

**Verdict:** same-probe-multi-day **with raw + quality + decoding** genuinely exists in IBL
Reproducible Ephys — but the task is decision-making. The **chronic-Neuropixels *motor* unicorn**
(one probe, many days, decode kinematics) still has not surfaced on DANDI/IBL.

### 1c. What do the Perich/`000688` NWBs actually contain? (checked via the brainsets pipeline)
Searched the `brainsets_pipelines/perich_miller_population_2018/pipeline.py` source. It reads the NWB
via `extract_spikes_from_nwbfile` (spike times) plus cursor kinematics + trial tables. Its own
description: *"spiking activity — manually spike sorted in three subjects, and threshold crossings in
the fourth — obtained from up to 192 electrodes per session, cursor position and velocity, and other
task related metadata."*
=> **spikes + cursor (pos/vel) + trial metadata; no quality metrics, no raw.**
**CORRECTION (2026-10-03):** the sub-C NWBs **do** contain `units/waveforms` (48 samples/spike) — the
pipeline simply didn't read them. See `findings/2026-10-03_waveform_probe.md`. So **waveform-based unit
matching IS available on Perich**; only *continuous raw voltage/LFP* is absent.
Tasks: center-out (`CO`) and random-target (`RT`); session ids like `c_20161021_center_out_reaching`;
3–4 rhesus macaques (M1/PMd).

**Units-field verdict (the check you asked for):** **neither FALCON nor Perich `Units` carries
waveforms or quality metrics** — both are spike-times-only (plus behavior). Consequences:
- a **waveform-SNR proxy is NOT available**;
- **unit-match survival** must be computed from **spike-train / cross-session-correlation matching**,
  not waveforms;
- i.e. the reduced panel is *leaner* than the "PARTIAL" row in Part 3 below.

---

## Part 2 — Decision #1: macaque vs human -> primary dataset + baseline paper

### The call: **macaque primary.**

| Criterion | Macaque (Perich / FALCON) | Human iBCI (FALCON H1/H2) |
|-----------|---------------------------|---------------------------|
| Session count | **111** (Perich 000688) | tens, but only a few usable held-out |
| Span | **months** (real drift) | weeks–months, sparse |
| Raw/LFP | no (spikes) | no (spikes) |
| Motivation ("cannot lean") | weaker | **stronger** |
| Baseline defined | yes (FALCON) | yes (FALCON) |

The whole point of *prediction-first* is **feasibility**; feasibility needs many sessions over a long
span. That is squarely macaque. Human iBCI is the better *motivation* but the weaker *testbed* — keep it
as a secondary generalization check, not the primary.

### The single primary dataset + Step-1 baseline

- **Primary dataset: Perich & Miller, DANDI `000688`** — 111 sessions, macaque M1/PMd reaching over
  months. This is the longest public multi-session motor set, i.e. the best drift/forecast backbone.
- **Step-1 baseline to reproduce: FALCON (Karpowicz et al.; `snel-repo.github.io/falcon/`; Pandarinath
  lab, GaTech; EvalAI leaderboard).** FALCON = *"Few-shot Algorithms for Consistent Neural decoding"*:
  it is *explicitly* about cross-day decoder failure and its stated motivation is **non-stationarity
  caused by "changes in the recording environment, changes in the user's behavior, and changes in the
  neural signals"** — which maps almost one-to-one onto the decomposition's unit-loss / drift / gain.
  It gives a **pre-defined held-in / held-out split + few-shot protocol + leaderboard**, so reproducing
  its decoder/baseline is a well-posed "Step 1". Note: the metric there is decode error on new days,
  not the drift decomposition.
- **Drift framing paper: Gallego et al. 2020 (Nat Neuro)** — already cited in the decomposition
  proposal; provides the "population dynamics stable, single-unit tuning drifts" baseline for `000688`.

**Plan shape:** Step 1 = reproduce FALCON's cross-session decoder/baseline (sanity + a turnkey
decoder). Then run that decoder across `000688`'s 111 sessions to get the per-session R² series — the
target the predictor (`04_...`) will forecast.

## Part 3 — The reduced panel: what survives spikes-only data

The forecast note's panel, evaluated against what is computable from **spike-sorted units + behavior**
no raw, no impedance, no LFP):

| Panel feature (forecast note) | Spikes-only? | How / note |
|---|---|---|
| firing rate | **YES** | per-unit spike counts in task windows |
| fraction dead | **YES (approx)** | count units/channels going quiet across sessions |
| unit-match survival | **YES (spike-train OR waveform)** | CORRECTED: Perich sub-C has `units/waveforms` → match by waveform (median corr ≈0.996). FALCON unchecked. |
| rest-only noise correlations | **YES** | spike-count correlation matrix (rest epochs) |
| factor-angle vs baseline rest | **YES (approx)** | factor analysis / PCA on rates; angle vs a reference session |
| decode entropy | **YES** | from the decoder's posterior/softmax at inference |
| waveform SNR (proxy) | **YES (Perich)** | CORRECTED: Perich sub-C has `units/waveforms` (48 s/spike) → amplitude/kurtosis/stability proxies. FALCON unchecked. |
| channel SNR (true) | **NO** | needs raw voltage |
| impedance | **NO** | hardware metadata; not in these releases |
| **DREDge position/velocity** | **NO** | DREDge needs raw AP/LFP; a unit-position *proxy* only if sorted positions exist |
| LFP state (band power) | **NO** | no LFP in FALCON/Perich NWB |

**Ceiling of Prong 1 (macaque spikes-only):** rate, fraction-dead, match-survival (spike-train only),
noise correlations, factor-angle, decode entropy. **Missing:** true SNR, **waveform-SNR proxy**,
impedance, **DREDge motion**, LFP state. So the *quality*, *motion*, and *state* arms of the forecast
note's panel all require raw — i.e. Prong 2 (IBL).

## Part 4 — Consequence + immediate next

- The **forecast** can still be tested on Prong 1 with a rate/quality/match-driven predictor — that is
  exactly the `04` feasibility test (does the reduced panel beat persistence?).
- The **full** typed panel (motion + state) needs raw, i.e. IBL Prong 2 — accept the task mismatch, or
  treat motion/state as future work.
- **Units-field check: DONE.** Both FALCON and Perich `Units` = spike times only (no waveforms, no
  quality metrics). So the reduced panel is the lean version: rate, fraction-dead, spike-train
  match-survival, noise correlations, factor-angle, decode entropy. Quality/motion/state need raw.
- **Two consequences for the plan:**
  1. The `04` feasibility test can run on Prong 1 with a **rate + match-survival** predictor only — a
     genuinely minimal first test. If that already beats persistence, the panel matters; if not, the
     spike-derived features are too thin and raw (Prong 2) is mandatory.
  2. Because Perich/`000688` has **no waveforms**, unit-matching across sessions is weaker -> the
     decomposition's H3 ("units unavailable") is even harder to split biologically on this data; that
     pushes the identifiability fix (panel covariates / death-vs-isolation) toward **IBL raw**.

## Part 5 — Suggested immediate next (pick)
- **Make the `04` persistence test concrete on Prong 1:** define the per-session R² series from a
  FALCON-style ridge/KF decoder over `000688` sessions, define the reduced predictor (rate +
  match-survival), and pre-register the GREEN/YELLOW/RED gate. (No download yet — this is a spec.)
- **Or** open the IBL Prong 2 line: read the Reproducible Ephys `fig_data_quality` / `fig_ephysfeatures`
  methodology to see exactly which quality/motion features exist there (the H3 identifiability payload).

