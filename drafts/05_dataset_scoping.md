# Dataset scoping (HPC, near-infinite storage/compute)

**Status:** thinking / scoping notes. **No downloads.** All facts below come from read-only
online lookups (DANDI REST API, brainsets README, NLB, IBL docs) plus local repo inspection.

Follows `04_prediction_feasibility_test.md` (prediction-first) and the three-step plan:
**Step 0 find datasets -> Step 1 recreate decoder/setup/baseline from a paper -> Step 2 build the predictor.**

---

## 0. The lens: storage/compute are NOT the constraint

With HPC + near-infinite storage, file size is no longer a screening criterion. You can pull raw,
re-sort units, run DREDge, run LFADS/CEBRA, per-trial decoders, and permutation nulls across hundreds
of sessions. So the binding filters change to **information content**, not bytes.

## 1. The three binding filters

1. **Same subject/probe across days** — there must be real drift to forecast.
2. **A decoder task** — motor/behavior, so per-session R² is computable.
3. **Raw/LFP availability** — the forecast note's panel (DREDge motion, SNR, LFP state) needs raw
   broadband, not only spike-sorted data.

## 2. Re-scoped candidate list (by role in the pipeline)

| Dataset | Where | Species / area | Task | Multi-session | Raw/LFP? | Role |
|---|---|---|---|---|---|---|
| **Perich & Miller** "Long-term recordings ... reaching in monkeys" | DANDI **000688** (111 assets, ~13 GB) | macaque M1/PMd | reaching | **111 sessions, ~months** (paths `sub-C/sub-C_ses-<date>_behavior+ecephys.nwb`) | spikes (NWB) | decoder + drift backbone; = Gallego-2020 anchor |
| **FALCON Benchmark M1-A/M1-B/M2** | DANDI **000941 / 001209 / 000953** | macaque M1 | reach-to-grasp / finger | yes, **pre-defined held-in / held-out (minival)** | spikes | **Step-1 baseline reproduction** |
| **FALCON H1 / H2** | DANDI **000954 / 000950** | **human iBCI** (Utah) | 7-DoF reach-grasp / handwriting | yes | spikes | closest to the "cannot lean" motivation |
| **NeuroTask** (Random Target, Maze, Center-Out, Key Grasp, ...) | DANDI **001055-001060, 001078** | macaque | maze / center-out / grasp / wrist | **multi-task, -session, -subject** | spikes | cross-subject generalization |
| Churchland "Neural population dynamics during reaching" | DANDI **000070**; brainsets `churchland_shenoy_neural_2012` | macaque M1 | reaching | yes | spikes | secondary |
| brainsets `odoherty_sabes_nonhuman_2017`, `perich_miller_population_2018` | brainsets (neuro-galaxy) | macaque | reaching | multi-session | processed | standardized loaders over the above |
| **IBL Brain Wide Map + "Reproducible Ephys"** | IBL bulk store (HTTP/FTP/Globus) + Alyx | mouse, whole-brain | decision-making | many sessions; **"reproducible ephys" arm repeats** | **raw AP + LFP + spike sorting** | **the panel-feature / DREDge arm** |
| Single-session sets (MC_Maze 000128, MC_RTT 000129) | DANDI | macaque | reaching | **NO** | spikes | **excluded** — no drift to forecast |

"Reproducible Ephys" (IBL 2024 dataset) is the closest public thing to *same mouse/probe repeated*
with raw — the best lead for a same-probe-multi-day story.

---

## 3. The constraint HPC does NOT solve: raw availability in multi-session motor sets

- The macaque multi-session motor sets (Perich, FALCON, NeuroTask) are **spikes-only** in their public
  form. 13 GB / 111 sessions = sorted spikes, not broadband. So the *full* panel (DREDge motion, LFP,
  SNR from raw) is **not** directly computable from them.
- The dataset that **does** ship raw + LFP is **IBL** — but its task is **decision-making** (not motor)
  and it is mostly **one insertion per mouse** (weak multi-day), except the "reproducible ephys" arm.
- **No single public set** has all three: *multi-month motor decoder* **and** *raw/LFP* **and**
  *same-probe-multi-day*. The ideal (chronically implanted Neuropixels, same probe across days, motor
  task) did **not** surface on DANDI (Neuropixels 2.0 chronic is not obviously public there).

**Consequence:** Step 1 + the decomposition/drift backbone are feasible on the macaque sets with a
**reduced panel** (rate, unit count, unit-match survival — all spike-derived). The *full* panel needs
IBL raw, at the cost of task/longitudinal mismatch.

## 4. Strategy that fits HPC + big storage

Two-pronged (cheap when compute isn't a limit):

- **Prong 1 — decoder + drift + decomposition.** FALCON (defined cross-session baseline) for Step 1,
  then Perich 000688 (111 sessions) for the long horizon. Panel = reduced (spike-derived) features.
- **Prong 2 — panel-feature engineering.** IBL raw -> build/validate DREDge + SNR + LFP features and the
  death-vs-isolation split, then port the feature schema to Prong 1.
- **Bridge:** a shared feature schema so Prong 2's features can be computed on Prong 1's data where raw
  permits (or approximated where it doesn't).

This yields a working Step-1 baseline, a real multi-month drift axis, and a validated panel — without
pretending one dataset does everything.

## 5. Open decisions (to settle before choosing)

1. **Prong-1 subject:** macaque (111 sessions, cleanest drift) vs **human iBCI** (FALCON H1/H2 — best
   motivation, fewer sessions)?
2. **Panel scope:** accept a **reduced (spike-derived)** panel, or put **IBL-raw** on the critical path
   now for the full panel?
3. **Prediction granularity:** session-to-session (macaque sets are natural here) vs within-session
   ramp (IBL-style)?

## 6. Leads worth chasing next (lookups, not downloads)

- Confirm **IBL "Reproducible Ephys"** session structure (same mouse/probe across days?) and whether it
  has an analyzable task metric.
- Check whether **Perich 000688** or **FALCON** releases include any raw/ or quality-metric fields
  beyond spike times (a quick metadata read of one NWB).
- Look for any public **chronic Neuropixels, same-probe-multi-day, motor** set (the missing unicorn):
  e.g., Neuropixels 2.0 chronic lineage, or macaque chronic Neuropixels releases.
- Confirm FALCON's **human H1/H2** session counts and whether the held-in/held-out split is usable as
  the Step-1 baseline.

## 7. Workflow note (no downloads now)

On HPC the usual move is to **stream** from DANDI's S3 / IBL Globus without landing the whole dataset
locally. Recorded here only as a later workflow note; per instruction, nothing is being downloaded now.

