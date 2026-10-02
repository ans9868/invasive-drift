# IBL Prong 2 — the quality/motion feature payload (H3 identifiability)

**Status:** thinking / scoping notes. **No downloads.** Facts from read-only lookups of the
`int-brain-lab/paper-reproducible-ephys` repo (master branch) + DANDI.

Does option 2 of `06` Part 5: read the IBL Reproducible Ephys `fig_data_quality` / `fig_ephysfeatures`
methodology to enumerate exactly which quality/motion features exist — i.e. the payload that can make
the decomposition's "unit loss" identifiable (H3).

---

## 1. What the IBL Reproducible Ephys pipeline actually computes

Source files read:
- `RIGOR_script.py` (the reproducibility/QC metrics, runnable on raw)
- `reproducible_ephys_functions.py` (grid query + LFP features + streaming)
- `fig_data_quality/fig_supplemental_neuron_metrics.py` (neuron QC metrics, Fig 1 Supp 4)
- `fig_ephysfeatures/*` (Figure 3 = ephys features; mostly plotting helpers)

### 1a. Quality / QC metrics
- **AP RMS per channel** — `compute_ap_metrics` writes `_iblqc_ephysChannels.apRMS.npy`
  (raw-AP RMS, µV; sanity limit `MAX_AP_RMS = 40`). -> channel SNR / dead-channel signal.
- **LFP metrics** — `compute_lfp_metrics` (limit `MAX_LFP_DERIVATIVE = 1`).
- **Neuron yield** — `MIN_NEURONS_PER_CHANNEL = 0.1`; `compute_yields` -> `passing_units`,
  `num_sites`, `passing_per_site`, `all_units`. -> units alive per site.
- **Cluster QC (Kilosort2 metrics)** — `spike_sorting_metrics_ks2` -> `clusters.metrics.pqt`, with
  columns used downstream: **`label`** (1.0 = passing QC / good vs MUA), **`amp_median`** (amplitude,
  threshold >50 µV), **`noise_cutoff`** (threshold <5), plus the usual contamination/isolation fields.
  -> **the death-vs-isolation / quality split.**
- **Spike depth + amplitude** — `spikes.depths.npy`, `spikes.amps` (depth-vs-amplitude plots).
  -> a per-unit position signal.

### 1b. Feature payload (raw + waveforms + LFP)
- **Waveforms are saved** — the streaming output architecture per probe insertion is
  `T00500/{ap.npy, ap.yml, lf.npy, lf.yml, spikes.pqt, waveforms.npy}`. -> waveform/spike-width
  features are available (unlike FALCON/Perich, which are spike-times-only).
- **LFP band features** — `lfp_power` (20–80 Hz) and `lfp_theta` (6–12 Hz), from `rms_lf_band`,
  `rms_theta_band`, `rms_lf`; `lfp_destripe` / `voltage.destripe` (AP) produce the cleaned raw.
  -> LFP state / band power.
- **Destriping** of raw AP — `voltage.destripe(...)`. -> artifact/reference handling.

### 1c. Scope / grouping
- Project filter: `ibl_neuropixel_brainwide_01`; QC `session__qc < 50`; `n_trials >= 400`.
- Regions: `PPC, CA1, DG, LP, PO` (-> `VISa/am, CA1, DG, LP, PO`).
- Grouping unit = **probe insertion (pid)**; the Reproducible Ephys project is a
  **repeated-measurement / reproducibility** design (same animals/sites re-recorded) -> the
  same-probe-multi-day axis.
- Caveat from the repo README: **bilateral (Fig 3 Supp 3) data is not released.**

---

## 2. Mapping: IBL feature -> forecast-note panel -> decomposition role

| IBL feature (source) | forecast-note panel feature | What it buys |
|---|---|---|
| **AP RMS per channel** (`apRMS.npy`) | channel SNR / fraction-dead | **splits channel death from isolation** (dead = RMS collapse) |
| **neuron yield** (passing/site) | fraction dead | magnitude of "unit loss" |
| **`label` good/MUA** (ks2 metrics) | — | death vs isolation vs MUA |
| **`amp_median`** | waveform/SNR proxy | isolation quality (amplitude) |
| **`noise_cutoff`** | — | isolation / contamination |
| **`waveforms.npy`** | waveform SNR | spike width / waveform stability |
| **LFP power 20–80 Hz + theta 6–12 Hz** | LFP state (band power) | **gain / state** |
| **spike depth + amplitude** | (DREDge-ish) position | drift / motion proxy |
| **destripe / artifact handling** | reference/artifact | alert hygiene |

## 3. Why this IS the H3 identifiability instrument

Decomposition hole **H3**: the four conditions can only see "in the matched set / not", so "unit loss"
conflates (i) physical death, (ii) isolation/sorting failure, (iii) matcher false-negative. The IBL
payload separates them:
- **(i) death** -> AP RMS collapse + zero yield at a site;
- **(ii) isolation failure** -> `noise_cutoff` / `amp_median` / waveform drift while the channel is
  alive (RMS intact);
- **(iii) matcher false-negative** -> unit present and isolated but absent from the matched set
  (measurable only with a hand-labeled subset).

It also carries the **gain** (LFP state) and **motion/drift** (depth/amp) arms the reduced
spikes-only panel lacks (doc `06` Part 3). So the note's named "Kilosort / UnitMatch / Bombcell
death-vs-isolation" idea already exists here as `label` / `amp_median` / `noise_cutoff` / yield.

## 4. Caveats (honest)

- **Task mismatch:** IBL = decision-making (wheel/contrast), not motor; decode target is
  choice/stimulus, not kinematics.
- **One insertion per mouse** in Brain Wide Map (mostly); the *Reproducible Ephys* arm is the
  repeated-measurement subset -> same-probe-multi-day lives there, not in the full BWM.
- **Signals are computed by the IBL stack** (`ibldsp`, `brainbox`, `spikeglx`) which assumes IBL file
  layout -> a port to macaque data (Prong 1) means re-implementing RMS/quality, not copying outputs.
- **Bilateral data not released** (Fig 3 Supp 3).

## 5. Consequence + next

- The **H3 fix and the full panel both live in IBL raw** -> Prong 2 is not optional if we want the
  typed panel; it is optional only if we accept the reduced (rate + match-survival) predictor.
- ~~Cheapest next step to de-risk Prong 2: confirm the repeated-insertion design.~~ **DONE — see §6.**

---

## 6. Resolution (the de-risking question): cross-session or cross-insertion? — ANSWERED

**Source found:** Banga et al., *"Reproducibility of in vivo electrophysiological measurements in
mice"*, **eLife 2025, 13:RP100840** (doi `10.7554/eLife.100840`; PMID 40354112; PMC12068871; bioRxiv
preprint `10.1101/2022.05.09.491042`, 2022-05-09). Repo = `int-brain-lab/paper-reproducible-ephys`.

**Design (from the abstract, verbatim content):**
- multi-lab collaboration, **10 laboratories**;
- "**repeatedly targeted Neuropixels probes to the same location**" (secondary visual areas,
  hippocampus, thalamus) in mice doing the shared decision-making task;
- "**a total of 121 experimental replicates**, a unique dataset for evaluating reproducibility";
- reproducibility hurt by **electrode targeting variability** and **limited statistical power**
  (single-neuron tests of modulation); improved by **histological + electrophysiological QC criteria**.
- Repo manifests: `freeze_2024_03.csv` = **104 pids**, `release_2023_12.csv` = **106 pids**; columns =
  `[pid, level]` only (no subject/probe names) — consistent with "~121 replicates across ~100 insertions".

**Verdict: it is a CROSS-INSERTION (cross-lab / cross-animal) reproducibility design — NOT a
same-probe-across-days chronic design.** "Repeatedly targeted to the same location" means *many
independent insertions into the same target region*, not one animal/probe recorded over many days.

**Consequence for the plan:**
- IBL = (a) the **feature/QC vocabulary** (AP RMS, yield, cluster `label`/`amp_median`/`noise_cutoff`,
  waveforms, LFP power/theta, depth) and (b) a **statistical-power / reproducibility benchmark** —
  i.e. the H3 tooling and the panel-feature definitions.
- IBL = **NOT** the **cross-session drift axis**. "Forecast session N+1 from earlier sessions" needs a
  **longitudinal** set (Perich `000688` / FALCON / NeuroTask).
- So the original two-prong framing sharpens: **Prong 1 = longitudinal motor (cross-session axis);
  Prong 2 = IBL (feature/QC definitions + reproducibility benchmark).** They are complementary, not
  interchangeable — and neither alone has "raw + same-probe-multi-day + motor".

**Remaining decision:** does Prong 2 mean "borrow IBL's feature definitions and compute them on
longitudinal macaque data (where raw permits)" or "run the typed panel on IBL's cross-insertion,
non-motor task and accept the mismatch"? The metric of interest differs: cross-session drift vs
cross-insertion reproducibility.

