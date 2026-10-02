# Paper list (annotated, relevance-rated)

Working bibliography for the cross-session drift project. **Relevance is to *our* question** — forecasting
(and decomposing) cross-session decoder degradation from a chronic same-probe recording.

Compiled from read-only lookups (OpenAlex/arXiv/PubMed) + the two source docs. Where I am unsure of an
author list I say so — **verify DOIs/authors before citing.**

Legend: **Relevance: 5/5** = must-read / direct; **3/5** = useful support; **1/5** = background only.

---

## Category 0 — Source documents (not papers)

- **Invasive-drift decomposition proposal** (`drafts/invasive_drift_decomposition_proposal.pdf`). The
  4-condition decomposition (unit loss / drift / gain). **Relevance: 5/5.**
- **Forecast note** (`drafts/bci_drift_forecast_note-2.pdf`). Physical taxonomy + 3 clocks + Bayesian
  panel + transport map `G`. **Relevance: 5/5.**

## Category 1 — Cross-session degradation & instability *measurement*

- **"Measuring instability in chronic human intracortical neural recordings towards stable, long-term
  brain-computer interfaces"** (MINDFUL) — Pun et al., *Communications Biology* 2024,
  doi `10.1038/s42003-024-06784-4`. **Relevance: 5/5.**
  - *Who:* Pun, Silversmith, ... BrainGate collaborators.
  - *What:* a **label-free instability score** for iBCI recordings.
  - *Why us:* the closest prior art and the **bar to beat**; it is *contemporaneous*.
  - *How:* score computed from neural activity without behavior labels.
  - *Results:* correlates strongly (r ≈ 0.7–0.9) with closed-loop decoder error over weeks.
- **"Intra-day signal instabilities affect decoding performance in an intracortical neural interface
  system"** — *J. Neural Engineering* 2013, doi `10.1088/1741-2560/10/3/036004` (~260 cites).
  *(author list unverified)* **Relevance: 4/5.**
  - *What:* demonstrates that within-day signal instability degrades decoding.
  - *Why us:* empirical precursor — instabilities matter; but intra-day & contemporaneous.
- **"Intracortical recording stability in human brain–computer interface users"** — *J. Neural
  Engineering* 2018, doi `10.1088/1741-2552/aab7a0` (~167 cites). *(author list unverified)*
  **Relevance: 3/5.**
  - *What:* characterizes multi-month stability of intracortical recordings in humans. *Why us:*
    background on how unstable signals are; not predictive.

## Category 2 — The decomposition / attribution of degradation

- **Hayashi et al. 2026 (under review, CCN)** — "Representational drift dominates cross-session decoder
  degradation in mouse primary visual cortex." **Relevance: 5/5.**
  - *What:* the **four-condition decomposition** (drift 59.5% / neuron loss 31.6% / gain 8.9%) in
    Ca-imaging. *Why us:* the direct anchor we port to ephys. *How:* 4 conditions + pairwise diffs.
  - *Results:* representational drift dominates; exact (telescoping) decomposition.
- **"To sort or not to sort: the impact of spike-sorting on neural decoding performance"** — *J.
  Neural Engineering* 2014, doi `10.1088/1741-2560/11/5/056005` (~125 cites). **Relevance: 3/5.**
  - *What:* how spike sorting choices change decoder performance. *Why us:* informs the "before/after
    spike" (H3) ambiguity — sorting artifacts vs biology.

## Category 3 — Drift mitigation / adaptation / alignment (fix it, don't forecast it)

- **"Stabilization of a brain–computer interface via the alignment of low-dimensional spaces of neural
  activity"** — Degenhart et al., *Nature Biomedical Engineering* 2020, doi `10.1038/s41551-020-0542-9`.
  **Relevance: 4/5.** *What:* align low-D latent spaces across days so a decoder keeps working.
  *Why us:* canonical *fix*, not *predict*; a baseline the decomposition would grade.
- **"Stabilizing brain-computer interfaces through alignment of latent dynamics"** — Karpowicz et al.
  2022, bioRxiv. **Relevance: 4/5.** *What:* latent-dynamics alignment across sessions. *Why us:*
  the note's "alignment addresses drift".
- **"Making brain–machine interfaces robust to future neural variability"** — Sussillo, Stavisky, Kao,
  Ryu, Shenoy, *Nature Communications* 2016, doi `10.1038/ncomms13749`. **Relevance: 4/5.**
  *What:* train RNN decoders under simulated neural noise/dropout for robustness. *Why us:* "future
  variability" in the title — but robustness by training, not a forecast.
- **NoMAD** — *Nature Communications* 2025. *(author list unverified)* **Relevance: 4/5.**
  *What:* unsupervised day-to-day "stitch" so a frozen decoder keeps working. *Why us:* the note calls
  it "G without a forecast."
- **Farshchian et al.** — adversarial across-day adaptation, *eLife*. **Relevance: 4/5.** *What:*
  adversarial domain adaptation across sessions.
- **Ma et al. 2023** — adversarial training for cross-session decoding (per the decomposition proposal).
  **Relevance: 3/5.**
- **LFADS (Latent Factor Analysis via Dynamical Systems)** — Pandarinath et al., *Nature Methods* 2018,
  doi `10.1038/s41592-018-0109-9`. **Relevance: 4/5.** *What:* sequential autoencoder for single-trial
  neural dynamics; a latent-dynamics method the framework would evaluate.
- **CEBRA** — Schneider, Lee, Mathis, *Nature* 2023, doi `10.1038/s41586-023-06035-2`.
  **Relevance: 3/5.** *What:* contrastive latent embeddings for behaviour+neural data.

## Category 4 — Population & representational drift (neuroscience grounding)

- **"Long-term stability of cortical population dynamics underlying consistent behavior"** — Gallego,
  Perich, Chowdhury, Solla, Miller, *Nature Neuroscience* 2020, doi `10.1038/s41593-019-0555-4`.
  **Relevance: 5/5.** *What:* motor-cortex *population dynamics* are stable over months even as
  single-unit tuning varies. *Why us:* the "drift but stable manifold" framing; also the data source
  (Perich `000688`).
- **"A neural population mechanism for rapid learning"** — Perich & Miller, *Neuron* 2018.
  **Relevance: 4/5.** *What:* the `000688`-family recordings (reaching, M1/PMd over months).
- **"Long-term dynamics of CA1 hippocampal place codes"** — Ziv et al., *Nature Neuroscience* 2013.
  **Relevance: 3/5.** *What:* representational drift over days in place cells. *Why us:* the
  "representational drift" concept origin (though hippocampus).
- **"Dynamic reorganization of neuronal activity patterns in parietal cortex"** — Driscoll et al.,
  *Cell* 2017. **Relevance: 4/5.** *What:* strong single-neuron drift despite stable behaviour.
- **"Stable task information from dynamic neural activity"** — Rule, O'Leary, Harvey et al. 2019.
  **Relevance: 4/5.** *What:* stable *decodable* info amid unstable single-neuron responses — the
  conceptual spine of drift vs decodability.


## Category 5 — Datasets & benchmarks

- **FALCON Benchmark** — Karpowicz et al., *NeurIPS 2024 Datasets & Benchmarks* / `snel-repo.github.io/falcon/`.
  **Relevance: 5/5.** *What:* few-shot cross-session iBCI decoding benchmark (held-in / held-out days,
  5 datasets incl. human iBCI). *Why us:* the **Step-1 baseline to reproduce**; its non-stationarity
  motivation maps onto our three mechanisms.
- **Perich & Miller long-term monkey reaching** — DANDI **`000688`** (111 sessions over months).
  **Relevance: 5/5.** *What:* macaque M1/PMd spike times + cursor kinematics, chronic arrays.
  *Why us:* the **spine's testbed**.
- **IBL — "Reproducibility of in vivo electrophysiological measurements in mice"** — Banga et al.,
  *eLife* 2025, 13:RP100840, doi `10.7554/eLife.100840` (bioRxiv `10.1101/2022.05.09.491042`).
  **Relevance: 4/5.** *What:* 10 labs, 121 replicates, same target regions; raw + QC + decoding.
  *Why us:* **cross-insertion**, not cross-session — used as a **feature/QC resource** only.
- **IBL Brain-Wide Map** — IBL et al., *Nature* 2025, doi `10.1038/s41586-025-09235-0` *(verify)*.
  **Relevance: 3/5.** *What:* large Neuropixels decision-making dataset. *Why us:* panel-feature
  precedents; not motor.
- **Neuropixels 2.0** — Steinmetz et al., *Science* 2021, doi `10.1126/science.abf4588`.
  **Relevance: 3/5.** *What:* miniaturized probes enabling stable long-term recordings (the "chronic
  same-probe" hardware). 
- **Neural Latents Benchmark (NLB'21)** — Pei et al., *NeurIPS 2021*. **Relevance: 3/5.** *What:*
  single-session macaque datasets; useful for decoder recipes, not cross-day.
- **brainsets** — Azabou et al., *NeurIPS 2023*, `github.com/neuro-galaxy/brainsets`.
  **Relevance: 3/5.** *What:* standard loaders (`perich_miller_population_2018`, `churchland_shenoy_2012`).

## Category 6 — Ephys QC / spike sorting / motion

- **DREDge** — Windolf et al., *Nature Methods* 2025. **Relevance: 4/5.** *What:* compute probe motion
  from the raw AP/LFP band. *Why us:* the panel's motion feature (needs raw; absent in our macaque sets).
- **UnitMatch** — tracking the same units across days via waveform similarity. **Relevance: 4/5.**
  *What:* unit-identity matching; central to the decomposition's "matched units" (H3).
- **Bombcell / quality metrics (Kilosort lineage)** — cell-quality classification (good vs MUA, SNR,
  amplitude). **Relevance: 3/5.** *Why us:* the "death vs isolation" split for H3.
- **RIGOR metrics** (IBL `paper-reproducible-ephys`) — AP RMS, yield, `noise_cutoff`, `amp_median`.
  **Relevance: 4/5.** *Why us:* concrete QC-feature definitions to borrow (`07`).

## Category 7 — Decoders & forecasting methodology

- **Kalman filter for cursor decoding** — Wu et al. 2006 / Kim et al. 2008 (`10.1088/1741-2560/5/4/010`);
  the classic iBCI velocity decoder. **Relevance: 3/5.** *Why us:* one decoder family for the test.
- **Ridge / OLE decoders** — standard linear baselines. **Relevance: 3/5.** *Why us:* the MVP decoder.
- **Gneiting & Katzfuss 2014**, "Probabilistic forecasting" (*Annu. Rev. Stat.*). **Relevance: 3/5.**
  *Why us:* how to *evaluate* a forecast (calibration, proper scores) — needed for the gate.
- **MINDFUL code** — `github.com/ewinapun/MINDFUL`. **Relevance: 4/5.** *Why us:* reference scalar
  baseline to beat.

## Gaps found (what is NOT in this list)

- **No** paper forecasting *cross-session decoder failure* ahead of time from a feature panel.
- **No** paper *decomposing* an ephys cross-session drop into unit loss / drift / gain (the Ca-imaging
  study is the only anchor).
- **No** public chronic-same-probe **motor** Neuropixels dataset (the "unicorn").

## Caveats
- Relevance is **to our project**, not general importance.
- Author lists/DOIs marked *(verify / unverified)* were not confirmed in this pass; Semantic Scholar was
  rate-limited. Run a Google Scholar + forward-citation pass on **MINDFUL** before citing.

