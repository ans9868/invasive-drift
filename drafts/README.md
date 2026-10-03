# drafts/ — index

Thinking notes (no code) on two related documents about iBCI / invasive-electrophysiology
nonstationarity.

## Sources (local copies)
- `invasive_drift_decomposition_proposal.pdf`
  orig: `/Volumes/CrucialX6/Home/Zotero/storage/2X4EU83K/invasive_drift_decomposition_proposal.pdf`
- `bci_drift_forecast_note-2.pdf`
  orig: `/Volumes/CrucialX6/Home/Zotero PDFs/storage/bci_drift_forecast_note-2.pdf`
  (identical to the `Downloads/` copy.)

## Notes
- `00_breakdown_holes_buildup.md` — the **decomposition proposal**: break-down, 12 holes (H1–H12),
  11 patches (P1–P11), a paper dummy execution. Key holes: H1 (attribution is path-dependent),
  H2 (unit loss = dimensionality), H3 (unit loss not biologically identifiable = the
  "before/after spike" gap).
- `01_forecast_note_breakdown.md` — the **forecast note** ("clocks, Bayes panel, map G"):
  break-down, 12 holes (F1–F12), 10 patches (G1–G10), a paper dummy execution. Key holes:
  F1 (feature/label circularity), F2 (detection ≠ forecast), F4 (no policy break-even).
- `02_matchup_decomposition_vs_forecast.md` — **how the two match up**: two arms of one program
  (retrospective attribution vs prospective forecast), a shared typed-drift taxonomy, and how each
  fixes the other's fatal hole (forecast panel => identifiability for H3; decomposition =>
  label-free check for F1/F2). Includes a combined "same crash, both ends" dummy execution.
- `03_merge_decision.md` — **should they be merged?** Yes, but only *compositionally*: the
  decomposition becomes the forecast's **label generator** and the panel becomes the decomposition's
  **covariates**. Includes bad-merge-vs-good-merge, the 4-stage unified pipeline, costs/risks, a
  decision rule, and a minimal viable merged paper outline.
- `04_prediction_feasibility_test.md` — **prediction-first de-risking.** A feasibility test of the
  "functional" (forecast) arm: the falsifiable question, a 4-tier feasibility ladder (harness →
  session-scale → within-session ramp → step latency), a test spec with the crucial **persistence
  baseline**, a leakage checklist, a green/yellow/red gate, and the bridge from "can we predict?"
  to "which mechanism are we predicting?" (the component break-down).
- `05_dataset_scoping.md` — **dataset scoping (HPC lens).** Candidate multi-session iBCI datasets
  (Perich–Miller 000688, FALCON incl. human H1/H2, NeuroTask, IBL), the three binding filters, the
  "raw availability in multi-session motor sets" constraint HPC doesn't solve, a two-pronged strategy,
  and open decisions. No downloads — lookups only.
- `06_leads_primary_dataset_reduced_panel.md` — **lead-chase + decisions.** (1) FALCON NWB is
  spikes-only (no raw/LFP); IBL "Reproducible Ephys" has raw+quality+decoding but is **cross-insertion**,
  not same-probe-multi-day (corrected in `07` §6). (2) Decision: **macaque primary**, dataset =
  Perich–Miller `000688`, Step-1 baseline = reproduce **FALCON**'s cross-session decoder. (3) The
  **reduced panel**: which forecast-note features survive spikes-only data, and which (SNR, impedance,
  DREDge motion, LFP) do not.
- `07_ibl_prong2_quality_payload.md` — **the Prong-2 (IBL) quality payload.** Enumerates what the
  Reproducible Ephys pipeline computes (AP RMS, neuron yield, cluster `label`/`amp_median`/
  `noise_cutoff`, waveforms, LFP power/theta, spike depth) and maps it to the forecast-note panel and
  the decomposition's **H3** identifiability fix. §6 resolves the de-risking question: the dataset
  (Banga et al., eLife 2025, 13:RP100840) is a **cross-INSERTION** reproducibility design (10 labs,
  121 replicates targeting the same regions), **not** a same-probe-across-days chronic set — so IBL is
  the feature/QC + H3 tooling, while the cross-**session** drift axis must come from Perich/FALCON.

- `08_two_track_plan.md` — **scope decision.** The goal is **cross-session drift** (forecast +
  decomposition). **Cross-insertion reproducibility is NOT the goal** — IBL is demoted to a **resource**
  (feature/QC definitions + H3 metrics), not a second track. Guardrails + IBL's role.
- `09_trackA_persistence_spec.md` — **the spine's next deliverable.** A concrete persistence-test spec
  on Perich `000688`: the pre-registered question, FALCON-style decoder, health series, reduced-panel
  features, baselines (incl. **persistence**), leakage discipline, and the GREEN/YELLOW/RED gate.
- `10_mvp_hackathon.md` — **the hackathon MVP.** The smallest end-to-end slice: one subject/same-probe
  chain, one decoder, health(t) curve, a couple of spike-derived features, a **persistence** baseline,
  and **one figure** ("is next-session decoder failure forecastable at all?"). Scope-in/out, a 24h
  timebox, kill criteria, stretch (decomposition), and a 5-bullet demo slide.
- `11_prior_art_check.md` — **has anyone done this?** Literature scan (OpenAlex/arXiv/PubMed): the
  *forecasting* object is **unoccupied**; the closest is **MINDFUL** (Pun et al., Commun Biol 2024,
  `10.1038/s42003-024-06784-4`) which is **contemporaneous** instability measurement, not a forecast.
  Adaptation work (NoMAD/Degenhart/Sussillo) *fixes* drift rather than predicting it.
- `12_paper_list.md` — **annotated, relevance-rated bibliography** (8 categories, each paper rated
  /5 with who/what/why/how/results blurbs): sources, instability measurement, decomposition, drift
  mitigation, population/representational drift, datasets/benchmarks, ephys QC, decoders/forecasting.
  Ends with the **gaps** (no forecasting paper, no ephys decomposition paper, no chronic-motor Neuropixels).
- `17_plan_27_adapter_curves.md` — **PLAN** for experiment `27`: an **adapter grid × adaptation-data
  learning curves** (labeled vs unlabeled, with complexity/params), replacing the earlier fixed-window
  comparison. Key idea: for **unlabeled** adapters data is *free*, so a data-hungry one is a win.

## One-line takeaway
The decomposition is the forecast note's slowest clock (an integral vs a derivative). They can be
**merged — but only as "forecast the mechanisms," wiring labels<->covariates — not as two stacked
analyses.** Keep them mergeable siblings so the merge is an upside option, not a forced marriage.
