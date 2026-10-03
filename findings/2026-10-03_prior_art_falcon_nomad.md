# 2026-10-03 — Prior art read: FALCON + NoMAD

**These are the two papers that define the evaluation our project is implicitly competing in.** Read in full
(text extracted with `pdftotext -layout`; PDFs in `research-papers/`, gitignored). Reading time ~1 h.

- **FALCON** — Karpowicz et al., *Few-shot Algorithms for Consistent Neural Decoding (FALCON) Benchmark*,
  NeurIPS 2024 Datasets & Benchmarks track; bioRxiv `10.1101/2024.09.15.613126` (posted 2024-10-31).
  `research-papers/falcon.txt` (1884 lines).
- **NoMAD** — Karpowicz, Ali, Wimalasena, Sedler, Keshtkaran, Bodkin, Ma, Rubin, Williams, Cash, Hochberg,
  Miller & Pandarinath, *Stabilizing brain-computer interfaces through alignment of latent dynamics*,
  **Nature Communications 16:4662 (2025)**, doi `10.1038/s41467-025-59652-y`.
  `research-papers/natcomm.txt` (1055 lines). This is the peer-reviewed version of the earlier bioRxiv
  `2022.04.06.487388` already in `drafts/12_paper_list.md`.

**Headline:** FALCON is the benchmark; NoMAD is the current state of the art *in that benchmark*. FALCON
already releases the monkey M1 dataset we are using (Dandiset `000941`), already scores with R², and already
publishes the exact baseline numbers our zoo is reproducing. We are not in an unclaimed space — we are in a
space with a published scoreboard, and our job is to be *legible on that scoreboard*.

---

## 1. FALCON — the evaluation standard

### 1.1 Design

- **5 datasets**, one 0–5 min calibration split + a withheld evaluation split per session:
  - `M1-A`, `M1-B` — macaque reach & grasp, **FMA** arrays in M1, 16 EMG channels (M1-A: 4 held-in sessions /
    5 days, 53–61 min calibration each; 3 held-out sessions / 21 days with only **1.1–2.2 min** calibration).
  - `M2` — macaque **finger movement** (Indiana). `H1` — human, Utah array, 7-DoF robotic arm command
    (3D limb kinematics + 1D rotation + 3D grasp). `H2` — human handwriting → **WER**. `B1` — zebra finch
    songbird (Neuropixels in RA) → **MSE** on spectrogram.
- Format **NWB**; hosted on **DANDI**, **CC-BY-4.0**. IDs: M1-A `000941`, M1-B `001209`, M2 `000953`,
  H1 `000954`, H2 `000950`, B1 `001046`.
- Splits are **contiguous in time**: `held-in` (full data released) / `held-out` (tiny calibration released).
  An evaluation split is withheld from *both*.
- **Evaluation is continuous, causal, open-loop, timestep-by-timestep, and trial labels are NOT provided.**
  Scoring via EvalAI + Docker. → *This is the strongest single design lesson for us: no trial structure.*

### 1.2 The metric, exactly (Section 2.3 + A.2)

> "accuracy is reported using the coefficient of determination (R²), computed as a **variance-weighted
> average across the R² of individual motor covariates**."

In code: `sklearn.metrics.r2_score(y, y_hat, multioutput='variance_weighted')`.

- A **constant-mean prediction scores R² = 0**; max is 1. Unbounded below.
- R² "heavily penalizes predictions that are shifted from the expected center point."
- Metrics computed **per session**, then **mean ± std across sessions**, reported **separately for held-in
  and held-out**. They do **not** report per-bin CIs — the across-session spread *is* the error bar.
- Their own A.2 warning: "a poor-scoring model may not necessarily have unreasonable outputs. We recommend
  visualizing predictions."

### 1.3 Baselines and the published numbers (Table 1) — M1-A, held-out / held-in R²

| Class | Method | M1-A | M2 | H1 |
|---|---|---|---|---|
| OR (oracle, upper bound) | Wiener Filter | 0.53±0.04 / 0.54 | 0.26±0.03 / 0.27 | 0.21±0.04 / 0.24 |
| OR | RNN (LSTM) | 0.75±0.05 / 0.75 | 0.56±0.04 / 0.59 | 0.44±0.13 / 0.51 |
| OR | NDT2 Multi | 0.78±0.04 / 0.77 | 0.58±0.04 / 0.62 | 0.63±0.08 / 0.68 |
| **ZS (static, lower bound)** | **Wiener Filter** | **0.34±0.06 / 0.46** | 0.06±0.04 / 0.15 | 0.16±0.03 / 0.20 |
| ZS | RNN | −0.60±0.45 / 0.52 | −0.07±0.23 / 0.20 | 0.09±0.18 / 0.31 |
| FSU (few-shot unsup) | CycleGAN + WF | 0.43±0.04 / 0.61 | 0.22±0.06 / 0.32 | 0.12±0.06 / 0.15 |
| **FSU** | **NoMAD + WF** | **0.49±0.03 / 0.64** | 0.20±0.10 / 0.35 | 0.13±0.10 / 0.21 |
| FSS (few-shot sup) | NDT2 Multi | 0.59±0.07 / 0.77 | 0.43±0.08 / 0.63 | 0.52±0.04 / 0.62 |

H2 (WER, ↓): RNN Multi OR 0.15/0.11; +LM OR 0.03/0.02; ZS 0.53/0.11; +LM ZS 0.37/0.02; CORP (TTA) 0.11/0.02.
B1 (MSE ×1e-4, ↓): EnSongdec OR 7.47/5.61; ZS 21.8/5.18.

**Zero-shot WF loses 0.12–0.28 R² from held-in to held-out.** The static RNN on M1-A is *catastrophic*
(−0.60 ± 0.45) — higher-capacity single-session decoders are the least stable. Meanwhile CycleGAN and NoMAD
recover only part of the gap (0.43, 0.49 vs 0.53 oracle). **NDT2 Multi (few-shot *supervised*) is the best
available at 0.59.** No method reaches oracle.

### 1.4 The calibration-budget table (Table 3) — *the single most useful table for us*

WF decoders trained **from scratch** on the *held-out* calibration split vs the *held-out oracle* split:

| Training data | M1 | M2 | H1 |
|---|---|---|---|
| Held-out calibration (~1 min) | **0.24±0.04** | 0.14±0.05 | 0.11±0.03 |
| Held-out oracle | 0.53±0.04 | 0.26±0.03 | 0.21±0.04 |

Their stated purpose: *"to ensure that the few-shot problem was well-represented, we established that the
held-out calibration splits were insufficient to train new linear decoders on their own."*

→ **This is the field's canonical, honest answer to "what does a calibration budget buy you?"** and it is
exactly the shape of our `staleness.py` expanding-calibration-budget curve. Our "N_fracs" sweep is the
same experiment with finer granularity. We should report it in the same two-reference framing
(from-scratch vs oracle) rather than in raw R².

### 1.5 Continuous vs trialized evaluation (A.5.1) — a warning that lands directly on us

- FALCON trains decoders both **trialized** (data split into single behavioural trials) and **continuous**
  (fixed-length segments), and evaluates both ways (trialized = trial-change signal may reset model input).
- **Trialized-trained decoders collapse under continuous evaluation in M1 and M2** ("performance dropped
  precipitously"). Continuous-trained decoders *do not* degrade with long history — but their performance
  is **sensitive to the length of history used**.
- Their reading: trialized models "are exploiting trial structure (distinct behavior at the start, middle,
  and end of trials) to reduce uncertainty about decoding at different timepoints... likely to not benefit
  closed loop control."
- H1 is the anomaly: performance *keeps improving* with longer context even under continuous evaluation,
  which they flag as "a particularly concerning edge case that may be exploited in FALCON leaderboards."

→ **Consequences for us.** (i) Our grid is already continuous and trial-label-free — correct, keep it.
(ii) Any decoder in our zoo that gets an edge from segment boundaries is measuring an artifact. (iii) The
**history-length sensitivity** is the thing to sweep, and it is a *confound that grows with R²*, i.e. it
mimics "drift". If a decoder's performance depends strongly on `bin_ms × history_bins`, part of what we call
within-session degradation could be a context-length effect.

### 1.6 Limitations FALCON states about itself (Discussion, honest and quotable)

1. **Trial structure is still implicit** — datasets are cued/stereotyped even though labels are withheld;
   "FALCON is susceptible to models that exploit these gains."
2. **Open loop ≠ closed loop.** "worse decoder predictions may not yield poor control." They explicitly
   leave open→closed-loop evaluation as "an open problem for the field."
3. **Few subjects** — the cost of intracortical experiments is "prohibitive to providing data from the high
   number of subjects needed to support claims of subject generalization."
4. **Spikes only** — thresholding is researcher-discretionary (they released raw 30 kHz for M2/B1 to let
   people avoid thresholding).
5. Metrics alone don't capture everything (A.2).

---

## 2. NoMAD — the current state of the art, and its own within-session numbers

### 2.1 Method (Nat Commun Methods)

Two stages:

- **Day 0**: fit **LFADS** (GRU Encoder → latent Z → Generator RNN → factors F(t) → Poisson rates) on one
  supervised session. Optional **behavioural readout** from Generator states (force + d(force), or cursor
  position + velocity) as a *second training objective* to make the manifold behaviour-predictive with fixed
  hyperparameters (avoids AutoLFADS sweeps). Then fit a **Wiener filter from Generator states → behaviour**.
- **Day K**: **freeze** Generator/Encoder. Learn *only* four things by unsupervised alignment:
  (1) a **feedforward Alignment network** (2-layer Dense, ReLU, **identity init**) that rewrites Day-K input,
  (2) the low-D **read-in** matrix, (3) the factors readout, (4) the rates readout.
  Loss = **KL divergence between the Day-0 and Day-K Generator-state distributions** (fitted as full-
  covariance multivariate normals, m = 100 units, closed-form Eq. 1) **+ Poisson reconstruction cost**,
  as a weighted sum. Adam + gradient clipping + LR annealed ×0.95 on plateau. Then apply the frozen
  Day-0 Wiener filter to Day-K Generator states.
- Chops: **600 ms with 120 ms overlap** (monkeys; 30 bins / 6 bins overlap at 20 ms), 1000/350 for human.
- A **z-scored read-in** lets the same model absorb a *changed channel count* between days.
- Nothing here uses Day-K behaviour. Purely unsupervised.

### 2.2 Causal inference — they took the same constraint we did

> "we perform inference in a causal manner... a **sliding window** of observed data, where the majority of
> this window consists of **previously observed** data. At each time step, one new bin of input data is added
> to the window, resulting in one new bin of the model's inferred output."

Plus a runtime shortcut: **take the posterior mean instead of sampling** (minimal latency), and they show
(Suppl. Fig. 3, 13) minimal performance change and < 10 ms latency in online use.

→ Our causal/sliding-window framing is **not** idiosyncratic; it is what the SOTA does and defends.

### 2.3 Their per-channel normalisation is *literally* our normalizer finding

> "**Normalization.** To account for large changes in the firing rates of individual channels across days,
> we normalized **each channel to have zero mean and unit standard deviation**... we first smooth the data
> with a 20 ms Gaussian kernel... compute a per-channel mean and standard deviation... applied to the LFADS
> input spiking data (not smoothed) **before the low-dimensional read-in layer**."

They also **verified it isn't cheating** (Suppl. Fig. 14 compares with/without).

→ This is precisely the mechanism our `2026-10-03_normalizer_perspective.md` isolated: the consistent-but-small
gain came from **per-unit gain whitening**, *not* drift realignment. **NoMAD does the same thing** — it just
does it as a pre-processing layer *and then* adds a KL-constrained alignment network on top. Our result
therefore reads as: *"we reproduced the cheap first half of the NoMAD recipe; the expensive second half is
where the rest of the gain lives."* That is a **useful, citable, honest positioning**, not a failure.

### 2.4 Their stability metric IS our "staleness" — adopt it verbatim

They quantify degradation with **half-life in days**:

1. Compute median R² within each 5-day bin.
2. Convert to SNR: **SNR := −10·log₁₀(1 − R²)** (from Makin et al. 2020).
3. Fit exponential decay **y = A·e^(−Bt)** (y = bin median SNR; **append the Day-0 within-day median as the
   first point at t = 0**).
4. **half-life = ln(2)/B** days.

Also: **"We term evaluations with negative R² to be decoding failures"** — a failure *count* is reported
alongside the median. Statistics: **Wilcoxon signed-rank** across all session pairs, one-sided.

→ We should (a) adopt `SNR = −10log10(1−R²)` + exponential fit + half-life as the headline staleness number,
(b) report **decoding-failure counts** (R² < 0), (c) use signed-rank across blocks/pairs rather than a t-test
on correlated bins. This directly answers our open item #1 (CIs) and #2 (staleness.py).

### 2.5 NoMAD results (all R²; negative = failure)

**Isometric force, 20 sessions / 95 days** (all session pairs, 380 pairs):

| Method | within-day Day 0 (upper bound) | across-session median R² | half-life | failures |
|---|---|---|---|---|
| NoMAD (+WF) | **0.971 [0.965, 0.974]** | **0.91** | **208.7 d** | **0** |
| ADAN | 0.916 [0.908, 0.929] | 0.65 | 76.7 d | 0 |
| Aligned FA (Degenhart) | 0.749 [0.722, 0.761] | 0.59 | 45.1 d | 51 |
| Static (+WF) | 0.842 [0.834, 0.846] | 0.14 (<5 d pairs), negative beyond | ~1 d | **223** |

**Unloaded reaching, 12 sessions / 38 days:**

| Method | within-day Day 0 | across-session median R² | half-life | failures |
|---|---|---|---|---|
| NoMAD | 0.918 [0.896, 0.934] | 0.78 | 57.9 d | 0 |
| ADAN | — | 0.29 | 7.12 d | 15 |
| Aligned FA | — | 0.17 | 1.03 d | 53 |

Cross-check against FALCON: FALCON's M1-A NoMAD+WF is **0.49±0.03 (few-shot held-out)** vs 0.64 (held-in).
The two papers are consistent once the held-in/held-out split is accounted for.

### 2.6 ⚠️ The NoMAD group's **own** within-session half-lives — direct prior art for our paper

Closed-loop human iBCI (BrainGate participant T11), re-calibrating **per block** while the session runs:

| Method | session 1 R² | session 2 R² | **within-session half-life** |
|---|---|---|---|
| Static | 0.32 [0.23, 0.37] | 0.49 [0.45, 0.51] | **3.26 min** / 7.02 d |
| RTI (retrospective target inference) | 0.46 [0.41, 0.49] | 0.67 [0.63, 0.70] | **3.21 min** / doubling time 2.40 h |
| NoMAD | 0.66 [0.63, 0.70] | 0.79 [0.76, 0.83] | **5.60 h** / 21.05 h |
| NoMAD + RTI | 0.72 [0.68, 0.79] | 0.83 [0.78, 0.85] | **11.73 h** / 4.47 d |

And they say explicitly:

> "Another class of approaches... has been demonstrated to be highly effective **when applied within
> sessions of closed-loop iBCI control over timescales of minutes to hours**."

**This is the key datum for us.** Within-session instability is a *known, quantified, minutes-to-hours
phenomenon* — and the SOTA's own numbers say a static decoder within a session has a **half-life of ~3
minutes**. Our `CO-20131003` 2.4-minute evaluation window sits squarely inside that regime.

What they do **not** do anywhere:
- characterise within-session drift as a **curve you plot against elapsed time** (for them half-life is a
  summary of a *recalibration strategy*, not a scientific object),
- **decompose** the within-session drop into unit loss / drift / gain,
- ask whether the within-session drop is **forecastable**.

→ That is still our lane. But we must now **cite these half-lives** and frame our contribution as
*"describe and forecast the within-session decay curve that the alignment literature only summarises with a
post-hoc half-life"* — **not** as "discovering that within-session drift exists."

### 2.7 Limitations NoMAD states about itself

1. **Requires a stable manifold↔behaviour relationship** over the alignment horizon: "over extremely long
   timescales (e.g., many months to years)... the likelihood increases that this method's assumptions of a
   stable manifold and consistent behavior will be violated."
2. **Needs behaviour spanning a large enough region of behavioural space** for periodic alignment to work.
3. **Single reference session** alignment beats sequential, because chaining "aggregates error" and it is
   "not clear how a method should handle any reductions in alignment quality." (Justifies our
   align-to-Day-0 design.)
4. **Bin size is a real tradeoff** — they criticise ADAN/Aligned FA for "larger bin sizes... which would incur
   latencies inappropriate for iBCI use"; large bins "could cause long latencies and degrade closed-loop iBCI
   performance." They use **20 ms** throughout. → our 20 ms is the defensible choice.
5. **Offline ≠ online**: "maximizing offline decoding accuracy does not necessarily lead to improvements in
   online performance."
6. **Nonlinear decoders can overfit and degrade online control** — they deliberately use a *linear* readout to
   probe representation quality, citing the representation-learning literature. → supports our
   "linear probe as the honest headline, nonlinear as secondary" plan.
7. Manifold stabilisation has only been tested on **single-behaviour** datasets; different behaviours may
   occupy distinct manifolds.

---

## 3. What this changes for us — concrete decisions

### 3.1 Schema / metric changes (small, mostly analysis-time)

| # | Change | Why | Where |
|---|---|---|---|
| 1 | Compute R² as **variance-weighted multi-output R²** and *state it*, alongside our current pooled R² | exact FALCON/NoMAD comparability | `metrics.py::velocity_metrics` — add `r2_vw` |
| 2 | Add **`snr = −10log10(1 − r2)`** + **exponential-decay fit** + **`half_life_bins`** | it is the field's stability currency | `metrics.py` + `staleness.py` |
| 3 | Add **`failure = r2 < 0`** flag / count per cell and per session | NoMAD's own convention; our grid can already do it | summary step |
| 4 | Report **held-in style "oracle"** (refit on eval block) next to **zero-shot static** | FALCON's OR/ZS framing makes any number interpretable | grid already has `refit_decoders` |
| 5 | Add **from-scratch calibration curve** (k minutes → R²) i.e. FALCON Table 3 shape | this is the field's "what does a budget buy" table | `staleness.py`, `N_fracs` |
| 6 | Use **Wilcoxon signed-rank across paired blocks** instead of t-tests on bins | correct for the autocorrelation; sidesteps the CI problem | summary/analysis step |

Items 1–3 and 6 are **analysis-time** — the persisted per-row grid numbers do not change, so
**no re-run is needed for them**. Items 4–5 need the grid run (which is already planned).

### 3.2 Our numbers land on the published scoreboard

- FALCON M1-A **zero-shot WF = 0.34 ± 0.06 held-out** on 16 EMG channels (decode EMG from spikes).
- Our zoo on `CO-20131003` (velocity from spikes): **ridge 0.357**, `wiener 0.404`, `kf_posvel 0.393`.
- FALCON M1-A **few-shot trained-from-scratch WF on ~1 min = 0.24**.

→ We are in the right range, on the right dataset family, with the right metric. **The wildcard is that
FALCON decodes EMG while we decode kinematics** — worth a one-line caveat, and worth checking whether the
`000941` release we hold ships EMG (it does — `M1-A` is 16-muscle EMG). If so, matching FALCON exactly is a
*cheap credibility win* and gives us a literal apples-to-apples number.

**Lead worth checking (same data family?):** NoMAD's *unloaded reaching* dataset is `Chewie_CO_2016`, a
macaque planar-manipulandum centre-out task, citing Perich et al. *Neuron* 2018 / Gallego et al. *Nat
Neurosci* 2020 / Ma & Rizzoglio et al. *eLife* 2023 — i.e. the **same Miller-lab centre-out lineage** as our
Perich DANDI `000688` sub-C ("CO-*" session IDs, 53 sessions 2013–2016). If our `CO-*` sessions overlap the
same animal/protocol, then NoMAD's published **across-session** half-life (57.9 d on reaching) and our
**within-session** decay curve would sit on *the same data family* — which would make "the across-session
literature never plots the within-session axis" a direct, checkable comparison rather than a rhetorical one.
**Action:** confirm the animal/protocol lineage before relying on this framing.

### 3.3 The persistence floor, restated in FALCON's language

FALCON's R² anchor point is **"a constant mean prediction achieves R² = 0"**. Our persistence floor
(R²_persist = 0.995 on `CO-20131003`) is *not* a competing model — it is `R² = 1 − Var(Δv)/Var(v)` for a
20 ms bin, which is high purely because **velocity is autocorrelated at 20 ms**. Restated honestly:

- **raw R² is directly comparable to FALCON** (both are variance-normalised against the mean), so our
  headline R² *is* the FALCON number and needs no correction;
- the persistence floor is a statement about the **behaviour**, not about the decoder — it is the
  predictability ceiling implied by `bin_ms` alone;
- therefore it should be reported **once, as a property of the task/bin size**, and never as a rival decoder
  in the results table. Same for `r2_target` (condition-mean) — a *task* ceiling, not a decoder.

This is consistent with the "rankings are all safe / only the absolute framing changes" conclusion:
`r2_persist` and `r2_target` are cell-invariant.

### 3.4 CIs — now settled by prior art

Nobody in this literature reports per-bin CIs. They report:
- **mean ± std across sessions** (FALCON), and
- **[Q1, Q3] across session pairs** + **Wilcoxon signed-rank** + **failure counts** (NoMAD).

So the fix for our over-confident CIs is: **stop computing within-session CIs from bins at all**. Either
(i) report the across-block IQR as NoMAD does, or (ii) if a within-session CI is genuinely needed, use a
**block bootstrap** with the block length set to the integrated autocorrelation time (~830 bins ≈ 16.6 s)
— never a naive i.i.d. bootstrap. This removes the ρ₁≈0.9976 → N_eff≈9 embarrassment entirely.

### 3.5 What we should *not* claim

- **Not** "within-session drift is unstudied" — NoMAD reports within-session half-lives of 3.2 min–21 h.
- **Not** "nobody aligns latent spaces" — Degenhart/ADAN/NoMAD/CycleGAN do, and NoMAD gets 208.7-day
  half-life on isometric force.
- **Not** "our MSE-on-velocity objective is the wrong objective" — FALCON's own movement baselines are a
  **Wiener filter (ridge with history)** and an RNN, both trained on behaviour. NLB uses Poisson NLL because
  its *object* is neural-rate prediction, not behaviour decoding. **We are on the field's objective.**

### 3.6 What the reading *does* not answer

- **Does FALCON/NoMAD ever bind a *time-varying* drift to a *within-session elapsed-time axis*?** No. That
  remains the novel axis.
- **Does either paper forecast the drop?** No. FALCON calls crash-prediction an open problem; NoMAD measures
  it retrospectively (half-life after the fact).
- **Does either decompose the drop into unit-loss / drift / gain?** No — the Ca-imaging anchor (Hayashi)
  is still the only decomposition.

---

## 4. Status of our six open items after this reading

| # | Open item | Status after reading |
|---|---|---|
| 1 | Persistence-floor framing | **Resolved.** Frame as a property of the behaviour at `bin_ms`; raw R² needs no correction (FALCON normalises against the mean too). See §3.3. |
| 2 | CIs over-confident | **Resolved — approach decided.** Drop per-bin CIs; use across-block IQR + Wilcoxon signed-rank (NoMAD convention) and, if needed, a block bootstrap at the ~16.6 s autocorrelation block length. See §3.4. |
| 3 | τ smoothing sweep | **Still worth running, and now better motivated.** FALCON's WF history sweep is the same experiment (they picked 600 ms for M1 by elbow). NoMAD uses 20 ms bins + 40 ms Gaussian for the *comparison* methods and 20 ms Gaussian for its own normalisation. So {60, 120, 240} ms is a sensible, defensible grid — and 240 ms should be reported *with* the note that it costs closed-loop latency (NoMAD's own criticism). |
| 4 | `SS_free` + innovation R² | **Still our own addition** — not in either paper. Both papers only report R², so our skill-score/innovation decomposition is genuinely additive. Keep, and cite FALCON's A.2 "visualize predictions / metrics don't capture everything" as the justification. |
| 5 | Reframe at top of write-up | **Now has its citation backbone** — §2.6 gives the within-session half-lives to cite, §3.5 gives the four "do not claim" statements. |
| 6 | FALCON read | **Done.** See §1. |

## 5. Provenance

```bash
cd research-papers
pdftotext -layout "School — https::www.biorxiv.org:content:10.1101:2024.09.15.613126v2.full.pdf" falcon.txt
pdftotext -layout s41467-025-59652-y.pdf natcomm.txt
```

`research-papers/` is gitignored (copyrighted PDFs + derived text). Page/line numbers above refer to the
compressed `.c.txt` copies (`tr -s ' '`), which preserve line numbering: `falcon.c.txt` 1884 lines,
`natcomm.c.txt` 1055 lines.

**Not yet read / to follow up:** FALCON Supplementary Tables 1–7 (exact NoMAD/ADAN hyperparameters for the
FALCON re-implementation), NoMAD Supplementary Figs. 3, 5, 9, 14, 15, 16 (causal inference, Aligned-FA
active-trial dependence, chain-alignment, normalisation control, I=4 vs I=1, latent dims), and
`github.com/snel-repo/nomad` (published code) + `github.com/snel-repo/falcon-challenge`
(`decoder_demos/sklearn_decoder.py` = the reference WF).
