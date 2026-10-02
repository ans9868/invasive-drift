# Prior-art check — has anyone done this?

**Status:** literature scan via open APIs (OpenAlex, arXiv, PubMed). Not exhaustive (no full Google
Scholar / Semantic Scholar pass). No downloads.

Question: has anyone built **forecasting of cross-session decoder failure** from a predictive panel —
and/or the **decomposition** into unit loss / drift / gain?

---

## The landscape (three buckets)

### A. Instability *measurement* — contemporaneous, NOT forecasting
| Work | Where | What it does | Why it's not ours |
|---|---|---|---|
| **MINDFUL** (Pun et al.) | **Commun Biol 2024**, doi `10.1038/s42003-024-06784-4` (OA, ~18 cites) | label-free instability score on BrainGate; correlates ~0.7–0.9 with closed-loop error over weeks | **measures instability now**; no forecast, no classes/typing, no policy |
| Intra-day signal instabilities affect decoding | **J Neural Eng 2013**, doi `10.1088/1741-2560/10/3/036004` (~260 cites) | shows signal instabilities *affect* decoding | intra-day, contemporaneous; no prediction |
| Intracortical recording stability in human BCI users | **J Neural Eng 2018**, doi `10.1088/1741-2552/aab7a0` (~167 cites) | characterizes recording stability | descriptive, not predictive |

### B. Adaptation / alignment — *fix* the drift, don't forecast it
- Degenhart et al. 2020; Sadtler; Karpowicz et al. 2022; Farshchian et al. (adversarial);
  **NoMAD** (Nat Commun 2025); **Sussillo et al. 2016** "Making brain–machine interfaces robust to
  future neural variability" (doi `10.1038/ncomms13749`).
- All **react or infer**; none *forecast* failure ahead of time, none *type* the cause.

### C. Forecasting / proactive recalibration — **essentially empty**
- Searches for "predicting when to recalibrate", "predicting decoder accuracy across days",
  "decoder failure prediction" returned **no** matches for the forecasting object.
- No paper found that: (i) predicts decoder degradation *ahead of time* from a feature panel,
  (ii) *types* the likely cause, and (iii) drives a policy.

## Verdict

- **The forecasting object is unoccupied.** The closest (MINDFUL) is a *contemporaneous* instability
  scalar — it tells you the decoder is *currently* off, not that it *will be*. This matches the
  forecast note's own claim ("instability metrics exist; they are contemporaneous and untyped").
- **The decomposition object** (attributing a cross-session drop to unit loss/drift/gain) also has no
  direct ephys prior art; the anchor is the Ca-imaging study (Hayashi et al., under review). Adjacent:
  "To sort or not to sort: impact of spike-sorting on decoding" (2014), and a 2023 Front Comput Neurosci
  *simulation lesion catalog*.
- So this is **novel but adjacent** — a real gap, sitting between a mature *measurement* literature
  (MINDFUL) and a mature *adaptation* literature (NoMAD/Degenhart), with a thin *prediction* area.

## Caveats
- Searches used OpenAlex/arXiv/PubMed; **Semantic Scholar was rate-limited** — a fuller pass (Google
  Scholar, forward-citations of MINDFUL) may surface 2024–2025 preprints.
- The forecast note already did a partial pass (MINDFUL, DREDge, NoMAD, Farshchian, Degenhart) and
  concluded the exact object is "thin" — this scan agrees.

## Implication for the project
- The MVP (`10`) is a **legitimate first cut** at an unoccupied niche.
- **MINDFUL is the bar to beat**, and the honest framing is: *"MINDFUL tells you it's degrading now;
  we ask whether it can be seen coming."* That is the novelty sentence.
- **Forward-cite MINDFUL** (and check its follow-ups) before writing anything — that's the fastest way
  to be sure the niche is still open.
