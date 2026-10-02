# Scope: cross-session is the goal (IBL is a resource, not a second track)

**Status:** decision record + plan. No downloads.

## Decision (recorded, revised)
- **The goal = cross-session drift.** The project's question: can we forecast *session N+1* decoder
  degradation in the same subject, and decompose it into mechanisms? (Faithful to both source docs:
  the forecast note and the decomposition proposal.)
- **Cross-insertion reproducibility is NOT the goal.** IBL's cross-insertion question ("why do
  independent insertions into the same region vary?", Banga et al. eLife 2025) is a *different,
  methods-reproducibility* literature — no decoder-over-days, no forecast. **Demoted from "companion
  study" to "tooling source + optional side note."**
- **IBL = tooling source only** for the spine: the feature/QC vocabulary (AP RMS, yield, cluster
  `label`/`amp_median`/`noise_cutoff`, waveforms, LFP power/theta, depth) and the H3 identifiability
  metrics. We borrow *definitions*, not its question, and not necessarily its data.

## Definition of the cross-session axis — CHRONIC SAME-PROBE

The goal, stated precisely (per the user): **same subject, same physical probe (not removed), across
the same day / next day / week / month...**

- **Same subject** (animal or human participant).
- **Same probe, not removed** — a chronically implanted device (Utah / microwire array, or chronic
  Neuropixels) that stays in place; units come and go, but the *device* does not. Explicitly excluded:
  probes re-inserted each session (acute).
- **Longitudinal** — same-day blocks, next day, week-over-week, month-over-month (the "clocks").

### Dataset fit against this definition

| Dataset | same subject | same probe (not removed) | fits? |
|---|---|---|---|
| Perich `000688` | yes (per-subject date chains) | yes — chronically implanted M1/PMd arrays | ✓ |
| FALCON M1-A / M1-B | yes (`MonkeyL`, ...) | yes — implanted M1 arrays | ✓ |
| FALCON H1 / H2 | yes (human) | yes — chronic Utah arrays | ✓ |
| IBL Reproducible Ephys | no (cross-insertion) | **no** — acute, re-inserted probes | ✗ as data |
| Neuropixels *chronic* (same probe across days) | yes | yes (NP2.0 kept in) | not found public |

**Consequence:** this definition *confirms* Perich/FALCON as the testbed and *rules out* IBL as data
(consistent with `07` §6). It also means session ordering must be **same-probe chains**, not just
same-subject.

## Track A — Spine: cross-session drift (primary)

- **Question:** does a strictly-past feature panel predict next-session decoder health better than
  persistence? (`04_prediction_feasibility_test.md`.)
- **Data:** Perich `000688` (longitudinal, 111 sessions, macaque M1/PMd) + FALCON (Step-1 baseline +
  split convention).
- **Sequence:**
  1. Reproduce a FALCON-style cross-session decoder/baseline (sanity + turnkey decoder).
  2. Build the per-session R² series on `000688` (health(t) from a frozen reference decoder).
  3. Run the persistence test with the **reduced panel** (rate, unit count, match-survival,
     factor-angle, noise corr, decode entropy).
- **IBL's role:** feature/QC definitions only.

## Track B — NOT a track: IBL as a resource only

- **The cross-insertion question is not our goal** and is not pursued as a study.
- We still mine IBL for the **feature/QC definitions** and the **H3 identifiability metrics**
  (`07_ibl_prong2_quality_payload.md`).
- Optionally, if a reviewer asks "do your panel metrics behave like IBL's?", the IBL reproducibility
  data is an *external consistency check* — a figure, not a paper.

## Guardrails

- Keep the spine's question single: **cross-session drift, forecast then decompose.**
- Do not let "we could also study cross-insertion reproducibility" creep into the paper; it is a
  different metric/literature. If it ever appears, it appears as an external validation figure only.
- Shared assets with IBL: feature *definitions*, QC-metric *definitions*, and the leakage/session-out
  discipline.

## IBL's role

| | Uses IBL data? | Uses IBL feature/QC definitions? |
|---|---|---|
| The project (cross-session) | no (optional figure only) | **yes** (borrow definitions) |

## Next step
- The persistence-test spec is written: `09_trackA_persistence_spec.md`. Remaining: lock the five open
  choices in `09` §10, then (much later) run it.
