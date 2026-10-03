# 2026-10-03 — Framing note: the adapter grid (written **before** the grid ran)

**Status.** This is step 1 of the revised sequence in `../drafts/18_next_steps_before_after_paper_read.md`.
It is written *before* `run_grid.py` has been run over the 53 sessions, deliberately: the framing should
choose the grid's columns, not the reverse. When the grid's own findings note is written, it should open by
pointing here.

**What this note is.** The three registers we report in (classical / ours / ceiling), the **pre-registered
prediction** the grid is designed to test, the measurement conventions adopted from the field, and the
claims we refuse to make. Paper content and citations: `2026-10-03_prior_art_falcon_nomad.md`.

---

## 1. The three registers

Every number in this project belongs to exactly one of three registers. Conflating them is what made the
earlier write-ups read as over-confident.

| Register | What it is | Example | How to report |
|---|---|---|---|
| **Classical** | Raw R² of a decoder on held-out rows, normalised against the **mean** | ridge `0.357` on `CO-20131003` | Directly comparable to FALCON/NoMAD. Headline number for "does this decoder work". |
| **Ours** | Skill scores that separate *how much* of the decodable signal is captured from *how* it is captured — `SS_free`, innovation R², `r2_recon` | — | The contribution. Neither FALCON nor NoMAD computes anything but R², so this is additive. |
| **Ceiling** | Quantities that are properties of the **task or the behaviour**, not of any decoder | `r2_persist ≈ 0.995` (`lag1`, 20 ms bins); `r2_target = 0.671` | Reported **once**, as task context. **Never** in a results table next to decoders. |

**Why the ceiling register exists.** `r2_persist = 0.995` is not a strong decoder — it is
`R² = 1 − Var(Δv)/Var(v)` for a 20 ms bin, which is high purely because **velocity is autocorrelated at
20 ms**. It is the predictability ceiling implied by `bin_ms` alone. It says nothing about neural data and
nothing about any method we are comparing. Same for `r2_target` (the direction-condition mean): a *task*
ceiling.

**The consequence for raw R².** FALCON's own anchor is *"a constant mean prediction achieves an R² of 0"* —
the same normalisation we use. So our raw R² is **already** on FALCON's scale and needs **no correction**.
This is now **proved, not assumed**: pooled R² is *identically* sklearn's
`multioutput='variance_weighted'` R² (the FALCON §2.3 / NoMAD Eq. (4) metric), because
`Σ_d w_d(1 − num_d/w_d)/Σ_d w_d = 1 − Σ_d num_d/Σ_d w_d`. Verified numerically to `1.4e-13`
(`grid_within_session_27/selftest_metrics.py`). The persistence floor does not require re-scaling
anything; it only requires us to stop presenting it as a competing decoder.

**What is invariant.** `r2_persist` and `r2_target` are **cell-invariant** (they depend only on `y` and
`dirbin`). Therefore every adapter/decoder comparison, every learning curve and every 53-session aggregate
computed so far stands as computed. **Only the absolute framing changed** — no result was invalidated.

---

## 2. The pre-registered prediction (the spine of the grid)

### 2.1 Two prior findings that constrain the answer

- `2026-10-03_lowrank_normalizer.md`: **landmark low-rank rotation alignment fails — negative at every k.**
  Combined with `2026-10-03_manifold_motion_and_repeat.md` ("not a rigid moving object"; "naive landmark
  alignment *hurts*"), the picture is that within-session drift is a **deformation**, not a **rotation**.
- `2026-10-03_normalizer_perspective.md`: moment re-matching gives a **real, consistent but small** gain
  (+0.015 R², 94% of sessions, p ≈ 5e-12), and the mechanism is **per-unit gain whitening** — the
  unit-shuffled control changes nothing, so it is not drift realignment.

### 2.2 The adapters are already a ladder in *moment order*

From `adapters/feature.py` (names as in `grid_within_session_27/config.json`):

| Adapter | Family | What it matches | Params (d ≈ 100) |
|---|---|---|---|
| `identity` | — | nothing (no-op control) | 0 |
| `shuffled_ref` | control | per-unit moments, **reference permuted across units** | 0 |
| `mom_global` | moment | one global mean + std | 2 |
| `mom_diag` | moment | **per-unit** mean + std (causal; moments from the fit pool) | 2d ≈ 200 |
| `mom_diag_self` | moment | per-unit mean + std, two-pass (non-causal) | 2d |
| `cov_lowrank` | moment | mean + **rank-5** covariance | dk + k²/2 ≈ 525 |
| `zca` | moment | **full-rank** covariance: `A = C_f^{-1/2} C_0^{1/2}` (min-MSE linear map) | d² ≈ 10 000 |
| `subspace` | **direction-only** | top-k principal subspace, orthogonal map (no scale) | 2dk ≈ 1 000 |
| `centroid_proc` | **direction-only** | orthogonal **Procrustes on direction landmarks** (GRAY: uses target identity) | ~d(K−1) ≈ 700 |
| `null_proj` | moment, restricted | per-unit moments projected onto the decoder's **row space** | 2d |

The two families are separated by whether the map can change **scale/shape** or only **direction**:

- **Moment family** (`mom_global` → `mom_diag` → `cov_lowrank` → `zca`): monotone in moment order.
- **Direction-only family** (`subspace`, `centroid_proc`): scale-free, purely orthogonal.

### 2.3 The same ladder explains *why* the field's methods split

| Field method | What it actually does | Nearest adapter cell |
|---|---|---|
| **NoMAD** — per-channel z-score of smoothed spikes (Methods) | per-unit mean + std computed **from that day's full session**, then applied to that day | **`mom_diag_self`** (two-pass) |
| **NoMAD** — KL divergence between **full-covariance multivariate normals** of Generator states | first- + second-order, *cross-channel*; alignment net trained **offline on the day's data** | **`zca`** (its linear surrogate, two-pass) |
| **Aligned FA** (Degenhart) — Procrustes / orthonormal transform of stable loading rows | direction only | **`subspace` / `centroid_proc`** |
| **ADAN** (Ma et al.) — adversarial reconstruction-loss match | distributional, nonlinear | *(no cell yet; deferred to `28`)* |

So **NoMAD's two-stage recipe is, to first order, our `mom_diag_self` then our `zca`.**

**⚠️ Two independent axes, not one.** NoMAD's *statistics* are computed from the whole day and applied to
that day — i.e. **two-pass**. Its *inference* is causal (sliding window), but the alignment and
normalisation statistics are not. Our grid separates these cleanly:

- **Moment-order axis**: `mom_global` → `mom_diag`/`mom_diag_self` → `cov_lowrank` → `zca`.
- **Causality axis**: `mom_diag` (moments from the **fit pool** — online-safe) vs `mom_diag_self` (moments
  from the **data being decoded** — two-pass). `feature.py` documents `mom_diag_self` as *"the f25 win"*,
  which means **the earlier +0.015 R² result was the non-causal variant.**

This matters because the two axes can be confused: a gain that only appears in `mom_diag_self` is a gain we
cannot ship to a closed-loop decoder without a warm-up window. NoMAD accepts that cost (it has a whole
calibration block); a within-session detector may not be able to.

**Direction-only family.** Aligned FA sits in the direction-only family — the family our own findings say
*should* fail against a deformation.

This is consistent with the field's published ranking — NoMAD median R² 0.91 / half-life 208.7 d vs Aligned
FA 0.59 / 45.1 d, on the same isometric data — **but NoMAD never explains why the winning class of method is
the winning class.** They report that it wins; they do not connect the win to the geometry of the drift.


### 2.4 The prediction, and what would falsify it

> **P1.** Within the moment family, the adapter increment over `identity` is **positive and monotone in
> moment order**: `mom_global` < `mom_diag` ≤ `cov_lowrank` ≤ `zca` in mean ΔR².
>
> **P2.** Within the direction-only family, the increment is **≈ 0 or negative**: `subspace` and
> `centroid_proc` do **not** beat `identity`, and at least one is negative.
> ⚠️ **The two members are not label-comparable.** `subspace` is clean (unlabeled); `centroid_proc` is
> **gray** (it uses target identity). P2's *primary* test is therefore `subspace` vs `identity`; the
> `centroid_proc` comparison is reported **separately** and cannot be pooled with it.
>
> **P3.** The increment is **not** explained by unit-agnostic statistics: `mom_diag` must beat
> `shuffled_ref`, and the *size* of that gap is the identity-specific component.
>
> **P4.** *(Relocated — NOT testable by the grid.)* ΔR² should track the **magnitude of the drift**, not
> elapsed time. The adapter grid emits one row per cell with `block_idx=-1` / `t_start_min=NaN`, so it has
> **no time axis**; this prediction belongs to `staleness.py`, or must be restated at session level
> (ΔR² vs the session's `ctx_sess` drift magnitude across the 53 sessions). Left here as an explicit
> mismatch rather than silently dropped.
>
> **P5.** Most of the moment-family increment is **two-pass**, not causal: `mom_diag_self` > `mom_diag` by
> a margin comparable to the total increment. If `mom_diag ≈ mom_diag_self`, the gain is genuinely
> online-safe.

**Falsifiers.** P1 dies if `zca` or `cov_lowrank` do not beat `mom_diag` (→ second-order carries nothing
beyond per-unit gain). P2 dies if `subspace` beats `mom_diag` (→ the drift *is* rotation-like after all, and
`lowrank_normalizer.md`'s negative result was a landmark/estimation artifact, not a geometry fact). P3 dies
if `shuffled_ref` matches `mom_diag` (→ the gain is generic, as `normalizer_perspective.md` already suspects
at this scale). P4 dies if ΔR² tracks elapsed time with drift magnitude held fixed. P5 dies if
`mom_diag_self ≈ mom_diag` — which would be the *good* outcome, because it would mean the correction is
deployable online.

**Why this is worth pre-registering.** It converts the grid from a 12-adapter fishing expedition into a
**single ordered hypothesis about moment order**, tested on 53 sessions. It also produces a mechanistic
statement the alignment literature cannot currently make: *the class of alignment method that works is
determined by which moment order the drift lives in.*

---

## 3. Measurement conventions (adopted from the field — locked)

These come from `2026-10-03_prior_art_falcon_nomad.md` §3.1 and supersede earlier plans.

**Accuracy**
- Report **`r2_all`** — and note it is **already** FALCON's metric. `r2_all` (pooled) is *identical* to
  sklearn `r2_score(y, ŷ, multioutput='variance_weighted')`, which is what FALCON §2.3 and NoMAD Eq. (4)
  report: both equal `1 − Σ_d num_d / Σ_d w_d` with `w_d = SS_d`. Verified numerically, not assumed
  (`|pooled − variance-weighted| = 1.4e-13`). `r2_vw` is computed alongside as a cross-check; if the two
  ever diverge, there is a bug.
- **Do not** use a uniform mean over dimensions as a stand-in: on unequal-variance data it gave −0.32 where
  the correct value was +0.88.
- Report **`r2_vx`/`r2_vy`** per dimension (already in `metrics.py`).

**Stability** (NoMAD's currency, adopted verbatim)
- `snr := −10·log₁₀(1 − r2)`.
- Fit `y = A·e^(−Bt)` to the per-window (or per-bin) median SNR series.
- `half_life = ln2 / B`, reported in the same units as the x-axis.
- Report **`failure = r2 < 0`** as a **count** per cell and per session — NoMAD's own convention for
  "decoding failure".

**Error bars — and the CI decision**
- **Do not compute per-bin CIs.** The bins are massively autocorrelated (ρ₁ ≈ 0.9976 at `bin_ms = 20`),
  so a naive CI implies N_eff ≈ 9 where it claims 10,600.
- Instead adopt the two field conventions: **mean ± std across sessions** (FALCON) and **[Q1, Q3] across
  paired blocks + `scipy.stats.wilcoxon` signed-rank** (NoMAD).
- If a within-session interval is *ever* genuinely required, use a **block bootstrap at the integrated
  autocorrelation time as the block length** (~830 bins ≈ 16.6 s) — never an i.i.d. bootstrap.

**Presentation**
- **OR / ZS framing**: every results table prints the oracle (decoder refit on the eval block) next to the
  zero-shot static number. The grid already has `refit_decoders` for this — it is presentation, not new
  computation.
- The **ceiling register** (`r2_persist_lag1`, `r2_persist_lag12`, `r2_target`) is printed **once**, in a
  task-context block, never as a row in a decoder comparison table.
- ΔR² is always reported **relative to the cell's own `identity` (`frozen`) row**, not as an absolute
  number, so that session-to-session differences in difficulty cancel.

**Causality**
- All adapters and decoders stay **causal**: `mom_diag` uses fit-pool moments, not the data being decoded.
  `mom_diag_self` and `null_proj` are **two-pass / non-causal** and must be flagged as such wherever they
  appear. This matters because NoMAD explicitly adopted a causal sliding-window inference mode to simulate
  real iBCI use, and we should not silently buy accuracy with future data.

---

## 4. Claims we refuse to make

- **Not** "within-session drift is unstudied." NoMAD publishes within-session half-lives of **3.26 min
  (static) / 3.21 min (RTI) / 5.60 h (NoMAD) / 11.73 h (NoMAD + RTI)** on closed-loop human iBCI, and
  states alignment is "highly effective... within sessions... over timescales of minutes to hours."
- **Not** "nobody aligns latent spaces." Degenhart / Aligned FA / ADAN / CycleGAN / NoMAD all do; NoMAD
  reaches a 208.7-day half-life on isometric force.
- **Not** "our objective is wrong." FALCON's own movement baselines are a Wiener filter (ridge with
  history) and an LSTM, both trained on behaviour. NLB trains with Poisson NLL because its *object* is
  neural-rate prediction, not behaviour decoding.
- **Not** "the persistence floor is a baseline we beat." It is a property of the behaviour at `bin_ms`.
- **Not** "our per-unit normaliser is drift correction." `normalizer_perspective.md` already showed the
  unit-shuffled control cannot be distinguished — the gain is gain whitening.

**What we do claim.**
1. We bind within-session drift to an **elapsed-time axis** and plot the **decay curve** that the
   alignment literature only summarises with a post-hoc half-life.
2. We **decompose** that decay by **moment order** (P1–P4 above), giving a mechanistic account of *which
   class of alignment method can work and why* — which the field currently reports only as a ranking.
3. We **forecast / detect** the decay rather than only measuring it after the fact.
4. We report **skill scores** (`SS_free`, innovation R²) that separate *how much* is decoded from *how* —
   a register neither FALCON nor NoMAD uses.

## 5. The numbers we must sit next to

From FALCON (M1-A, 16-muscle EMG decoded from spikes) — mean ± std across sessions, held-out / held-in:

| Class | Method | M1-A R² |
|---|---|---|
| OR | Wiener Filter | 0.53 ± 0.04 / 0.54 |
| OR | RNN (LSTM) | 0.75 ± 0.05 / 0.75 |
| OR | NDT2 Multi | 0.78 ± 0.04 / 0.77 |
| **ZS** | **Wiener Filter (static)** | **0.34 ± 0.06 / 0.46** |
| FSU | CycleGAN + WF | 0.43 ± 0.04 / 0.61 |
| FSU | NoMAD + WF | 0.49 ± 0.03 / 0.64 |
| FSS | NDT2 Multi | 0.59 ± 0.07 / 0.77 |

FALCON Table 3 (WF trained **from scratch** on the ~1 min held-out calibration split): M1 **0.24 ± 0.04**,
M2 0.14 ± 0.05, H1 0.11 ± 0.03 — the field's canonical calibration-budget curve, and the shape
`staleness.py` should reproduce.

Our current zoo on `CO-20131003` (cursor velocity from spikes, Perich `000688` sub-C): ridge **0.357**,
`wiener` 0.404, `kf_posvel` 0.393 — the same range as FALCON's zero-shot WF. **Caveat to state once:**
FALCON M1-A decodes EMG, we decode kinematics; the comparison is indicative, not identical.

## 6. Open in this note

- `SS_free`'s exact reference free-run is **not yet pinned** (decoder's own recurred state vs a kinematic
  prior). Until it is, register 2 cannot be coded.
- `r2_recon` / innovation R² definitions are not yet written down to the standard of PLAN.md §5b.
- P1–P5 name `subspace` as the Aligned-FA analogue, but `subspace` is *unlabeled orthogonal subspace
  alignment*, whereas Aligned FA uses stable-loading **landmarks**. Whether these are close enough to
  treat as one family is an assumption, flagged.
- **P2 mixes label uses** — `subspace` is clean, `centroid_proc` is **gray**. Primary test is `subspace`.
- **P4 is not testable by the grid** (no time axis) → belongs to `staleness.py`. Flagged, not dropped.
- **Re-attribution needed:** the +0.015 R² in `normalizer_perspective.md` is the **two-pass**
  (`mom_diag_self`) variant, per its own documentation. When that finding is cited, the causality of the
  gain must be stated, or register 2 will overclaim what is deployable.
- The `Chewie_CO_2016` lineage question (`../drafts/18_...` §B.4 / Part E) is unresolved.

**Resolved after this note was written** (pre-run hardening, see `grid_within_session_27/README.md`
progress log 5): `r2_vw` turned out to be **identical to our existing pooled `r2_all`**, so register 1
needed *verification*, not new code; and the adapter-metadata columns, `gru` in `refit_decoders`,
all-five `active_decoders`, `r2_burnin_in/out` and `eval_contiguous` are now written by `run_grid.py`.

## 7. Provenance

Derived from: `2026-10-03_prior_art_falcon_nomad.md` (the paper read),
`../drafts/18_next_steps_before_after_paper_read.md` (the before/after plan), the four prior findings cited
in §2.1, `grid_within_session_27/config.json` (adapter list) and `adapters/feature.py` (adapter semantics).
Adapter parameter counts are the `n_params` reported by the adapters themselves, at d ≈ 100.
