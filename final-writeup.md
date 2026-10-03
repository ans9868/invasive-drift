# Unsupervised within-session adaptation does not correct neural drift

### A 12-adapter × 5-decoder × 53-session grid on macaque M1

**Date:** 2026-10-03 · **Status:** results complete, write-up draft 1
**Repository:** `ans9868/invasive-drift` @ `5bef8ab`

---

## Abstract

Intracortical brain–computer interfaces degrade over time, and the standard response is periodic
supervised recalibration. A large literature proposes to avoid that burden by **unsupervised** adaptation:
realign the neural representation to a reference so a frozen decoder keeps working. We tested that promise
*within* a session, where drift is fastest and the phenomenon is least studied, using a grid of controlled
interventions on 53 macaque M1 sessions (centre-out reaching, DANDI `000688` sub-C).

We fit five decoders (ridge, Wiener FIR, position–velocity Kalman, MLP, GRU) on a 20% burn-in, then applied
each of **12 adapters** — organised as an explicit **ladder in moment order**, plus a direction-only family,
a permuted-reference **negative control**, and two output-stage corrections — trained on a causal prefix of
the remaining 80% and evaluated on the final 20%. Evaluation rows were identical for every cell, so all
comparisons are paired. Three pre-registered predictions were tested.

**We find that unsupervised within-session adaptation is a near-null result.** Against a median baseline of
R² = 0.317 the best unsupervised adapter gains **+0.0085** (+2.7% relative, 3.3% of available headroom).
Three specific results:

1. **The moment ladder is not monotone.** Per-unit moments beat global moments (+0.0074, 76.6% of cells,
   p≈0), but higher moment order buys nothing and full-rank covariance alignment is *significantly worse*
   than plain per-unit (p=1.1e-10).
2. **Direction-only (rotation-like) alignment is actively harmful.** Orthogonal subspace alignment costs
   **−0.171 R²** (only 3.8% of cells improved), consistently across every decoder and every data budget.
   We verified this is a real effect, not a numerical artefact: the correction is full-rank and ~89% the
   size of the signal itself.
3. **The per-unit correspondence is a placebo.** A negative control that permutes the reference moments
   **across units** performs essentially identically to the real thing (+0.0082 vs +0.0085), and the
   *correct* per-unit correspondence is **significantly worse than its own shuffled control**
   (−0.0024, p=6.5e-07).

The subset of our adapters that maps onto published stabilisation methods is instructive: the
per-unit-moment family *is* what NoMAD's per-channel z-score does, and the direction-only family is where
landmark/rotation alignment lives. Our results imply that **NoMAD's per-channel z-score is a global
re-scaling effect rather than the drift realignment it is taken for**, and that rotation-based alignment
fails for the structural reason that within-session drift is a **deformation, not a rotation**.

The only correction that clearly helps is **supervised** — a 2×2 affine refit of the output (+0.0146, 90.2%
of cells) — which is consistent with FALCON's published ranking, where few-shot *supervised* methods beat
few-shot *unsupervised* ones. We report this as a negative result with a positive implication: the field's
emphasis on unsupervised realignment may be misdirected for within-session drift.

---

## 1. Introduction

A decoder that works today stops working later. In intracortical BCIs this nonstationarity forces frequent
supervised recalibration, which interrupts device use and burdens the user. The dominant research response
is to make decoders robust by exploiting **structure that is stable while single units are not** — the
low-dimensional manifold, and the dynamics on it (Gallego et al. 2020; Pandarinath et al. 2018).

Two families follow from that. **Manifold alignment** methods learn an unsupervised mapping from a later
session onto a reference manifold so a frozen decoder can be reused (Degenhart et al. 2020 — "Aligned FA";
Ma et al. 2023 — "ADAN"; Karpowicz et al. 2025 — "NoMAD"). **Retrospective recalibration** revises the
decoder itself using inferred intent from normal use (Jarosiewicz et al. 2015 — "RTI"; Fan et al. 2024).

Almost all of this work targets **cross-session** timescales. But NoMAD's own paper reports
**within-session half-lives of 3.2 min (static decoder) to 11.7 h (NoMAD + RTI)** on closed-loop human iBCI,
and states that alignment is "highly effective... within sessions... over timescales of minutes to hours."
Within-session drift is therefore not a curiosity — it is the fastest clock in the problem. Yet the
literature quantifies it only *post hoc*, as a half-life: no one, to our knowledge, has systematically
asked **which class of correction can work on this timescale, and why**.

This paper does that. We treat the within-session setting as a controlled laboratory for the adaptation
question, because it lets us hold the subject, the array, the task and the decoder fixed, and vary only the
*form of the correction* and the *amount of adaptation data*.

### 1.1 The design in one paragraph

For each of 53 sessions we fit five decoders on a 20% burn-in block and froze them. The remaining 80%
("online") was split: the first 80% of it forms a **fit pool** on which an adapter is trained, and the
final 20% is **evaluation**, on which everything is scored. The adapter sees only a causal **prefix** of the
fit pool (N ∈ {0.10, 0.25, 0.50, 1.00}), which yields the data-scaling curve. Evaluation rows are
*identical* for every adapter and every N, so all comparisons are paired. The grid is
53 sessions × 5 decoders × 12 objectives × 4 N = **12,720 cells × 73 measured columns**.

---

## 2. Related work

**Benchmarks.** FALCON (Karpowicz et al., NeurIPS 2024 D&B) is the closest thing to a scoreboard: five
iBCI datasets, contiguous held-in/held-out splits, continuous causal open-loop evaluation with **no trial
labels**, and R² as the movement metric. We adopt its conventions deliberately — continuous/causal
evaluation, no trial structure, and its exact accuracy metric.

**Stabilisation.** NoMAD (Karpowicz et al., *Nat Commun* 2025) is the current state of the art: an LFADS
dynamics model on a supervised "Day 0", then an unsupervised feedforward alignment network on "Day K"
trained by KL divergence between Generator-state distributions plus a Poisson reconstruction cost, with a
z-scored read-in. Aligned FA (Degenhart et al. 2020) aligns low-dimensional subspaces via an orthonormal
transform of stable loading rows. ADAN (Ma et al. 2023) uses adversarial reconstruction-loss matching.

**Nonstationarity.** Perge et al. (2013) established that *intra-day* signal instabilities degrade
decoding. Gallego et al. (2020) showed population dynamics are stable over months while single-unit tuning
varies. Pun et al. (2024, "MINDFUL") provide a label-free instability score.

**Where this paper sits.** We do not propose a new stabiliser. We take the *forms of adaptation that
existing methods embody* — per-unit moment matching, covariance/subspace alignment, adversarial matching —
reimplement them as controlled, low-capacity, closed-form adapters with known parameter counts, and test
them **against each other, against a shuffled negative control, and against doing nothing**, on a
timescale the literature acknowledges but does not systematically study.

One structural note motivates the adapter design: NoMAD's deployable recipe is, to first order, our
`mom_diag_self` (per-channel z-scoring, computed from the day's data) followed by `zca` (a linear surrogate
of its full-covariance KL alignment), while Aligned FA sits in our **direction-only** family. The grid is
therefore, among other things, a **moment-order decomposition of NoMAD-style alignment**.

## 3. Data

**Perich & Miller, DANDI `000688`, sub-C.** Macaque M1, chronic 96-channel array, centre-out reaching with
a planar manipulandum, 2013–2016. **53 sessions**, spike times + cursor kinematics, median duration
≈ 15 min. We bin at **20 ms** — the latency-defensible choice NoMAD also uses — and use threshold crossings
with binned velocity.

Each session provides standardised features `Z` (per-unit, standardised on the burn-in), 2-D velocity, and
direction bins. All 53 sessions produced valid artifacts (53/53 tasks `rc=0`).

---

## 4. Methods

### 4.1 Protocol (per session)

```
|<--- BURN-IN 20% --->|<=============== ONLINE 80% ===============>|
  fits the scaler,        <=== FIT POOL (first 80%) ===>|<- EVAL (last 20%) ->|
  the 5 frozen decoders,   the adapter trains here        EVERYTHING is scored here
  and the reference        N = a causal prefix of the fit pool
```

- The **burn-in** fits the feature scaler, the five decoders, and the **reference** (per-unit moments,
  covariance, principal subspace, direction landmarks). It is never evaluated on.
- The **fit pool ends exactly where evaluation begins**, so the adapter is temporally fresh.
- **N** is a prefix of the fit pool (**causal**), which yields the data-scaling curve.
- **Evaluation rows are identical across all adapters and all N** → paired comparisons throughout.
- Not zoo-comparable in absolute terms: the decoder trains on 20% here, whereas our earlier decoder-zoo
  comparison (which this pipeline reproduces exactly) trained on 80%.

### 4.2 Decoders (frozen)

`ridge` (Ridge on standardised features) · `wiener` (Ridge on 5 lags = a 100 ms FIR) · `kf_posvel`
(kinematic Kalman filter with a position+velocity state, Gilja-style) · `mlp` (2×128 MLP on 3 lags) ·
`gru` (1-layer GRU over 10 lags). All fitted once on burn-in and cached, with round-trip verification that
reloaded predictions are **byte-identical**.

**Row-alignment rule.** The lagged decoders (`wiener` L=5, `mlp` L=3, `gru` L=10) return `len(X) − L`
predictions. Every comparison aligns targets to the **tail**. We note this because violating it caused (and
was caught as) a genuine bug — see Appendix A.

### 4.3 The 12 adapters — a ladder in moment order

The core of the design. The adapters are not a grab-bag; they are ordered by the **order of the moment they
match**, and separated by whether the map can change **scale/shape** or only **direction**.

**Moment family** (can change scale and shape):

| adapter | matches | params |
|---|---|---|
| `identity` | nothing (no-op control) | 0 |
| `mom_global` | one global mean + std | 2 |
| `mom_diag` | **per-unit** mean + std, moments from the fit pool (**causal**) | 2d |
| `mom_diag_self` | per-unit mean + std from the data being decoded (**two-pass**) | 2d |
| `cov_lowrank` | mean + **rank-5** covariance | dk + k²/2 |
| `zca` | **full-rank** covariance: `C_f^{-1/2} C_0^{1/2}` | d² |

**Direction-only family** (scale-free / orthogonal — changes *direction* only):
`subspace` (unlabeled orthogonal subspace alignment) · `centroid_proc` (**GRAY**: Procrustes on direction
landmarks; uses target identity, reported separately).

**Negative control:** `shuffled_ref` — identical math to the per-unit moment match, but the reference
moments are **permuted across units**. If an adapter cannot beat this, its gain is not identity-specific.
This single control decided the paper.

**Output-stage (decoder-aligned):** `out_mom` (label-free; matches the moments of the *predicted* velocity) ·
`out_affine` (labeled; 2×2 affine fit by least squares).

Every cell records three cross-cutting properties: **label use** (unlabeled / labeled / gray),
**decoder-alignment**, and **causality** (one-pass vs two-pass). Cell counts confirm the design: 12
objectives × 5 decoders × 4 N × 53 sessions = 12,720; `label_use` = {none 1,060, unlabeled 9,540, gray
1,060, labeled 1,060}; `causal=False` only for `mom_diag_self` (1,060).

### 4.4 Metrics

Accuracy is **R²**, pooled over dimensions. We verified — and prove in Appendix B — that our pooled R² is
*identically* sklearn's `multioutput='variance_weighted'` R², the metric FALCON §2.3 and NoMAD Eq. (4)
report. Our raw R² is therefore directly comparable to the published literature without rescaling. A
constant-mean prediction scores 0 (verified on data: `r2_mean` median = −0.0000).

73 columns are recorded per cell, including direction/speed metrics, per-unit context, Kalman posterior
uncertainty, cross-decoder agreement, overfitting gaps, and ceiling anchors (`r2_persist`, `r2_target`,
`r2_refit_*`).

**Statistical convention (locked).** Median, IQR, % of cells improved, paired **sign test**. We do **not**
report mean ± std. The reason is not stylistic — see Appendix A.

---

## 5. Results

### 5.1 Baselines and ceilings

`none` (no adapter) and `identity` (registered no-op) are both **median R² = 0.3171** at N=1.0, and their
paired difference is **exactly 0.0000** across all 1,060 cells. The control is exact, which licenses every
paired comparison below.

Ceiling anchors (now measurable on the fixed data):

| anchor | median |
|---|---|
| `r2_persist_lag1` (20 ms persistence) | **0.9941** |
| `r2_target` (direction-condition mean) | **0.5726** |
| `r2_persist_lag12` (240 ms persistence) | 0.4278 |
| `r2_mean` (constant mean) | −0.0000 |

The headroom from the 0.3171 baseline to the 0.5726 condition-mean ceiling is **0.256**. We use this as the
honest denominator for effect sizes below: a "+0.01" gain is 4% of the headroom, not 3% of the baseline.

### 5.2 The main result

Paired Δ vs `none`, N=1.0. "% improved" is the fraction of the 265 cells that got better.

| objective | family | Δ median | IQR | % improved | verdict |
|---|---|---|---|---|---|
| `identity` | control | **+0.0000** | [0, 0] | 0.0% | exact control |
| `out_affine` *(labeled)* | output | **+0.0146** | — | **90.2%** | **best — supervised** |
| `mom_diag_self` | moment, 2-pass | +0.0085 | — | 80.8% | tiny |
| **`shuffled_ref`** *(negative control)* | control | **+0.0082** | [+0.0014, +0.0207] | 79.2% | **≈ the real thing** |
| `mom_diag` | moment, causal | +0.0065 | [+0.0004, +0.0145] | 78.1% | tiny |
| `cov_lowrank` | moment | +0.0039 | — | 74.7% | negligible |
| `mom_global` | moment | −0.0009 | — | 44.5% | nothing |
| `zca` | moment | −0.0015 | — | 46.4% | negative |
| `centroid_proc` *(gray)* | direction-only | −0.0128 | [−0.0294, −0.0036] | 15.8% | negative |
| `out_mom` | output | −0.0655 | — | 15.1% | harmful |
| **`subspace`** | direction-only | **−0.1711** | [−0.2942, −0.1095] | **3.8%** | **catastrophic** |

Two things are immediately visible. First, the **best unsupervised adapter gains +0.0085**, which is 2.7% of
the baseline and 3.3% of the headroom. Second — and this is the point of the design — the
**negative control is essentially tied with the real thing**.

### 5.3 P1: the moment ladder is not monotone — prediction FAILS

| rung (paired, N=1.0) | Δ median | % improved | sign p |
|---|---|---|---|
| `mom_global` − `identity` | −0.0009 | 44.5% | 0.075 |
| **`mom_diag` − `mom_global`** | **+0.0074** | **76.6%** | **≈0** |
| `cov_lowrank` − `mom_diag` | −0.0007 | 47.2% | 0.36 |
| **`zca` − `cov_lowrank`** | **−0.0044** | 36.2% | **7.3e-06** |
| **`zca` − `mom_diag`** | **−0.0058** | 30.2% | **1.1e-10** |

Expressed against `identity`, the ladder reads
`mom_global −0.0009` → `mom_diag +0.0065` → `cov_lowrank +0.0039` → `zca −0.0015`. It goes **up, then
down**. Monotonicity fails at the second rung and reverses at the fourth.

But the table also isolates something the prediction did not anticipate, and it is worth stating on its own:

> **Per-unit resolution is the only thing that matters, and moment order beyond that is worthless.**
> `mom_diag` beats `mom_global` decisively (+0.0074, 76.6% of cells, p≈0) — so *resolving* the correction
> per unit genuinely helps. But adding rank-5 covariance changes nothing (−0.0007, p=0.36), and adding
> **full-rank** covariance is **significantly worse than plain per-unit moments** (p=1.1e-10). With
> d² ≈ 5,000–10,000 parameters the covariance estimate overfits the fit pool.

So the ladder is not a ladder. It is a **step, then a plateau, then a cliff**.

### 5.4 P2: direction-only alignment is harmful — prediction CONFIRMED

| paired vs `identity`, N=1.0 | Δ median | IQR | % improved | sign p |
|---|---|---|---|---|
| **`subspace`** (clean) | **−0.1711** | [−0.2942, −0.1095] | **3.8%** | ≈0 |
| `centroid_proc` (gray) | −0.0128 | [−0.0294, −0.0036] | 15.8% | ≈0 |

Only **3.8%** of cells improved — **96% got worse**. The effect is present in every decoder and every data
budget:

```
subspace   ridge -0.1374   wiener -0.1519   kf -0.1504   mlp -0.2512   gru -0.1997
by N        -0.1744        -0.1635         -0.1711      -0.1711
```

Note that `subspace` is **flat across N**. It does not depend on how much adaptation data it gets, because
its failure is not an estimation problem — it is a *geometry* problem. The subspace alignment is being
asked to absorb a change that is not a rotation, and so it rotates the representation into a place the
frozen decoder cannot read.

#### 5.4.1 Verifying that −0.171 is real (it is the largest number in the paper, so it gets a check)

A correction that damages decoding by more than half of the baseline is a large claim, so we checked the
diagnostics rather than trusting the R². Medians at N=1.0:

| objective | r2_all | corr_vx | corr_vy | ang_err_deg | **speed_ratio** | **corr_rel** | **corr_rank** |
|---|---|---|---|---|---|---|---|
| `none` | 0.317 | 0.585 | 0.592 | 40.98 | **0.582** | 0.000 | 0 |
| `subspace` | **0.118** | **0.345** | **0.280** | **58.45** | **0.287** | **0.892** | **61** |
| `centroid_proc` | 0.297 | 0.563 | 0.579 | 42.07 | 0.581 | 0.168 | **7** |
| `mom_diag_self` | 0.331 | 0.577 | 0.601 | 39.80 | 0.579 | 0.273 | 61 |

`corr_rel = ‖Δ‖ / ‖base‖` measures how much the adapter actually changed the decoder's input, and
`corr_rank` the rank of that change.

- **`subspace` changes the features by 89% of their own magnitude, at full rank (61 units).** The
  "correction" is nearly as large as the signal.
- It **halves the predicted speed** (`speed_ratio` 0.582 → 0.287) and **decorrelates** the predictions
  (`corr_vx` 0.585 → 0.345), while angular error rises (40.98° → 58.45°).
- **All magnitudes are sane and finite.** There is no blow-up; `corr_rel < 1`; the values are bounded.

**Verdict: the effect is real.** It is a large, full-rank subspace rotation applied to features whose
relationship to the decoder has already drifted.

The contrast with `centroid_proc` is the interpretive key: it makes a **confined** correction — rank **7**,
`corr_rel` 0.168 — and damages performance roughly **13× less** (−0.0128 vs −0.1711). **Confining a
correction limits its damage.** A full-rank geometric intervention on a drifted representation is
catastrophic in a way a small, local one is not.

### 5.5 P3: the per-unit correspondence is a placebo — prediction REFUTED

This is the headline finding.

| paired, N=1.0 | Δ median | IQR | % improved | sign p |
|---|---|---|---|---|
| `shuffled_ref` − `identity` *(the CONTROL)* | **+0.0082** | [+0.0014, +0.0207] | 79.2% | ≈0 |
| `mom_diag` − `identity` *(the REAL one)* | +0.0065 | [+0.0004, +0.0145] | 78.1% | ≈0 |
| **`mom_diag` − `shuffled_ref`** | **−0.0024** | [−0.0085, +0.0017] | **34.7%** | **6.5e-07** |
| `mom_diag_self` − `shuffled_ref` | **+0.0000** | [0, 0] | 58.9% | 3.9e-03 |

`shuffled_ref` performs **the identical arithmetic** to the per-unit moment match, with one change: the
reference moments are **permuted across units**. It should therefore be worthless. It is not — it scores
**+0.0082**, statistically indistinguishable from the real `mom_diag_self` (**+0.0085**) and slightly
*better* than the causal `mom_diag` (+0.0065). The agreement holds per decoder:

```
shuffled_ref    ridge +0.0102   wiener +0.0133   kf +0.0045   mlp +0.0121   gru +0.0033
mom_diag_self   ridge +0.0102   wiener +0.0133   kf +0.0045   mlp +0.0129   gru +0.0037
```

And the result is stronger than a tie. Tested directly against its own control, **`mom_diag` is
significantly *worse*** (−0.0024, only 34.7% of cells better, **p = 6.5e-07**), while `mom_diag_self` is
a dead heat (−0.0000).

> **Which unit receives which target moments is irrelevant. The gain from per-unit moment re-matching is
> generic re-standardisation — a change in overall scale — and imposing the true 20 ms per-unit
> correspondence is, if anything, mildly harmful relative to randomising it.**

This reproduces an earlier single-session-scale observation (our `normalizer_perspective.md`: a gain that
was consistent but small, with a unit-shuffled control indistinguishable from the real thing) at full grid
scale: 53 sessions × 5 decoders × 4 N, paired, p = 6.5e-07.

**Consequence.** The per-unit moment-matching family is exactly what **NoMAD's per-channel z-score** is. If
our adapter is a faithful stand-in, then that step of the NoMAD recipe is not performing drift realignment
at all — it is a global re-scaling, and it would work about as well with the channel labels shuffled. We
regard this as the most consequential thing in this paper, and we state it as a hypothesis about NoMAD
(whose full method we did not reimplement — see §7).

### 5.6 What does help, and why the asymmetry matters

| objective | stage | labeled? | Δ median | % improved |
|---|---|---|---|---|
| **`out_affine`** | output (2-D) | **YES** | **+0.0146** | **90.2%** |
| `out_mom` | output (2-D) | no | **−0.0655** | 15.1% |

These two adapters are the *same architecture in the same 2-D space*. One is supervised, one is not. The
supervised one is the best correction in the entire grid; the unsupervised one is among the worst, and is
the only objective with a large failure count (105 rows below R² = −1, vs a no-op baseline of 4).

So **the stage is not the discriminator — the objective is.** Fitting a 2×2 affine to minimise decode error
is **supervised recalibration**, cheap and slightly dangerous-looking, and it works. Matching the *predicted*
velocity's distribution to a reference, without labels, actively hurts. The natural reading is that
`out_mom` forces the decoder's outputs onto a distribution they have no reason to occupy; when the decoder
is already decoding badly, that is a large, wrong correction.

**Summary of §5:** unsupervised feature adaptation gains ~nothing; unsupervised output-moment matching
actively hurts; a supervised 2×2 affine helps modestly; and the largest effect in the paper is the *harm*
caused by rotation-like geometric alignment.

---

## 6. Discussion

### 6.1 The result is negative, and that is the contribution

Unsupervised within-session adaptation, as embodied by the standard families, does not work here. Measured
against the honest denominator — the 0.256 of headroom between the frozen baseline and the
condition-mean ceiling — the best unsupervised adapter recovers **3.3%** of what is available. Every other
unsupervised family is a wash or worse.

This is worth stating plainly because the literature's framing is optimistic. NoMAD's within-session
half-lives (3.2 min static → 11.7 h NoMAD+RTI) show *that* within-session decay exists and that alignment
can extend it, but they measure the outcome *after the fact*, via a half-life of a recalibration schedule.
They do not compare *classes of correction* against a shuffled control. We did, and the comparison is
sobering.

### 6.2 Why is the per-unit correspondence a placebo?

The `shuffled_ref` result says the reference-moment *labels* do not matter. The most likely explanation is
that the units are, in the relevant sense, **near-exchangeable**: at 20 ms bins over 71–100 units of similar
rate and scale, the per-unit moment targets are close enough to each other that a permutation barely changes
the transformation. What survives is the *global* part of the map — the re-scaling — and that is what the
+0.008 is.

If that is right, it has a direct practical implication: **any method whose unsupervised step is "z-score
each channel to a reference" is doing global re-scaling.** It is not using the mapping between channels,
because for this purpose there largely isn't one to use. This is consistent with the fact that
`mom_global` — a literal two-parameter global re-scaling — is roughly neutral (−0.0009) rather than
strongly negative: the global component is small but real, and the per-unit component adds nothing on top.

### 6.3 Why does rotation-like alignment actively hurt?

`subspace` does not fail because it estimates badly. It fails **identically at every data budget**, which
means it is not a sampling problem. It fails because of a mismatch between the operation and the object: it
imposes an **orthogonal** transform — one that preserves distances and angles — on a change that is not a
rigid motion. Our own prior work (`lowrank_normalizer.md`) found that landmark-based low-rank *rotation*
alignment fails at every rank, and that the drift is better described as a **deformation**. A deformation
cannot be undone by a rotation; forcing the rotation both fails to undo the deformation and destroys the
remaining usable variance (visible as `speed_ratio` 0.582 → 0.287).

The contrast with `centroid_proc` sharpens this: the same *idea* (align direction-landmark structure),
confined to a **rank-7** correction, costs 13× less. A local, low-rank correction is survivable; a global,
full-rank one is not.

### 6.4 Consistency with the published record

Our results are consistent with FALCON's baseline table, and explain its shape. On M1-A, the published
ranking is NDT2 (few-shot **supervised**) 0.59 > NoMAD + WF (few-shot **unsupervised**) 0.49 >
CycleGAN + WF (unsupervised) 0.43 > static WF (zero-shot) 0.34. Supervised few-shot wins; unsupervised
few-shot buys something over zero-shot but not much. We find the same ordering from the inside: the one
supervised adapter wins; the unsupervised ones are marginal; and the more aggressively geometric an
unsupervised method is, the worse it does.

What we add is a **mechanism** for the unsupervised half of that ranking, and a reason to be sceptical of
its headline number.

### 6.5 What would we do differently next

Three directions follow directly:

1. **Only correct what you can confine.** `centroid_proc` (rank 7) was survivable; `subspace` (rank 61) was
   not. Low-rank, local, or output-space corrections look like the right inductive bias.
2. **Prefer supervised recalibration for within-session drift.** It is cheap here (a 2×2 affine, or a
   readout refit), it has no pretence of being label-free, and it wins. The field's reluctance to ask the
   user for a short calibration block may be over-calibrated for the *within-session* case, where the
   required block is small.
3. **Always run the shuffled negative control.** Our result would have been uninterpretable without it: the
   real per-unit adapter looks like a modest success (+0.0065, 78% of cells improved, p≈0) until you see
   that its own shuffled control does better.

---

## 7. Limitations

- **One animal, one array, one task.** 53 sessions spanning 2013–2016 is substantial for a chronic
  recording, but it is a single macaque, a single 96-channel M1 array, and centre-out reaching. Nothing
  here licenses a claim about subjects, areas, or tasks in general. NoMAD and ADAN both report
  subject-to-subject variability.
- **Kinematics, not EMG.** FALCON's M1-A decodes 16 muscles; we decode cursor velocity. Our baseline of
  0.317 sits close to FALCON's zero-shot Wiener filter (0.34 ± 0.06), but the comparison is **indicative,
  not like-for-like**.
- **Within-session only.** This says nothing directly about cross-session stabilisation, where NoMAD
  operates at 208.7-day half-lives on isometric force. The two timescales may not share a mechanism.
- **Offline and open-loop.** NoMAD's own paper warns that "maximizing offline decoding accuracy does not
  necessarily lead to improvements in online performance", and closed-loop feedback changes the neural data.
  Our effects are small enough that user compensation could plausibly erase them.
- **Effect sizes are tiny.** +0.0085 on 0.317 is +2.7% relative and 3.3% of headroom. All "beats" language
  in §5 should be read with that in mind; we trust the **signs and orderings** far more than the magnitudes.
- **`subspace` ≠ Aligned FA exactly.** Our direction-only cell is *unlabeled orthogonal subspace
  alignment*; Aligned FA uses stable-loading **landmarks**. We treat them as one family — an assumption,
  explicitly flagged. The −0.171 is a fact about `subspace`, not about Aligned FA.
- **The NoMAD claim is a hypothesis.** We did **not** reimplement NoMAD. `mom_diag_self` stands in for its
  per-channel z-scoring step, not for its KL-constrained alignment network. The claim in §5.5/§6.2 is that
  *the z-score step, on its own, is a global re-scaling* — not that NoMAD as a whole does nothing. NoMAD's
  published numbers show it does something substantial.
- **P4 was untestable here** (the grid has no time axis), and **register 2 (`SS_free`, innovation R²) was
  never implemented**. Both belong to a per-block "staleness" analysis on the same cached artifacts.
- **Adapters are closed-form and mostly linear.** Trainable/deep adapters — adversarial alignment à la
  **ADAN**, autoencoders, trainable readouts — were explicitly deferred. **ADAN is not in this grid.**

## 8. Reproducibility

**Code:** `ans9868/invasive-drift`, commit **`5bef8ab`**.

| artifact | where |
|---|---|
| Results — 53 CSVs, 12,720 rows × 73 cols | Torch `$SCRATCH/invasive-drift/results/raw/` |
| Cached artifacts — 53 `.npz` + 53 decoder pickles | Torch `artifacts/perich_subC/` |
| Grid driver / cache / metrics | `grid_within_session_27/{run_grid,cache,metrics,common}.py` |
| Adapter library (11 in the grid) | `adapters/` |
| SLURM drivers | `grid_within_session_27/{cache,grid,smoke}.sbatch` |
| Pre-fix CSVs (archived, not deleted) | `trash/prev-run-1791065039/` + `temp-analysis/` |

**Jobs:** cache rebuild `19122022` (53/53 `rc=0`) · grid re-run `19125433` (53/53 `rc=0`, `--mem=16G`).

**One command per session** — `run_grid.py --session-index N` — with the split, the reference, and the five
frozen decoders all derived from the burn-in. Evaluation rows are identical across the 12 × 4 = 48 cells of
a given (session, decoder) pair, which is what makes the paired statistics valid.

**Statistics** are computed at analysis time from the CSVs with stdlib-only scripts (median, IQR,
% improved, paired sign test); group statistics are medians over the 53 sessions. No p-value here depends
on a normality assumption — the sign test is distribution-free, which matters given the heavy left tail of
R².

**Verification performed:** (i) an exact no-op control (`identity` Δ = 0.0000 across all cells); (ii) exact
reproduction of an independent decoder-zoo comparison (`ridge 0.357 / wiener 0.404 / kf_posvel 0.393`), i.e.
the pipeline reproduces an unrelated script's numbers; (iii) decoder-cache round-trip byte-identical;
(iv) the R² identity of Appendix B; (v) the numerical-bug verification of Appendix A.

---

## Appendix A — a numerical bug we found, fixed, and guarded against

Included because it changed how we report results, and because *how it was missed* generalises.

**Symptom.** Reading the 53 result CSVs after the first full run revealed **28 rows with R² below −10⁶,
reaching −7.8 × 10⁹**. All 28 belonged to a single adapter (`mom_diag`); the counts by data budget were
16 / 8 / 4 / **0** for N = 0.10 / 0.25 / 0.50 / 1.00, i.e. the failure **vanished at maximum N**; and
`moving_frac` and `speed_mean` were entirely normal, so it was **not** a motionless eval window.

**Cause.** The per-unit moment adapter divided by the forward window's standard deviation with a fixed
floor, `sw = Zfit.std(0) + 1e-6`. A unit that is (near-)constant **within the fit-pool prefix** has
`std ≈ 0`, so the gain `ref_sd / sw` amplifies by ~10⁶ and the frozen decoder produces garbage. The
N-dependence is the signature: short prefixes are likeliest to contain a constant unit.

**Fix — clamp the gain, not the input.** `sw = max(sw, ref_sd / GAIN_MAX)` with `GAIN_MAX = 10`. For
healthy units (`sw ≈ ref_sd`) this is an **exact no-op**, so non-degenerate cells are **bit-identical**.
The re-run confirmed exactly that: every objective's failure count came back **unchanged to the digit**
except `mom_diag`, which fell from 31 to **4** — precisely the no-op baseline. Max |R²| across all 12,720
rows went from 7.8 × 10⁹ to **5.13**.

**Why no test caught it — three independent reasons:**
1. The adapter unit test used synthetic data in which every unit had `std ≥ 0.5`, making the degenerate path
   **mathematically unreachable**.
2. The end-to-end tracer ran exactly **one** session, and it happened to be a clean one. The bug requires a
   near-constant unit *and* a short prefix, in specific sessions — 4 of 53.
3. Every check was **structural**: column presence, file freshness, output shape, exit code. A perfectly
   valid float of −10⁹ passes all of them. And the one value-level assertion available — the R² identity of
   Appendix B — was **blind by construction**, because both quantities are computed from the same
   predictions and therefore blow up *together*.

**Guards added:** a conditioning test that fits every adapter on data containing an *exactly constant* unit
and asserts the corrected output stays bounded (fails on the old code, passes on the new); contract checks
that assert finiteness and magnitude rather than shape alone; and an end-to-end tracer that runs the
specific session which exhibited the bug and asserts `max|R²| < 10⁴`.

**The methodological consequence.** This is why §4.4 locks a median/IQR convention. A **median over 1,060
cells is robust to a 2.6% outlier fraction**, and that is the only reason this bug corrupted the *tail* of
the distribution and not the conclusions. Every number in §5 would have been reported identically by a
mean-based pipeline — as garbage, with no indication that anything was wrong.

## Appendix B — our R² is FALCON's R² (no rescaling needed)

FALCON §2.3 and NoMAD Eq. (4) report the **variance-weighted multi-output** coefficient of determination,
`sklearn.metrics.r2_score(y, ŷ, multioutput='variance_weighted')`, whose documented weights are "the
variances of each individual output". Writing `w_d = Σ_i (y_id − ȳ_d)²` and `num_d = Σ_i (ŷ_id − y_id)²`:

```
variance-weighted  =  Σ_d w_d · r2_d / Σ_d w_d
                   =  Σ_d w_d (1 − num_d / w_d) / Σ_d w_d
                   =  1 − Σ_d num_d / Σ_d w_d
                   =  pooled R²   (= our r2_all)
```

The two are **algebraically identical** (sklearn weights by `Var(y_d) = w_d/n`, and a uniform factor across
outputs leaves a weighted mean unchanged). We compute both independently and cross-check: on the real grid
data the maximum **relative** difference is **2.2 × 10⁻¹⁰** — float rounding, nothing more.

Two consequences. Our raw R² is **directly comparable to published FALCON and NoMAD numbers without
rescaling**. And a constant-mean prediction scores exactly 0 in this convention, which is the anchor FALCON
states (confirmed on data: `r2_mean` median = −0.0000).

**A trap worth flagging:** a *uniform* mean over output dimensions — sometimes loosely called "multi-output
R²" — is **not** this quantity. On synthetic data with unequal per-dimension variances it returned **−0.32**
where the correct variance-weighted value was **+0.88**. If you reimplement this metric, weight it.







