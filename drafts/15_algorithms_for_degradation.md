# 15 — Algorithms for detecting/forecasting degradation (from finance, physics, stats)

Companion to `drafts/14` (target design). Our signal: **`h(t)` = decoder R² per ~2–5 min block** —
**noisy, non-stationary, short (~4–30 pts/session), often drifting down, sometimes stepping**.
Goal: level / **rate** / **acceleration** / change-points / early-warning / graded tiers.

## Finance / econometrics
| method | outputs | short-series? | note |
|---|---|---|---|
| **Drawdown / max drawdown** | graded severity (depth below running peak) | ✅ | simplest interpretable "degradation"; gives tiers directly |
| **CUSUM** | change-point / drift onset | ✅ | classic sequential mean-shift detector; cheap |
| **BOCPD** (Adams & MacKay 2007) | posterior change points / run-length | ✅ | online, noise-robust, short-series friendly |
| **Markov-switching / HMM** (Hamilton 1989) | regime probs (healthy/degrading) | ~ | needs ≥~20 pts |
| **GARCH / stochastic vol** | time-varying variance | ❌ | for "rising-variance" EWS; needs long |
| **Hurst / DFA** | long-memory/persistence | ❌ | needs long series |

## Physics / early-warning
| method | outputs | short-series? | note |
|---|---|---|---|
| **Early-warning indicators** (Scheffer 2009, Nature 461:53) | imminent-tipping flag | ~ | **rising variance + lag-1 autocorrelation** (+ skewness, flickering) before a critical transition |
| **Flickering** (Wang 2012, Nature 491:399) | bistability/pre-transition | ~ | |
| **RQA / Lyapunov** | determinism/chaos | ❌ | need long, clean series |

## Statistics / signal processing / ML
| method | outputs | short-series? | note |
|---|---|---|---|
| **PELT / BinSeg / Wild Binary Seg** (`ruptures`, Truong 2020) | offline change points | ✅ | multi-change-point; penalty-tuned |
| **Page–Hinkley / SPRT** | online drift onset | ✅ | complements CUSUM |
| **ADWIN / DDM / McDiarmid drift** | streaming regime change | ✅ | concept-drift detectors |
| **Kalman const-accel state-space** | level/rate/accel + uncertainty | ✅ | *what we use now* (`10`) |
| **GP regression** | smooth trend + uncertainty | ✅ | principled CIs |
| **Hawkes process** | self-exciting crashes (clustering) | ❌ | if drops cluster |
| **Survival/hazard model** | time-to-next-drop | ✅ | graded risk over horizon |

## Ranked shortlist (for short, noisy, irregular series)
1. **Kalman constant-acceleration state-space** (keep) — level/rate/accel w/ CI. Baseline.
2. **BOCPD** — change points + regime, online, robust; pairs with the tier idea.
3. **EWS (rising variance + lag-1 autocorrelation in a window)** — the *early-warning* arm;
   test on pre-drop windows.
4. **CUSUM / Page–Hinkley** — cheap online onset detection.
5. **Drawdown** — simple graded severity; natural tier definition.

## Validation (avoid the "it always goes down" bias)
- Same **ladder** (chance → persistence → trend → panel), **detrended** targets (`drafts/13`).
- For EWS: compare indicator rise in **true pre-drop windows** vs **label-shuffled** windows.
- Report **Δ over trend**, CIs, session-out.

## ADDENDUM — expert review (adopt): short-series realism

A sharp external review (pasted 2026) corrected our candidate list and the pipeline. Key points:

**Model each block as** `h(t_i) = μ(t_i) + ε_i`, `ε_i ~ N(0, σ_i²)`, with **σ_i the block's own R²
sampling variance** (bootstrap over trials, or Fisher information). **Pass σ_i as KNOWN observation
noise** — estimating it from ~10 points throws away the one bit of real information. *(We had used fixed
noise in `10` — fix this.)*

**DROP (not identifiable at n ≈ 4–30):** GARCH/stochastic-vol, Hurst/DFA, Lyapunov, RQA,
**critical slowing down** (rising variance / lag-1 autocorrelation — usually a dying channel, not a
bifurcation), Hawkes, cointegration, free HMM / hidden semi-Markov >2 states.

**KEEP (ranked):**
1. **Continuous-time constant-acceleration Kalman/RTS smoother**, with known σ_i → level, rate,
   acceleration **with posterior variance**; irregular Δt native; predictive distribution for warnings.
2. **Matérn-5/2 GP** on the same state (twice-differentiable), lengthscale **fixed from the block grid**
   (not MLE below ~15 pts); derivative uncertainties. Cross-check vs (1).
3. **Drawdown on the *smoothed* level**, tiers = **quantiles vs a flat-null simulation** (simulate ε under
   flat μ) — *not* cutoffs on raw R².
4. **BOCPD** (expected run ≈ 8–15) or **offline PELT** (min segment 4) as a **separate regime/step layer**
   — keep apart from slope so a step ≠ acceleration.
5. **Predictive floor-crossing probability** (1–2 blocks ahead) — the EWS that is actually identified.
- Page–Hinkley/CUSUM = cheap alarm beside (4); SPRT = pre-registered "degraded" call. Particle filter only
  to model jumps (Student-t obs) once the Gaussian smears known channel deaths.

**Validation (don't fool yourself):**
- **Walk-forward only**; never score the smoothed fit.
- Baselines in order: **persistence → historical-median-slope → "no further change"**; skill =
  `1 − MSE/MSE_base`. (Our "trend = persistence + mean-Δ" is close; the historical-slope extrapolation is
  stronger — adopt it.)
- **Injection tests + coverage** (plant known slope/accel/step; check the posterior interval covers it),
  not in-sample correlation.
- **Acceleration rarely earns its keep at n<20** → require it to beat a **local-linear (level+slope)**
  model on held-out floor-crossing before quoting it. *(Our preliminary Δaccel=+0.25 is therefore
  UNCONFIRMED.)*
- Change-point skill needs **labels or injections**; in-sample PELT breaks ≠ evidence.
- **Partial-pool the slope across days** (hierarchical) — a per-session MLE on ~8 pts rediscovers noise.

**Refs/code:** Durbin & Koopman ch.3; `rlabbe/filterpy` + Kalman-book; `andgoldschmidt/derivative`;
sklearn `Matern(nu=2.5)`; Adams & MacKay 2007 (`hildensia/bayesian_changepoint_detection`);
`deepcharles/ruptures`; `online-ml/river`; Magdon-Ismail & Atiya 2004; Solak et al. 2003;
Saatçi et al. 2010; Basseville & Nikiforov.

- Scheffer et al. 2009, *Nature* `10.1038/nature08227` (EWS review) — indicators: variance, autocorr, skew.
- Wang et al. 2012, *Nature* `10.1038/nature11655` (flickering).
- Adams & MacKay 2007 (BOCPD, arXiv:0710.3742).
- Truong, Oudre, Vayatis 2020, *Signal Processing* `10.1016/j.sigpro.2020.107619` (`ruptures`).
- Gama et al. 2014 / Lu et al. 2021 (`10.1016/j.jksuci.2021.11.006`) — concept-drift review.
