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

## Key refs
- Scheffer et al. 2009, *Nature* `10.1038/nature08227` (EWS review) — indicators: variance, autocorr, skew.
- Wang et al. 2012, *Nature* `10.1038/nature11655` (flickering).
- Adams & MacKay 2007 (BOCPD, arXiv:0710.3742).
- Truong, Oudre, Vayatis 2020, *Signal Processing* `10.1016/j.sigpro.2020.107619` (`ruptures`).
- Gama et al. 2014 / Lu et al. 2021 (`10.1016/j.jksuci.2021.11.006`) — concept-drift review.
