# 16 — Results so far: what we tried, what worked

## Decoders (intra-session, Perich sub-C, N=53, mean R²)
| decoder | mean R² | verdict |
|---|---|---|
| gru_L10 | 0.607 | ✅ best |
| mlp_L3 | 0.576 | ✅ |
| wiener_L5 | 0.404 | ✔ mid (history helps) |
| kf_posvel | 0.393 | ✔ mid (≥ kf_vel — Gilja #2 reproduced) |
| kf_vel | 0.383 | ✔ mid |
| ridge | 0.357 | baseline |

## Decoder target
- ❌ `tgt_loc` classification (wrong task); ✅ **EMG** (FALCON M1) & ✅ **cursor velocity** (Perich).
- ❌ zero-shot cross-session (unit mismatch); ✅ few-shot recalibration recovers.

## Phenomenon
- ✅ **within-session drift** (6/7 sessions decay; frozen R² 0.12→0.05).

## Degradation targets to forecast
| target | result | verdict |
|---|---|---|
| level | panel ≈ trend | trivial — no value |
| **rate** | panel > trend, Δ+0.50 | 🟡 promising; injection ⇒ bar is **hist-slope**, re-test |
| acceleration | Δ+0.25 | ❌ not real (injection: no gain over slope; contaminates slope) |
| D1 (5-min drop) | panel < trend | ❌ |
| D3 (15-min drop) | persist R²=−1.27 | ❌ numerically broken |

## Methods
| method | verdict |
|---|---|
| persistence | ✔ weak null |
| trend (persist + mean-Δ) | ✔ strong |
| **historical-slope** | ✔ **strongest — the real bar** |
| const-accel Kalman smoother | ✔ level/rate; ⚠️ wide CI; ❌ smears steps; ❌ accel unreliable |
| panel (ridge, rich feat.) | 🟡 beat *trend* on rate; not yet vs *hist-slope* |
| injection+coverage | ✔ (caught the accel over-fit) |
| CSD/Hurst/DFA/GARCH/RQA/Lyapunov/Hawkes/cointegration/HMM | ✗ not identifiable at n≈4–30 |

## Bottom line
Forecastable = **rate only** (must beat **hist-slope**). **Acceleration out**, fractional drops out, level trivial.
**Steps need a separate change-point layer** (BOCPD/PELT). All on small n → 🟡 until re-tested vs hist-slope.
