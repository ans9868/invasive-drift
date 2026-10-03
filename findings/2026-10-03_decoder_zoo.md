# Finding — Decoder zoo: nonlinear decoders win

**Date:** 2026-10-03
**Dataset:** Perich & Miller `000688`, sub-C, 53 center-out sessions (macaque M1/PMd, cursor velocity)
**Script:** `mvp/scripts/09_decode_zoo.py` (+ `mvp/decoders.py`), job ~19100558
**Metric:** intra-session R² (train first 80% → test last 20%), mean over 53 sessions.

## Result
| decoder | mean R² | note |
|---|---|---|
| **GRU** (`gru_L10`) | **0.607** | best |
| **MLP** (`mlp_L3`) | **0.576** | |
| Wiener (`wiener_L5`) | 0.404 | history lags help |
| KF pos+vel (`kf_posvel`) | 0.393 | ≥ kf_vel |
| KF velocity (`kf_vel`) | 0.383 | |
| Ridge | 0.357 | baseline |

## Interpretations
- **Nonlinear (GRU/MLP) ≫ linear** at this task/feature set.
- **History helps**: Wiener (lags) > plain Ridge.
- **Gilja et al. 2012 innovation #2 reproduces**: including **cursor position in the KF state**
  (`kf_posvel` ≥ `kf_vel`) helps in essentially every session.
- Range 0.03–0.88 (session-quality driven); in the published macaque-M1 ballpark.
- **Decoder of record = GRU (or MLP).**

## Caveats
- Simple kinematic KF underperforms here; ReFIT's other half (closed-loop intention training) can't be
  done offline, which is where the KF normally wins.
