"""Metric functions for grid_within_session_27. Pure functions, no I/O.

All operate on the EVAL rows only.
  y : (n,2) true velocity        p : (n,2) prediction
Definitions are LOCKED in PLAN.md section 5b.
"""
import numpy as np

EPS = 1e-9


def _r2(y, p):
    return float(1.0 - ((p - y) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + EPS))


def _corr(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    if a.std() < EPS or b.std() < EPS:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def wrap180(a):
    """Wrap degrees into [-180, 180)."""
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def velocity_metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    m = {}
    m["r2_all"] = _r2(y, p)
    m["r2_vx"] = _r2(y[:, [0]], p[:, [0]])
    m["r2_vy"] = _r2(y[:, [1]], p[:, [1]])
    m["corr_vx"] = _corr(p[:, 0], y[:, 0])
    m["corr_vy"] = _corr(p[:, 1], y[:, 1])
    err = p - y
    m["mse"] = float((err ** 2).sum(1).mean())
    m["bias_vx"] = float(err[:, 0].mean()); m["bias_vy"] = float(err[:, 1].mean())
    for i, nm in ((0, "vx"), (1, "vy")):
        m[f"slope_{nm}"] = float(np.cov(p[:, i], y[:, i])[0, 1] / (p[:, i].var() + EPS))
    me = err.mean(0)
    m["mse_bias2"] = float((me ** 2).sum())
    m["mse_var"] = float(((err - me) ** 2).sum(1).mean())
    return m


def lag_bins(y, p, max_lag=20):
    """argmax_l corr(p_t, y_{t+l}), returned negated so POSITIVE = the output LAGS the truth."""
    n = len(y); best, bl = -np.inf, 0
    for l in range(-max_lag, max_lag + 1):
        a, b = (p[:n - l], y[l:]) if l >= 0 else (p[-l:], y[:n + l])
        if len(a) < 30:
            continue
        c = _corr(a, b)
        if np.isfinite(c) and c > best:
            best, bl = c, l
    return int(-bl)


_DKEYS = ("ang_bias_deg", "ang_err_mean_deg", "ang_abs_err_deg", "ang_resultant_R",
          "circ_std_deg", "speed_ratio", "speed_corr", "dir_acc8")


def direction_metrics(y, p, moving):
    y = np.asarray(y, float); p = np.asarray(p, float)
    moving = np.asarray(moving, bool)
    if moving.sum() < 30:
        return {k: np.nan for k in _DKEYS}
    th = np.arctan2(y[moving, 1], y[moving, 0])
    thp = np.arctan2(p[moving, 1], p[moving, 0])
    d = wrap180(np.degrees(thp - th))                       # degrees, wrapped
    dr = np.radians(d)
    m = {}
    m["ang_bias_deg"] = float(np.degrees(np.arctan2(np.sin(dr).mean(), np.cos(dr).mean())))
    ad = np.abs(d)
    m["ang_err_mean_deg"] = float(ad.mean())
    m["ang_abs_err_deg"] = float(np.median(ad))
    R = float(np.hypot(np.sin(dr).mean(), np.cos(dr).mean()))
    m["ang_resultant_R"] = R
    m["circ_std_deg"] = float(np.degrees(np.sqrt(-2.0 * np.log(max(R, EPS)))))
    sy = np.linalg.norm(y[moving], axis=1); sp = np.linalg.norm(p[moving], axis=1)
    m["speed_ratio"] = float(np.median(sp / (sy + EPS)))
    m["speed_corr"] = _corr(sp, sy)
    bins = np.linspace(-180, 180, 9)
    bi = lambda a: np.clip(np.digitize(a, bins) - 1, 0, 7)
    m["dir_acc8"] = float((bi(np.degrees(thp)) == bi(np.degrees(th))).mean())
    return m


def baseline_metrics(y, dirbin=None, ref_dir_vel=None, ref_mean=None, lags=(1, 12)):
    y = np.asarray(y, float)
    out = {}
    mu = ref_mean if ref_mean is not None else y.mean(0)
    out["r2_mean"] = _r2(y, np.broadcast_to(mu, y.shape))
    for L in lags:
        out[f"r2_persist_lag{L}"] = _r2(y[L:], y[:-L]) if len(y) > L + 20 else np.nan
    if dirbin is not None and ref_dir_vel is not None:
        dirbin = np.asarray(dirbin)
        ok = dirbin >= 0
        if ok.sum() > 20:
            pt = np.asarray(ref_dir_vel, float)[np.clip(dirbin[ok], 0, len(ref_dir_vel) - 1)]
            out["r2_target"] = _r2(y[ok], pt)
        else:
            out["r2_target"] = np.nan
    return out


def agreement_metrics(preds, y):
    """preds: list of (n,2) predictions from DIFFERENT decoders on the SAME rows."""
    y = np.asarray(y, float); P = np.asarray(preds, float)
    if P.ndim != 3 or P.shape[0] < 2:
        return {k: np.nan for k in ("pred_corr_mean", "ens_disagree", "err_corr_mean", "r2_consensus")}
    cor, ec = [], []
    for i in range(P.shape[0]):
        for j in range(i + 1, P.shape[0]):
            cor.append(_corr(P[i], P[j]))
            ec.append(_corr(P[i] - y, P[j] - y))
    return dict(pred_corr_mean=float(np.nanmean(cor)), err_corr_mean=float(np.nanmean(ec)),
                ens_disagree=float(P.std(0).mean()), r2_consensus=_r2(y, P.mean(0)))


def reliability(predA, predB):
    """Unit split-half decoding ceiling: correlation of two half-unit decoders' predictions."""
    predA = np.asarray(predA, float); predB = np.asarray(predB, float)
    return float(np.nanmean([_corr(predA[:, 0], predB[:, 0]), _corr(predA[:, 1], predB[:, 1])]))


def kf_uncertainty(traceP):
    t = np.asarray(traceP, float).ravel()
    if len(t) < 5:
        return dict(kf_post_var_mean=float(t.mean()) if len(t) else np.nan, kf_post_var_growth=np.nan)
    return dict(kf_post_var_mean=float(t.mean()),
                kf_post_var_growth=float(np.polyfit(np.arange(len(t)), t, 1)[0]))
