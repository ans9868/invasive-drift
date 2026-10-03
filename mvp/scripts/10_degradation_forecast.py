#!/usr/bin/env python3
"""Degradation forecast - report ALL targets with the bias ladder.

Per session: MLP decoder frozen from the first ref_frac -> health(t) per block -> denoise with a
constant-acceleration Kalman smoother -> latent [level, slope, accel].
Targets: level(t+1), rate(t+1), accel(t+1), D_h(t) = (L(t)-L(t+h))/L(t) for h in {1,3} blocks, tiers.
Ladder per target: chance < persistence < trend < panel. Session-out CV.
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd
import h5py
from scipy.signal import lfilter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decoders import MLPDec
from sklearn.linear_model import Ridge

BIN_MS = 20.0; TAU_MS = 240.0


def load(path):
    with h5py.File(path, "r") as h:
        st = h["units/spike_times"][:]; idx = h["units/spike_times_index"][:]
        b = np.concatenate([[0], idx]); n = len(idx)
        spies = [st[b[u]:b[u + 1]] for u in range(n)]
        posg = h["processing/behavior/Position/cursor_pos"]; ts = posg["timestamps"][:]
        pos = posg["data"][:].astype(np.float32)
        vel = h["processing/behavior/Velocity/cursor_vel/data"][:].astype(np.float32)
    bs = BIN_MS / 1000.0; edges = np.concatenate([[ts[0] - bs], ts])
    X = np.zeros((len(ts), n), np.float32)
    for u in range(n):
        X[:, u] = np.histogram(spies[u], bins=edges)[0]
    return X, pos, vel


def exp_filt(x, tau_ms=TAU_MS, bin_ms=BIN_MS):
    t = np.arange(0.0, tau_ms, bin_ms); k = np.exp(-t / tau_ms); k /= k.sum()
    return lfilter(k, [1.0], x, axis=0).astype(np.float32)


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()


def kalman_smooth(z, dt, q=(1e-3, 1e-4, 1e-5), r=5e-3):
    """constant-acceleration model on health z (T,) -> smoothed [level, slope, accel] (T,3)."""
    T = len(z)
    F = np.array([[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]], float)
    H = np.array([[1.0, 0, 0]]); Q = np.diag(q); R = np.array([[r]])
    x = np.array([z[0], 0.0, 0.0]); P = np.eye(3)
    xs, Ps = [], []
    for t in range(T):
        x = F @ x; P = F @ P @ F.T + Q
        S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
        x = x + (K @ (z[t] - H @ x)).ravel(); P = (np.eye(3) - K @ H) @ P
        xs.append(x.copy()); Ps.append(P.copy())
    xs = np.array(xs); Ps = np.array(Ps); xsm = xs.copy()
    for t in range(T - 2, -1, -1):
        Pp = F @ Ps[t] @ F.T + Q; C = Ps[t] @ F.T @ np.linalg.inv(Pp)
        xsm[t] = xs[t] + C @ (xsm[t + 1] - F @ xs[t])
    return xsm


def session_series(path, block_s, ref_frac):
    X, pos, vel = load(path); X = exp_filt(X)
    keep = ~np.isnan(vel).any(1); X, pos, vel = X[keep], pos[keep], vel[keep]
    nref = int(ref_frac * len(vel))
    dec = MLPDec(L=3, hidden=64, max_iter=150).fit(X[:nref], vel[:nref])
    with h5py.File(path, "r") as h:
        ts = h["processing/behavior/Position/cursor_pos/timestamps"][:][keep]
    t0 = ts[0]
    nb = int((ts[-1] - t0) // block_s)
    H, F, prev = [], [], None
    for b in range(nb):
        lo = t0 + b * block_s; hi = lo + block_s; sel = (ts >= lo) & (ts < hi)
        if sel.sum() < 250:
            continue
        P = dec.predict(X[sel])
        H.append(r2(vel[sel][-len(P):], P))
        Xb = X[sel]; rates = Xb.mean(0)
        corr = np.nan
        if Xb.shape[0] > 10 and Xb.shape[1] > 1:
            C = np.corrcoef(Xb.T); iu = np.triu_indices_from(C, 1); corr = float(np.nanmean(C[iu]))
        rate = float(rates.mean()); rstd = float(rates.std())
        drate = float(np.abs(rates - prev).sum()) if prev is not None else 0.0
        prev = rates
        F.append([rate, float((rates > 0.05).sum()), rstd, corr, rate / (rstd + 1e-9), rstd / (rate + 1e-9), drate])
    return np.array(H), np.array(F)


FEATS = ["rate", "act", "rstd", "corr", "snr", "cv", "drate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/perich/sub-C")
    ap.add_argument("--block-s", type=float, default=300.0)
    ap.add_argument("--ref-frac", type=float, default=0.3)
    args = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(os.path.join(args.data, "*.nwb"))):
        try:
            H, F = session_series(p, args.block_s, args.ref_frac)
        except Exception as exc:
            print("SKIP", os.path.basename(p), exc); continue
        if len(H) < 5:
            continue
        sm = kalman_smooth(H, args.block_s / 60.0)   # per-minute units
        sess = os.path.basename(p).split("ses-")[1].split("_")[0]
        print(f"  {sess}: blocks={len(H)} H[{H.min():.2f},{H.max():.2f}] slope_start={sm[0,1]:+.3f}/min accel={sm[0,2]:+.3f}")
        for b in range(len(H) - 3):
            r = dict(sess=sess, lev=sm[b, 0], slp=sm[b, 1], acc=sm[b, 2])
            for i, k in enumerate(FEATS):
                r[k] = F[b, i]
            r["Y_level"] = sm[b + 1, 0]; r["Y_rate"] = sm[b + 1, 1]; r["Y_accel"] = sm[b + 1, 2]
            L = sm[b, 0]
            r["Y_D1"] = (L - sm[b + 1, 0]) / L if abs(L) > 1e-6 else np.nan
            r["Y_D3"] = (L - sm[b + 3, 0]) / L if abs(L) > 1e-6 else np.nan
            rows.append(r)
    df = pd.DataFrame(rows).dropna()
    sessions = sorted(df.sess.unique())
    print(f"\nrows={len(df)} sessions={len(sessions)}")
    targets = ["Y_level", "Y_rate", "Y_accel", "Y_D1", "Y_D3"]
    # tier thresholds from D3 (15-min fractional drop)
    q = df.Y_D3.quantile([0.33, 0.66]).values
    df["tier"] = np.digitize(df.Y_D3, q)
    print(f"D3 tiers (small/med/large) at {q.round(3)} -> counts {np.bincount(df.tier).tolist()}")
    Xc = df[FEATS].to_numpy()
    for tgt in targets:
        y = df[tgt].to_numpy()
        feat = np.column_stack([Xc, df[['lev', 'slp', 'acc']].to_numpy()])
        # persistence per target
        if tgt == "Y_level":
            pers = df.lev.to_numpy()
        elif tgt == "Y_rate":
            pers = df.slp.to_numpy()
        elif tgt == "Y_accel":
            pers = df.acc.to_numpy()
        elif tgt == "Y_D1":
            pers = -df.slp.to_numpy() * (args.block_s / 60.0) / (df.lev.to_numpy() + 1e-9)
        else:
            pers = -df.slp.to_numpy() * (3 * args.block_s / 60.0) / (df.lev.to_numpy() + 1e-9)
        preds = {k: np.zeros(len(df)) for k in ["chance", "persistence", "trend", "panel"]}
        for sess in sessions:
            te = (df.sess == sess).to_numpy(); tr = ~te
            preds["chance"][te] = y[tr].mean()
            preds["persistence"][te] = pers[te]
            preds["trend"][te] = pers[te] + (y[tr] - pers[tr]).mean()
            m, s = feat[tr].mean(0), feat[tr].std(0) + 1e-9
            dec = Ridge(alpha=10.0).fit((feat[tr] - m) / s, y[tr])
            preds["panel"][te] = dec.predict((feat[te] - m) / s)
        line = "  ".join(f"{k}={r2(y, v):+.3f}" for k, v in preds.items())
        d = r2(y, preds["panel"]) - r2(y, preds["trend"])
        print(f"{tgt:9s} {line}   dPanel_trend={d:+.3f}")


if __name__ == "__main__":
    main()
