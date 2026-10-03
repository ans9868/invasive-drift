#!/usr/bin/env python3
"""Degradation v2: level + RATE only (accel dropped). Known per-block sigma -> local-linear Kalman
smoother; baselines persistence -> hist-slope; walk-forward skill 1-MSE/MSE_base; floor-crossing prob;
simple CUSUM step count."""
import argparse, glob, os, sys
import numpy as np, pandas as pd, h5py
from scipy.signal import lfilter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decoders import MLPDec

BIN_MS = 20.0; TAU_MS = 240.0


def load(path):
    with h5py.File(path, "r") as h:
        st = h["units/spike_times"][:]; idx = h["units/spike_times_index"][:]
        b = np.concatenate([[0], idx]); n = len(idx)
        spies = [st[b[u]:b[u + 1]] for u in range(n)]
        posg = h["processing/behavior/Position/cursor_pos"]; ts = posg["timestamps"][:]
        vel = h["processing/behavior/Velocity/cursor_vel/data"][:].astype(np.float32)
    bs = BIN_MS / 1000.0; edges = np.concatenate([[ts[0] - bs], ts])
    X = np.zeros((len(ts), n), np.float32)
    for u in range(n):
        X[:, u] = np.histogram(spies[u], bins=edges)[0]
    return X, vel, ts


def exp_filt(x, tau_ms=TAU_MS, bin_ms=BIN_MS):
    t = np.arange(0.0, tau_ms, bin_ms); k = np.exp(-t / tau_ms); k /= k.sum()
    return lfilter(k, [1.0], x, axis=0).astype(np.float32)


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()


def smoother_known(z, sig, t, q=(1e-5, 1e-6)):
    """local-linear-trend Kalman (level, slope) with KNOWN obs noise sig. -> xs (T,2), Ps (T,2,2)."""
    T = len(z); n = 2
    xs = np.zeros((T, n)); Ps = np.zeros((T, n, n)); Fs = [np.eye(n)]
    x = np.array([z[0], 0.0]); P = np.eye(n)
    for i in range(T):
        if i > 0:
            dt = t[i] - t[i - 1]
            F = np.array([[1, dt], [0, 1]], float)
            x = F @ x; P = F @ P @ F.T + np.diag(q)
        else:
            F = np.eye(n)
        Fs.append(F)
        H = np.array([[1.0, 0]]); R = np.array([[sig[i] ** 2]])
        S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
        x = x + (K @ (z[i] - H @ x)).ravel(); P = (np.eye(n) - K @ H) @ P
        xs[i] = x; Ps[i] = P
    xs_s = xs.copy(); Ps_s = Ps.copy()
    for i in range(T - 2, -1, -1):
        F = Fs[i + 1]; Pp = F @ Ps[i] @ F.T + np.diag(q); C = Ps[i] @ F.T @ np.linalg.inv(Pp)
        xs_s[i] = xs[i] + C @ (xs_s[i + 1] - F @ xs[i])
        Ps_s[i] = Ps[i] + C @ (Ps_s[i + 1] - Pp) @ C.T
    return xs_s, Ps_s


def block_sigma(dec, Xb, vb):
    """split block in 4; R2 per quarter; sigma = std/sqrt(4)."""
    k = len(vb) // 4
    vals = [r2(vb[i * k:(i + 1) * k], dec.predict(Xb[i * k:(i + 1) * k])) for i in range(4)]
    return max(1e-3, float(np.std(vals) / np.sqrt(4)))


def session_series(path, block_s, ref_frac):
    X, vel, ts = load(path); X = exp_filt(X)
    keep = ~np.isnan(vel).any(1); X, vel, ts = X[keep], vel[keep], ts[keep]
    nref = int(ref_frac * len(vel))
    dec = MLPDec(L=3, hidden=64, max_iter=150).fit(X[:nref], vel[:nref])
    t0 = ts[0]; nb = int((ts[-1] - t0) // block_s)
    H, S = [], []
    for b in range(nb):
        lo = t0 + b * block_s; hi = lo + block_s; sel = (ts >= lo) & (ts < hi)
        if sel.sum() < 250:
            continue
        Xb = X[sel]; P = dec.predict(Xb); vb = vel[sel][-len(P):]
        H.append(r2(vb, P)); S.append(block_sigma(dec, Xb, vel[sel]))
    return np.array(H), np.array(S)


def cusum_steps(z, thresh=3.0):
    z = (z - z.mean()) / (z.std() + 1e-9); c = np.zeros(len(z)); cnt = 0
    for i in range(1, len(z)):
        c[i] = max(0, c[i - 1] + (z[i] - z[i - 1]))
        if c[i] > thresh:
            cnt += 1; c[i] = 0
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/perich/sub-C")
    ap.add_argument("--block-s", type=float, default=120.0)
    ap.add_argument("--ref-frac", type=float, default=0.3)
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data, "*.nwb")))
    if args.n:
        files = files[:args.n]
    rows = []; nsteps = []
    for p in files:
        try:
            H, S = session_series(p, args.block_s, args.ref_frac)
        except Exception as exc:
            print("SKIP", os.path.basename(p), exc); continue
        if len(H) < 4:
            continue
        t = np.arange(len(H)) * (args.block_s / 60.0)
        xs, Ps = smoother_known(H, S, t)
        sess = os.path.basename(p).split("ses-")[1].split("_")[0]
        nsteps.append(cusum_steps(xs[:, 0]))
        print(f"  {sess}: blocks={len(H)} H[{H.min():.2f},{H.max():.2f}] slope0={xs[0,1]:+.3f}/min steps={nsteps[-1]}")
        for b in range(len(H) - 1):
            rows.append(dict(sess=sess, lev=xs[b, 0], slp=xs[b, 1], sig=S[b],
                             hnext=H[b + 1], levnext=xs[b + 1, 0], slpnext=xs[b + 1, 1]))
    df = pd.DataFrame(rows)
    sessions = sorted(df.sess.unique())
    print(f"\nrows={len(df)} sessions={len(sessions)} mean steps/session={np.mean(nsteps):.1f}")
    alpha = args.block_s / 60.0
    for tgt, cur in [("hnext", "lev"), ("slpnext", "slp")]:
        y = df[tgt].to_numpy(); c = df[cur].to_numpy()
        preds = {k: np.zeros(len(df)) for k in ["persistence", "hist_slope", "model"]}
        for s in sessions:
            te = (df.sess == s).to_numpy(); tr = ~te
            preds["persistence"][te] = c[te]
            d = np.median(np.diff(c[tr])) if tr.sum() > 2 else 0.0
            preds["hist_slope"][te] = c[te] + d
            # model: use the smoothed slope to advance one block
            preds["model"][te] = c[te] + df.slp.to_numpy()[te] * alpha
        mse = {k: ((y - v) ** 2).mean() for k, v in preds.items()}
        base = mse["persistence"]
        print(f"{tgt:9s} skill(1-MSE/MSEpersist): persist=0.000 hist_slope={1-mse['hist_slope']/base:+.3f} model={1-mse['model']/base:+.3f}")


if __name__ == "__main__":
    main()
