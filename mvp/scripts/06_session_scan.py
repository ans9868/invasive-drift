#!/usr/bin/env python3
"""Scan within-session drift across sessions: frozen-decoder R2 first-vs-last block + slope.
Also reports the naive 'trend' baseline (mean slope) so we can see how much a trivial
down-trend predictor would explain.
"""
import argparse, glob, os
import h5py, numpy as np
from scipy.signal import lfilter
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

BIN_MS = 20.0
TAU_MS = 240.0


def load(path):
    with h5py.File(path, "r") as h:
        st = h["units/spike_times"][:]; idx = h["units/spike_times_index"][:]
        b = np.concatenate([[0], idx]); n = len(idx)
        spies = [st[b[u]:b[u + 1]] for u in range(n)]
        vg = h["processing/behavior/Velocity/cursor_vel"]
        ts = vg["timestamps"][:]; y = vg["data"][:].astype(np.float32)
    bs = BIN_MS / 1000.0; edges = np.concatenate([[ts[0] - bs], ts])
    X = np.zeros((len(ts), n), np.float32)
    for u in range(n):
        X[:, u] = np.histogram(spies[u], bins=edges)[0]
    return X, y, ts


def exp_filt(x, tau_ms=TAU_MS, bin_ms=BIN_MS):
    t = np.arange(0.0, tau_ms, bin_ms); k = np.exp(-t / tau_ms); k /= k.sum()
    return lfilter(k, [1.0], x, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/perich/sub-C/*.nwb")
    ap.add_argument("--block-s", type=float, default=300.0)
    ap.add_argument("--ref-frac", type=float, default=0.2)
    args = ap.parse_args()
    files = sorted(glob.glob(args.glob))
    rows = []
    print(f"{'session':14s} {'units':>5s} {'min':>6s} {'R2_first':>8s} {'R2_last':>8s} {'slope/min':>9s}")
    for p in files:
        try:
            X, y, ts = load(p)
        except Exception as exc:
            print(f"  SKIP {os.path.basename(p).split('ses-')[-1][:20]}: {exc}")
            continue
        X = exp_filt(X); keep = ~np.isnan(y).any(1); X, y, ts = X[keep], y[keep], ts[keep]
        nref = int(args.ref_frac * len(y))
        m, s = X[:nref].mean(0), X[:nref].std(0); s[s == 0] = 1; Xs = (X - m) / s
        dec = GridSearchCV(Ridge(), {"alpha": np.logspace(-5, 5, 20)}, cv=3).fit(Xs[:nref], y[:nref])
        t0 = ts[0]; nb = int((ts[-1] - t0) // args.block_s)
        xs, rs = [], []
        for b in range(nb):
            lo = t0 + b * args.block_s; hi = lo + args.block_s; sel = (ts >= lo) & (ts < hi)
            if sel.sum() < 200:
                continue
            xs.append((b * args.block_s + args.block_s / 2) / 60.0); rs.append(dec.score(Xs[sel], y[sel]))
        if len(rs) < 3:
            continue
        slope = float(np.polyfit(xs, rs, 1)[0])
        sess = os.path.basename(p).split("ses-")[1].split("_")[0]
        print(f"{sess:14s} {X.shape[1]:5d} {(ts[-1]-t0)/60:6.1f} {rs[0]:8.3f} {rs[-1]:8.3f} {slope:9.4f}")
        rows.append((sess, rs[0], rs[-1], slope))
    if rows:
        f = np.array([r[1] for r in rows]); l = np.array([r[2] for r in rows]); sl = np.array([r[3] for r in rows])
        print(f"\nN={len(rows)}  R2_first={f.mean():.3f}  R2_last={l.mean():.3f}  slope={sl.mean():+.4f}/min")
        print(f"decayed (first>last): {(f > l).sum()}/{len(rows)} sessions")


if __name__ == "__main__":
    main()
