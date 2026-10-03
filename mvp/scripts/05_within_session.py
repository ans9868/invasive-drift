#!/usr/bin/env python3
"""Within-session decoder drift (fast clock). One session, units FIXED (no matching).

Train a cursor-velocity decoder on the first ref_frac of a session; evaluate R^2 in successive
time blocks. A decaying curve = within-session drift.
Also prints per-block 'frozen from start' R^2 AND a within-block ceiling (CV).
"""
import argparse, os
import h5py, numpy as np
from scipy.signal import lfilter
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIN_MS = 20.0
TAU_MS = 240.0


def load(path):
    with h5py.File(path, "r") as h:
        st = h["units/spike_times"][:]
        idx = h["units/spike_times_index"][:]
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
    ap.add_argument("--file", required=True)
    ap.add_argument("--block-s", type=float, default=300.0)
    ap.add_argument("--ref-frac", type=float, default=0.2)
    ap.add_argument("--out", default="mvp/out")
    args = ap.parse_args()
    X, y, ts = load(args.file); X = exp_filt(X)
    keep = ~np.isnan(y).any(1); X, y, ts = X[keep], y[keep], ts[keep]
    nref = int(args.ref_frac * len(y))
    m, s = X[:nref].mean(0), X[:nref].std(0); s[s == 0] = 1
    Xs = (X - m) / s
    dec = GridSearchCV(Ridge(), {"alpha": np.logspace(-5, 5, 20)}, cv=3).fit(Xs[:nref], y[:nref])
    t0 = ts[0]; dur = (ts[-1] - t0) / 60.0
    nb = int((ts[-1] - t0) // args.block_s)
    print(f"units={X.shape[1]} dur={dur:.1f}min ref={args.ref_frac:.0%} blocks={nb}@{args.block_s:.0f}s")
    xs, froz, ceil = [], [], []
    for b in range(nb):
        lo = t0 + b * args.block_s; hi = lo + args.block_s
        sel = (ts >= lo) & (ts < hi)
        if sel.sum() < 200:
            continue
        xs.append((b * args.block_s + args.block_s / 2) / 60.0)
        froz.append(dec.score(Xs[sel], y[sel]))
        # within-block ceiling (train/test split of the block)
        nb2 = sel.sum(); k = int(0.7 * nb2)
        Xb = Xs[sel]; yb = y[sel]
        cb = GridSearchCV(Ridge(), {"alpha": np.logspace(-5, 5, 20)}, cv=3).fit(Xb[:k], yb[:k])
        ceil.append(cb.score(Xb[k:], yb[k:]))
        print(f"  t={xs[-1]:6.1f}min  frozen R2={froz[-1]: .3f}   block-ceiling R2={ceil[-1]: .3f}")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, froz, "o-", label="frozen-from-start")
    ax.plot(xs, ceil, "s--", color="gray", label="within-block ceiling")
    ax.axvline(args.ref_frac * dur, color="k", ls=":", lw=1)
    ax.set_xlabel("time in session (min)"); ax.set_ylabel("R2 (cursor vel)")
    ax.set_title(f"within-session drift: {os.path.basename(args.file)}")
    ax.legend(); fig.tight_layout()
    out = os.path.join(args.out, "within_session.png"); fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
