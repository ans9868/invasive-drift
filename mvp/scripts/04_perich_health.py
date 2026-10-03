#!/usr/bin/env python3
"""Perich 000688 - cursor-velocity decoder + health(t). sub-C center-out sessions.

spike bins 20 ms aligned to cursor_vel timestamps; causal exp filter tau=240 ms;
z-score; Ridge (alpha grid); metric = R^2 (cursor velocity). Modes: intra | zero | fewshot.
"""
import argparse, glob, os
import h5py, numpy as np
from scipy.signal import lfilter
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIN_MS = 20.0
TAU_MS = 240.0


def date_of(p):
    return os.path.basename(p).split("ses-")[1].split("_")[0]


def load_session(path):
    with h5py.File(path, "r") as h:
        st = h["units/spike_times"][:]
        idx = h["units/spike_times_index"][:]
        b = np.concatenate([[0], idx])
        n = len(idx)
        spies = [st[b[u]:b[u + 1]] for u in range(n)]
        vg = h["processing/behavior/Velocity/cursor_vel"]
        ts = vg["timestamps"][:]
        vel = vg["data"][:].astype(np.float32)
    bs = BIN_MS / 1000.0
    edges = np.concatenate([[ts[0] - bs], ts])
    X = np.zeros((len(ts), n), np.float32)
    for u in range(n):
        X[:, u] = np.histogram(spies[u], bins=edges)[0]
    return X, vel


def exp_filt(x, tau_ms=TAU_MS, bin_ms=BIN_MS):
    t = np.arange(0.0, tau_ms, bin_ms)
    k = np.exp(-t / tau_ms)
    k /= k.sum()
    return lfilter(k, [1.0], x, axis=0).astype(np.float32)


def fit_eval(xtr, ytr, xte, yte):
    m = xtr.mean(0); s = xtr.std(0); s[s == 0] = 1
    xtr = (xtr - m) / s; xte = (xte - m) / s
    dec = GridSearchCV(Ridge(), {"alpha": np.logspace(-5, 5, 20)}, cv=3).fit(xtr, ytr)
    return dec.score(xte, yte)


def valid(x, y):
    k = ~np.isnan(y).any(1)
    return x[k], y[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/perich/sub-C")
    ap.add_argument("--mode", choices=["intra", "zero", "fewshot"], default="fewshot")
    ap.add_argument("--few-frac", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="mvp/out")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data, "*.nwb")), key=date_of)
    if args.limit:
        files = files[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    data = []
    for p in files:
        X, y = load_session(p)
        X = exp_filt(X)
        X, y = valid(X, y)
        data.append((date_of(p), X, y))
        print(f"  {date_of(p)}: X={X.shape}")
    print(f"mode={args.mode} n_sessions={len(data)}")
    health = []
    if args.mode == "intra":
        for (d, X, y) in data:
            n = int(0.8 * len(y))
            health.append(fit_eval(X[:n], y[:n], X[n:], y[n:])); print(f"  {d} intra R2={health[-1]:.3f}")
    elif args.mode == "zero":
        _, Xr, yr = data[0]; m = Xr.mean(0); s = Xr.std(0); s[s == 0] = 1
        dec = GridSearchCV(Ridge(), {"alpha": np.logspace(-5, 5, 20)}, cv=3).fit((Xr - m) / s, yr)
        for (d, X, y) in data:
            health.append(dec.score((X - m) / s, y)); print(f"  {d} zero R2={health[-1]:.3f}")
    else:
        _, Xr, yr = data[0]
        for (d, X, y) in data:
            k = int(args.few_frac * len(y))
            xtr = np.vstack([Xr, X[:k]]); ytr = np.concatenate([yr, y[:k]])
            health.append(fit_eval(xtr, ytr, X[k:], y[k:])); print(f"  {d} few R2={health[-1]:.3f}")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(health)), health, "o-")
    ax.axhline(0, ls="--", color="gray")
    ax.set_xlabel("session index"); ax.set_ylabel("R2 (cursor vel)")
    ax.set_title(f"Perich sub-C health(t) - {args.mode} (n={len(data)})")
    fig.tight_layout()
    out = os.path.join(args.out, f"health_perich_{args.mode}.png")
    fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
