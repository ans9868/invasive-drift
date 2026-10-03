#!/usr/bin/env python3
"""MVP step 3 - decoder + health(t) for FALCON M1-A (EMG decoding).

FALCON M1 spec (snel-repo/falcon-challenge): bin=20 ms; target=16-muscle EMG;
apply causal exponential filter tau=240 ms; z-score neural; Ridge (alpha grid); metric=R^2.
Modes: intra | zero | fewshot.
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
        g = h["acquisition/preprocessed_emg"]
        muscles = list(g.keys())
        ts = g[muscles[0]]["timestamps"][:]
        emg = np.stack([g[m]["data"][:] for m in muscles], 1).astype(np.float32)
        ev = h["acquisition/eval_mask"]["data"][:].astype(bool)
    bs = BIN_MS / 1000.0
    edges = np.concatenate([[ts[0] - bs], ts])
    X = np.zeros((len(ts), n), np.float32)
    for u in range(n):
        X[:, u] = np.histogram(spies[u], bins=edges)[0]
    return X, emg, ev


def exp_filt(x, tau_ms=TAU_MS, bin_ms=BIN_MS):
    t = np.arange(0.0, tau_ms, bin_ms)
    k = np.exp(-t / tau_ms)
    k /= k.sum()
    return lfilter(k, [1.0], x, axis=0).astype(np.float32)


def fit_eval(xtr, ytr, xte, yte):
    m = xtr.mean(0); s = xtr.std(0); s[s == 0] = 1
    xtr = (xtr - m) / s; xte = (xte - m) / s
    dec = GridSearchCV(Ridge(), {"alpha": np.logspace(-5, 5, 20)}, cv=3)
    dec.fit(xtr, ytr)
    return dec.score(xte, yte)


def valid(x, y):
    keep = ~np.isnan(y).any(1)
    return x[keep], y[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/falcon/m1a/sub-MonkeyL-held-in-calib")
    ap.add_argument("--mode", choices=["intra", "zero", "fewshot"], default="fewshot")
    ap.add_argument("--few-frac", type=float, default=0.2)
    ap.add_argument("--out", default="mvp/out")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data, "*.nwb")), key=date_of)
    os.makedirs(args.out, exist_ok=True)
    data = []
    for p in files:
        X, y, ev = load_session(p)
        X = exp_filt(X)
        X, y = valid(X, y)
        data.append((date_of(p), X, y, ev))
        print(f"  {date_of(p)}: X={X.shape} y={y.shape}")
    dates = [d for d, _, _, _ in data]
    print(f"mode={args.mode}")
    health = []
    if args.mode == "intra":
        for (d, X, y, ev) in data:
            n = int(0.8 * len(y))
            health.append(fit_eval(X[:n], y[:n], X[n:], y[n:]))
            print(f"  {d}: intra R2={health[-1]:.3f}")
        ref_line = 0.0; line_label = "R2=0"
    elif args.mode == "zero":
        _, Xref, yref, _ = data[0]
        m = Xref.mean(0); s = Xref.std(0); s[s == 0] = 1
        dec = GridSearchCV(Ridge(), {"alpha": np.logspace(-5, 5, 20)}, cv=3).fit((Xref - m) / s, yref)
        for (d, X, y, ev) in data:
            health.append(dec.score((X - m) / s, y))
            print(f"  {d}: zero R2={health[-1]:.3f}")
        ref_line = 0.0; line_label = "R2=0"
    else:
        _, Xref, yref, _ = data[0]
        for (d, X, y, ev) in data:
            k = int(args.few_frac * len(y))
            xtr = np.vstack([Xref, X[:k]]); ytr = np.concatenate([yref, y[:k]])
            health.append(fit_eval(xtr, ytr, X[k:], y[k:]))
            print(f"  {d}: few-shot({args.few_frac:.0%}) R2={health[-1]:.3f}")
        ref_line = 0.0; line_label = "R2=0"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(dates)), health, "o-", label=args.mode)
    ax.axhline(ref_line, ls="--", color="gray", label=line_label)
    ax.set_xticks(range(len(dates))); ax.set_xticklabels(dates, rotation=45)
    ax.set_ylabel("R2 (EMG)"); ax.set_title(f"health(t) - FALCON M1-A EMG ({args.mode})")
    ax.legend(); fig.tight_layout()
    out = os.path.join(args.out, f"health_emg_{args.mode}.png")
    fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
