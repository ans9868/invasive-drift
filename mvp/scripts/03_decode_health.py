#!/usr/bin/env python3
"""MVP step 3 — decoder + health(t) curve across sessions (FALCON M1-A).

Trains a spike-count decoder on one session (target/object classification) and evaluates a *frozen*
decoder cross-session -> the 'health(t)' decay curve. Reads NWB via h5py (no pynwb needed).

Run on a compute node, e.g.:
  sbatch mvp/scripts/03_decode_health.sbatch
"""
import argparse
import glob
import os

import h5py
import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _as_str(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def load_session(path, win_pre, win_post, target):
    with h5py.File(path, "r") as h:
        st = h["units/spike_times"][:]
        idx = h["units/spike_times_index"][:]
        tr = h["intervals/trials"]
        move = tr["move_onset_time"][:]
        contact = tr["contact_time"][:]
        y = _as_str(tr[target][:])
        bounds = np.concatenate([[0], idx])
        n_units = len(idx)
        spikes = [st[bounds[u]:bounds[u + 1]] for u in range(n_units)]
    n_tr = len(move)
    X = np.zeros((n_tr, n_units), dtype=np.float32)
    for u in range(n_units):
        ts = spikes[u]
        if ts.size == 0:
            continue
        lo = np.searchsorted(ts, move - win_pre)
        hi = np.searchsorted(ts, contact + win_post)
        X[:, u] = hi - lo
    return X, y


def date_of(p):
    return os.path.basename(p).split("ses-")[1].split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/falcon/m1a/sub-MonkeyL-held-in-calib")
    ap.add_argument("--target", default="tgt_loc")
    ap.add_argument("--win-pre", type=float, default=0.3)
    ap.add_argument("--win-post", type=float, default=0.3)
    ap.add_argument("--out", default="mvp/out")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "*.nwb")), key=date_of)
    os.makedirs(args.out, exist_ok=True)
    print(f"sessions in {args.data}:")
    data = []
    for p in files:
        X, y = load_session(p, args.win_pre, args.win_post, args.target)
        print(f"  {date_of(p)}  trials={len(y):4d}  classes={len(np.unique(y)):3d}  units={X.shape[1]}")
        data.append((date_of(p), X, y))
    if not data:
        raise SystemExit("no sessions found")

    def prep(X):
        return np.log1p(X)

    print("\nintra-session accuracy (70/30):")
    for (d, X, y) in data:
        Xtr, Xte, ytr, yte = train_test_split(prep(X), y, test_size=0.3, random_state=0)
        acc = LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte)
        print(f"  {d}: acc={acc:.3f}")

    d0, X0, y0 = data[0]
    clf = LogisticRegression(max_iter=2000).fit(prep(X0), y0)
    print(f"\nfrozen decoder from {d0} -> health(t):")
    health, dates = [], []
    for (d, X, y) in data:
        acc = clf.score(prep(X), y)
        health.append(acc)
        dates.append(d)
        print(f"  {d}: acc={acc:.3f}")
    maj = np.bincount(np.unique(y0, return_inverse=True)[1]).max() / len(y0)
    print(f"  majority-class baseline (from {d0}): {maj:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(dates)), health, "o-", label="frozen decoder")
    ax.axhline(maj, ls="--", color="gray", label="majority baseline")
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45)
    ax.set_ylabel("accuracy")
    ax.set_title(f"health(t) — FALCON M1-A ({args.target})")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(args.out, "health.png")
    fig.savefig(out, dpi=120)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
