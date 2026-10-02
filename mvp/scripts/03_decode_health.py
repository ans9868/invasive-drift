#!/usr/bin/env python3
"""MVP step 3 - decoder + health(t) across sessions (FALCON M1-A).

Modes:
  intra   : 70/30 within each session (per-session decodability).
  zero    : frozen decoder from a reference session, tested on all (zero-shot).
  fewshot : train on reference session + a few trials of the target session (FALCON-style).

Run on a compute node:  sbatch mvp/scripts/03_decode_health.sbatch
"""
import argparse, glob, os
import h5py, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    X = np.zeros((len(move), n_units), dtype=np.float32)
    for u in range(n_units):
        ts = spikes[u]
        if ts.size == 0:
            continue
        X[:, u] = np.searchsorted(ts, contact + win_post) - np.searchsorted(ts, move - win_pre)
    return X, y


def date_of(p):
    return os.path.basename(p).split("ses-")[1].split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/falcon/m1a/sub-MonkeyL-held-in-calib")
    ap.add_argument("--target", default="tgt_loc")
    ap.add_argument("--mode", choices=["intra", "zero", "fewshot"], default="fewshot")
    ap.add_argument("--few-frac", type=float, default=0.2)
    ap.add_argument("--win-pre", type=float, default=0.3)
    ap.add_argument("--win-post", type=float, default=0.3)
    ap.add_argument("--out", default="mvp/out")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data, "*.nwb")), key=date_of)
    os.makedirs(args.out, exist_ok=True)
    data = []
    for p in files:
        X, y = load_session(p, args.win_pre, args.win_post, args.target)
        data.append((date_of(p), np.log1p(X), y))
    dates = [d for d, _, _ in data]
    print(f"mode={args.mode} target={args.target} sessions={dates}")
    health = []
    if args.mode == "intra":
        for (d, X, y) in data:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
            acc = LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte)
            health.append(acc); print(f"  {d}: intra acc={acc:.3f}")
        ref_line = 1.0 / len(np.unique(y)); line_label = "chance"
    elif args.mode == "zero":
        dref, Xref, yref = data[0]
        clf = LogisticRegression(max_iter=2000).fit(Xref, yref)
        for (d, X, y) in data:
            acc = clf.score(X, y); health.append(acc); print(f"  {d}: zero acc={acc:.3f}")
        ref_line = np.bincount(np.unique(yref, return_inverse=True)[1]).max() / len(yref)
        line_label = "majority"
    else:
        dref, Xref, yref = data[0]
        for (d, X, y) in data:
            Xfit, Xte, yfit, yte = train_test_split(X, y, test_size=1 - args.few_frac, random_state=0)
            Xtr = np.vstack([Xref, Xfit]); ytr = np.concatenate([yref, yfit])
            acc = LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte)
            health.append(acc); print(f"  {d}: few-shot({args.few_frac:.0%}) acc={acc:.3f}")
        ref_line = np.bincount(np.unique(yref, return_inverse=True)[1]).max() / len(yref)
        line_label = "majority"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(dates)), health, "o-", label=args.mode)
    ax.axhline(ref_line, ls="--", color="gray", label=line_label)
    ax.set_xticks(range(len(dates))); ax.set_xticklabels(dates, rotation=45)
    ax.set_ylabel("accuracy"); ax.set_title(f"health(t) - FALCON M1-A {args.mode} ({args.target})")
    ax.legend(); fig.tight_layout()
    out = os.path.join(args.out, f"health_{args.mode}.png"); fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
