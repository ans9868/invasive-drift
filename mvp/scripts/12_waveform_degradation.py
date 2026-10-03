#!/usr/bin/env python3
"""Waveform-based degradation: per-unit per-block waveform stability / amplitude / SNR over a session.

Idea: unit waveforms are the *upstream* signal of loss/isolation/gain. Track each unit's mean waveform
over time; a dying/drifting unit's waveform changes (low stability) and/or shrinks (amplitude drop).
"""
import argparse, os
import numpy as np, h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with h5py.File(path, "r") as h:
        st = h["units/spike_times"][:]
        sidx = h["units/spike_times_index"][:]
        wf = h["units/waveforms"][:, 0]
        wiidx = h["units/waveforms_index_index"][:]
    return st, sidx, wf, wiidx


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--block-s", type=float, default=120.0)
    ap.add_argument("--min-spikes", type=int, default=25)
    ap.add_argument("--out", default="mvp/out")
    args = ap.parse_args()
    st, sidx, wf, wiidx = load(args.file)
    n = len(sidx); W = 48
    wfm = wf.reshape(-1, W)                       # (n_spikes, 48)
    sb = np.concatenate([[0], sidx])              # per-unit spike offsets (into st)
    wb = np.concatenate([[0], wiidx])             # per-unit spike offsets (into wfm)
    t0 = st[0]; dur = st[-1] - t0
    nb = int(dur // args.block_s)
    # per-unit global waveform
    gmean = [wfm[wb[u]:wb[u + 1]].mean(0) for u in range(n)]
    print(f"units={n} spikes={len(st)} dur={dur/60:.1f}min blocks={nb}@{args.block_s:.0f}s")
    stab_ts, amp_ts, alive_ts = [], [], []
    for b in range(nb):
        lo = t0 + b * args.block_s; hi = lo + args.block_s
        stabs, amps, alive = [], [], 0
        for u in range(n):
            sp = st[sb[u]:sb[u + 1]]
            sel = (sp >= lo) & (sp < hi)
            if sel.sum() < args.min_spikes:
                continue
            idx = np.where(sel)[0] + sb[u]
            wfb = wfm[idx]
            mu = wfb.mean(0)
            stabs.append(corr(mu, gmean[u]))
            amps.append(float(mu.max() - mu.min()))
            alive += 1
        stab_ts.append(np.nanmean(stabs) if stabs else np.nan)
        amp_ts.append(np.nanmean(amps) if amps else np.nan)
        alive_ts.append(alive)
        print(f"  t={(b*args.block_s+args.block_s/2)/60:5.1f}min alive={alive:3d}  stability={stab_ts[-1]:.3f}  amp={amp_ts[-1]:.1f}")
    # session-level: waveform drift = 1 - corr(first-half mean, second-half mean)
    half = t0 + dur / 2
    drifts = []
    for u in range(n):
        sp = st[sb[u]:sb[u + 1]]
        if len(sp) < 100:
            continue
        idx = np.arange(sb[u], sb[u + 1])
        m1 = wfm[idx[sp < half]].mean(0); m2 = wfm[idx[sp >= half]].mean(0)
        drifts.append(1 - corr(m1, m2))
    print(f"\nwaveform drift (1-corr first/second half): n={len(drifts)} median={np.nanmedian(drifts):.3f} max={np.nanmax(drifts):.3f}")
    os.makedirs(args.out, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    xs = [(b * args.block_s + args.block_s / 2) / 60 for b in range(nb)]
    ax[0].plot(xs, stab_ts, "o-"); ax[0].set_title("mean waveform stability"); ax[0].set_ylim(0, 1.05)
    ax[1].plot(xs, amp_ts, "o-"); ax[1].set_title("mean waveform amplitude (uV)")
    ax[2].plot(xs, alive_ts, "o-"); ax[2].set_title("units with >=min spikes")
    for a in ax:
        a.set_xlabel("time in session (min)"
    )
    fig.tight_layout()
    out = os.path.join(args.out, "waveform_degradation.png"); fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
