#!/usr/bin/env python3
"""Cross-session waveform instrument: match units across days by waveform similarity; track how many
persist and how their amplitude/stability evolve. Direct evidence for unit loss vs identity."""
import argparse, glob, os
import numpy as np, h5py

W = 48


def load_units(path):
    with h5py.File(path, "r") as h:
        st = h["units/spike_times"][:]; sidx = h["units/spike_times_index"][:]
        wf = h["units/waveforms"][:, 0]; wiidx = h["units/waveforms_index_index"][:]
    wfm = wf.reshape(-1, W); wb = np.concatenate([[0], wiidx])
    means, amps, rates = [], [], []
    for u in range(len(sidx)):
        m = wfm[wb[u]:wb[u + 1]].mean(0)
        means.append(m); amps.append(float(m.max() - m.min()))
        start = sidx[u - 1] if u > 0 else 0
        rates.append(float(sidx[u] - start))
    return np.array(means), np.array(amps), np.array(rates)


def match(A, B, thresh):
    An = A - A.mean(1, keepdims=True); Bn = B - B.mean(1, keepdims=True)
    An /= np.linalg.norm(An, axis=1, keepdims=True) + 1e-9
    Bn /= np.linalg.norm(Bn, axis=1, keepdims=True) + 1e-9
    C = An @ Bn.T
    pairs = []; Cc = C.copy()
    while True:
        i, j = np.unravel_index(np.argmax(Cc), Cc.shape)
        if Cc[i, j] < thresh:
            break
        pairs.append((i, j, float(Cc[i, j])))
        Cc[i, :] = -1; Cc[:, j] = -1
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/perich/sub-C")
    ap.add_argument("--thresh", type=float, default=0.90)
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data, "*.nwb")))
    if args.n:
        files = files[:args.n]
    sess_means = []; sess_amp = []; dates = []; labels = []
    for p in files:
        try:
            M, A, R = load_units(p)
        except Exception as exc:
            print("SKIP", os.path.basename(p), exc); continue
        d = os.path.basename(p).split("ses-")[1].split("_")[0]
        sess_means.append(M); sess_amp.append(A); labels.append(d)
        print(f"  {d}: units={M.shape[0]} amp_median={np.median(A):.0f}")
    print(f"\nsessions={len(sess_means)}  match threshold={args.thresh}")
    print(f"{'pair':24s} {'nA':>4s} {'nB':>4s} {'matched':>8s} {'fracA':>7s} {'med_corr':>9s}")
    # consecutive-session matching
    prev_idx = np.arange(len(sess_means[0]))
    chain = list(prev_idx)
    for k in range(len(sess_means) - 1):
        A = sess_means[k]; B = sess_means[k + 1]
        pairs = match(A, B, args.thresh)
        corrs = [c for _, _, c in pairs]
        print(f"{labels[k]+'->'+labels[k+1]:24s} {A.shape[0]:4d} {B.shape[0]:4d} {len(pairs):8d} {len(pairs)/A.shape[0]:7.2f} {np.median(corrs) if corrs else 0:9.3f}")
    # full-chain persistence: units in session 0 matched all the way through
    print("\n(tracked-chain persistence needs full union-find; above = per-consecutive-pair)")


if __name__ == "__main__":
    main()
