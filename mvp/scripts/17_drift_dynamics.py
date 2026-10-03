#!/usr/bin/env python3
"""Drift dynamics: is representational drift structured / predictable?
Per session: per-block encoder weights -> (i) low-dimensionality (PCA), (ii) smoothness (angle),
(iii) extrapolation predictability of the next block's decoder vs persistence."""
import argparse, glob, os, sys
import numpy as np, h5py
from scipy.signal import lfilter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.linear_model import Ridge

BIN_MS = 20.0; TAU_MS = 240.0


def load(path):
    with h5py.File(path, "r") as h:
        st = h["units/spike_times"][:]; idx = h["units/spike_times_index"][:]
        b = np.concatenate([[0], idx]); n = len(idx)
        spies = [st[b[u]:b[u + 1]] for u in range(n)]
        vg = h["processing/behavior/Velocity/cursor_vel"]
        ts = vg["timestamps"][:]; vel = vg["data"][:].astype(np.float32)
    bs = BIN_MS / 1000.0; edges = np.concatenate([[ts[0] - bs], ts])
    X = np.zeros((len(ts), n), np.float32)
    for u in range(n):
        X[:, u] = np.histogram(spies[u], bins=edges)[0]
    return X, vel, ts


def exp_filt(x, tau_ms=TAU_MS, bin_ms=BIN_MS):
    t = np.arange(0.0, tau_ms, bin_ms); k = np.exp(-t / tau_ms); k /= k.sum()
    return lfilter(k, [1.0], x, axis=0).astype(np.float32)


def cosv(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/perich/sub-C")
    ap.add_argument("--block-s", type=float, default=120.0)
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data, "*.nwb")))
    if args.n:
        files = files[:args.n]
    pc1, sim, drift_last, smooth, extrap, persist_cos = [], [], [], [], [], []
    for p in files:
        try:
            X, vel, ts = load(p)
        except Exception as exc:
            print("SKIP", os.path.basename(p), exc); continue
        X = exp_filt(X); keep = ~np.isnan(vel).any(1); X, vel, ts = X[keep], vel[keep], ts[keep]
        t0 = ts[0]; nb = int((ts[-1] - t0) // args.block_s)
        W = []
        for b in range(nb):
            lo = t0 + b * args.block_s; hi = lo + args.block_s; sel = (ts >= lo) & (ts < hi)
            if sel.sum() < 400:
                continue
            m, s = X[sel].mean(0), X[sel].std(0) + 1e-6
            W.append(Ridge(alpha=10.0).fit((X[sel] - m) / s, vel[sel]).coef_.ravel())
        if len(W) < 4:
            continue
        W = np.array(W)
        # (i) low-dimensionality of the drift trajectory
        Wc = W - W.mean(0)
        sv = np.linalg.svd(Wc, compute_uv=False)
        pc1.append(float(sv[0] ** 2 / (sv ** 2).sum()))
        # (ii) smoothness (consecutive cosine)
        cons = [cosv(W[i], W[i + 1]) for i in range(len(W) - 1)]
        smooth.append(float(np.mean(cons)))
        # drift: cosine(first,last)
        drift_last.append(cosv(W[0], W[-1]))
        # (iii) extrapolation: predict W[b+1] = W[b] + (W[b]-W[b-1]); vs persistence W[b]
        ex, pe = [], []
        for b in range(1, len(W) - 1):
            ep = W[b] + (W[b] - W[b - 1])
            ex.append(cosv(ep, W[b + 1])); pe.append(cosv(W[b], W[b + 1]))
        extrap.append(float(np.mean(ex))); persist_cos.append(float(np.mean(pe)))
        sess = os.path.basename(p).split("ses-")[1].split("_")[0]
        print(f"  {sess}: nb={len(W)} PC1var={pc1[-1]:.2f} consecCos={smooth[-1]:.3f} "
              f"driftCos(first,last)={drift_last[-1]:.3f} extrap={extrap[-1]:.3f} persist={persist_cos[-1]:.3f}")
    print(f"\nN={len(pc1)} sessions")
    print(f"  PC1 variance fraction (low-D?)          = {np.nanmean(pc1):.2f}  (1.0=1-D drift, ~0.1=isotropic random)")
    print(f"  mean consecutive cosine (smoothness)     = {np.nanmean(smooth):.3f}  (1.0=white noise, lower=smoother drift)")
    print(f"  mean drift cos(first,last)               = {np.nanmean(drift_last):.3f}  (1.0=no drift)")
    print(f"  extrapolation cos(next) vs persistence   = {np.nanmean(extrap):.3f} vs {np.nanmean(persist_cos):.3f}")


if __name__ == "__main__":
    main()
