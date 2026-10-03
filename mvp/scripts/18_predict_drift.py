#!/usr/bin/env python3
"""Can we predict the NEXT block's decoder from the drift — beating persistence?
Models: persistence, trend, Procrustes-rotation (global + low-rank), VAR(ridge, PC), shrink-to-mean,
and a state-input model (predict the drift STEP from behavior/neural state).
Score: cosine to true next weights, and FUNCTIONAL R2 of decoding the next block."""
import argparse, glob, os, sys
import numpy as np, h5py
from scipy.signal import lfilter
from numpy.linalg import svd
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


def procrustes(X, Y):
    """orthogonal R minimizing ||X R - Y|| (rows = samples)"""
    U, _, Vt = svd(Y.T @ X)
    return U @ Vt


def r2(p, y):
    return float(1.0 - ((p - y) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-12))


MODELS = ["persist", "trend", "rot", "rot_lr", "var", "shrink", "state"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/perich/sub-C")
    ap.add_argument("--block-s", type=float, default=120.0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data, "*.nwb")))
    if args.n:
        files = files[:args.n]
    cos = {m: [] for m in MODELS}; fun = {m: [] for m in MODELS}
    for p in files:
        try:
            X, vel, ts = load(p)
        except Exception as exc:
            print("SKIP", os.path.basename(p), exc); continue
        X = exp_filt(X); keep = ~np.isnan(vel).any(1); X, vel, ts = X[keep], vel[keep], ts[keep]
        t0 = ts[0]; nb = int((ts[-1] - t0) // args.block_s)
        W, MB, SB, XB, VB = [], [], [], [], []
        for bi in range(nb):
            lo = t0 + bi * args.block_s; hi = lo + args.block_s; sel = (ts >= lo) & (ts < hi)
            if sel.sum() < 400:
                continue
            m, s = X[sel].mean(0), X[sel].std(0) + 1e-6
            W.append(Ridge(alpha=10.0).fit((X[sel] - m) / s, vel[sel]).coef_.ravel())
            MB.append(m); SB.append(s); XB.append(X[sel]); VB.append(vel[sel])
        if len(W) < 4:
            continue
        W = np.array(W); mu = W[0]
        # state features per block (low-dim)
        state = np.array([[bi / nb, float(XB[bi].mean()), float(XB[bi].std()),
                           float(np.abs(VB[bi]).mean())] for bi in range(len(XB))])
        for b in range(1, len(W) - 1):
            truth = W[b + 1]
            preds = {}
            preds["persist"] = W[b]
            preds["trend"] = 2 * W[b] - W[b - 1]
            # global orthogonal rotation from history
            R = procrustes(W[:b], W[1:b + 1]); preds["rot"] = W[b] @ R
            # low-rank: project history to top-k PCs, rotate there
            Wc = W[:b] - W[:b].mean(0)
            _, _, Vt = svd(Wc, full_matrices=False); P = Vt[:args.k].T  # dim x k
            Rk = procrustes(W[:b] @ P, W[1:b + 1] @ P)
            preds["rot_lr"] = (W[b] @ P @ Rk) @ P.T + (W[:b].mean(0) - W[:b].mean(0) @ P @ P.T)
            # VAR in PC space: W[i+1]_pc ~ A W[i]_pc
            A = Ridge(alpha=1.0).fit(W[:b] @ P, W[1:b + 1] @ P).coef_
            preds["var"] = (W[b] @ P @ A.T) @ P.T + (W[:b].mean(0) - W[:b].mean(0) @ P @ P.T)
            # shrink to running mean
            preds["shrink"] = 0.7 * W[b] + 0.3 * W[:b + 1].mean(0)
            # state: predict residual step from state (in PC space)
            steps = (W[1:b + 1] - W[:b]) @ P
            Am = Ridge(alpha=1.0).fit(state[:b], steps).coef_
            preds["state"] = W[b] + state[b] @ Am.T @ P.T
            for m in MODELS:
                cos[m].append(cosv(preds[m], truth))
                # functional: decode NEXT block with predicted decoder (block-b scaling)
                Xn = (XB[b + 1] - MB[b]) / SB[b]
                fun[m].append(r2(Xn @ preds[m], VB[b + 1]))
        print(f"  {os.path.basename(p).split('ses-')[1].split('_')[0]}: nb={len(W)}")
    print(f"\nN={len(cos['persist'])} block-predictions across sessions")
    print(f"{'model':8s}  {'cos(next w)':>11s}  {'functional R2':>13s}")
    for m in MODELS:
        print(f"{m:8s}  {np.nanmean(cos[m]):>11.3f}  {np.nanmean(fun[m]):>13.3f}")


if __name__ == "__main__":
    main()
