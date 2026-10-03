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
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
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
    ap.add_argument("--debug", action="store_true")
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
        vel_mean = vel.mean(0); ss_sess = float(((vel - vel_mean) ** 2).sum())
        for b in range(1, len(W) - 1):
            truth = W[b + 1]
            preds = {}
            preds["persist"] = W[b]
            preds["trend"] = 2 * W[b] - W[b - 1]
            # global orthogonal rotation from history
            R = procrustes(W[:b], W[1:b + 1]); preds["rot"] = W[b] @ R
            # PC subspace of the historical weights (shared by low-rank models)
            mean_h = W[:b].mean(0)
            _, _, Vt = svd(W[:b] - mean_h, full_matrices=False)
            kk = max(1, min(args.k, Vt.shape[0])); P = Vt[:kk].T          # dim x kk
            mat = np.atleast_2d
            to_pc = lambda a: mat(a - mean_h) @ P                        # (n,dim)->(n,kk)
            from_pc = lambda z: (mat(z) @ P.T + mean_h)[0]               # (n,kk)->(dim,)
            # low-rank orthogonal rotation in PC subspace
            Rk = procrustes(to_pc(W[:b]), to_pc(W[1:b + 1]))
            preds["rot_lr"] = from_pc(to_pc(W[b:b + 1]) @ Rk)
            # VAR in PC subspace: z_{i+1} ~ A z_i (least-squares -> deterministic shapes)
            A = np.linalg.lstsq(to_pc(W[:b]), to_pc(W[1:b + 1]), rcond=None)[0]   # (kk,kk)
            preds["var"] = from_pc(to_pc(W[b:b + 1]) @ A.T)
            # shrink to running mean
            preds["shrink"] = 0.7 * W[b] + 0.3 * W[:b + 1].mean(0)
            # state-input: predict the drift STEP (residual) from state features
            steps = to_pc(W[1:b + 1]) - to_pc(W[:b])
            Am = np.linalg.lstsq(mat(state[:b]), steps, rcond=None)[0]            # (4,kk)
            preds["state"] = W[b] + (mat(state[b:b + 1]) @ Am) @ P.T
            if args.debug and b == 1 and p == files[0]:
                print("DBG R sing", np.linalg.svd(R, compute_uv=False)[:3], "||W[b]||",
                      float(np.linalg.norm(W[b])))
                for m in MODELS:
                    print("DBG", m, np.shape(preds[m]), "norm",
                          float(np.linalg.norm(np.asarray(preds[m]).ravel())))
                print("DBG vel sd", float(VB[b + 1].std()), "Xn sd",
                      float(((XB[b + 1] - MB[b]) / SB[b]).std()))
            for m in MODELS:
                pv = np.asarray(preds[m]).ravel()
                if pv.size != truth.size:
                    print("SHAPE_WARN", m, np.shape(preds[m]), truth.shape); continue
                cos[m].append(cosv(pv, truth))
                # functional: decode NEXT block with predicted decoder (block-b scaling)
                Xn = (XB[b + 1] - MB[b]) / SB[b]
                scale = np.linalg.norm(W[b]) / (np.linalg.norm(pv) + 1e-12)
                w2 = (pv * scale).reshape(2, Xn.shape[1])   # predict DIRECTION; keep persistence gain
                err = float(((Xn @ w2.T - VB[b + 1]) ** 2).sum())
                fun[m].append(1.0 - err / (ss_sess + 1e-12))
        print(f"  {os.path.basename(p).split('ses-')[1].split('_')[0]}: nb={len(W)}")
    print(f"\nN={len(cos['persist'])} block-predictions across sessions")
    print(f"{'model':8s}  {'cos(next w)':>11s}  {'mean R2':>8s}  {'med R2':>7s}  {'out|R2|>5':>9s}")
    for m in MODELS:
        r = np.asarray(fun[m], float)
        print(f"{m:8s}  {np.nanmean(cos[m]):>11.3f}  {np.nanmean(r):>8.3f}  "
              f"{np.nanmedian(r):>7.3f}  {np.mean(np.abs(r) > 5):>9.3f}")


if __name__ == "__main__":
    main()
