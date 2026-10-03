#!/usr/bin/env python3
"""23 - LOW-RANK GATE + VOLATILITY (entropy) as a leading indicator.

Unit-free: per-block decoders are fit in the session-standardised basis and probed on a FIXED set of
per-direction mean states -> velocity field V_b (2K=16 vector). Drift step dV_b = V_{b+1} - V_b.

(A) LOW-RANK?: SVD of the steps dV. Report PC1/PC2 variance fraction and the rank needed for 80%.
              (high PC1/PC2 => the drift is a low-rank transform => a normalizer has a chance)
(B) VOLATILITY (entropy proxy): ||dV_b|| (and its detrended part). Is it time-varying (vs block index)?
              Is it predictable (lag-1 autocorr)?
(C) DOES VOLATILITY LEAD THE R2 DROP?: per-block R2 (block-0 decoder on block b). Correlate volatility
              with next-block R2 and with the R2 *change*. Also weight-space dV as a secondary check.
"""
import argparse, glob, os, sys, time, resource
import numpy as np, h5py
from scipy.signal import lfilter
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.linear_model import Ridge

BIN_MS = 20.0; TAU_MS = 240.0; K = 8


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


def sp(a, b):
    if len(a) < 4 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    return float(spearmanr(a, b)[0])


def main():
    t0w = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/perich/sub-C")
    ap.add_argument("--block-s", type=float, default=120.0)
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data, "*.nwb")))
    if args.n:
        files = files[:args.n]
    PC1v, PC2v, R80 = [], [], []
    PC1w = []
    VOLTIME, VOLAR1 = [], []
    LEAD_NEXT, CONTEMP, LEAD_DR2 = [], [], []
    for p in files:
        try:
            X, vel, ts = load(p)
        except Exception as exc:
            print("SKIP", os.path.basename(p), exc); continue
        X = exp_filt(X); keep = ~np.isnan(vel).any(1); X, vel, ts = X[keep], vel[keep], ts[keep]
        Xs = ((X - X.mean(0)) / (X.std(0) + 1e-6)).astype(np.float32)
        speed = np.linalg.norm(vel, axis=1); moving = speed > np.nanpercentile(speed, 60)
        ang = np.degrees(np.arctan2(vel[moving, 1], vel[moving, 0]))
        bidx = np.clip(np.digitize(ang, np.linspace(-180, 180, K + 1)) - 1, 0, K - 1)
        Xm = Xs[moving]; state = np.zeros((K, Xs.shape[1]), np.float32); nk = 0
        for k in range(K):
            sel = bidx == k
            if sel.sum() >= 100:
                state[k] = Xm[sel].mean(0); nk += 1
        if nk < 4:
            continue
        t0 = ts[0]; nb = int((ts[-1] - t0) // args.block_s)
        W, V, r2 = [], [], []; W0 = None
        for b in range(nb):
            lo = t0 + b * args.block_s; hi = lo + args.block_s; tm = (ts >= lo) & (ts < hi)
            if tm.sum() < 400:
                continue
            wb = Ridge(alpha=10.0).fit(Xs[tm], vel[tm]).coef_
            W.append(wb); V.append((wb @ state.T).ravel())
            if W0 is None:
                W0 = wb
            pr = Xs[tm] @ W0.T
            r2.append(1.0 - ((pr - vel[tm]) ** 2).sum() / (((vel[tm] - vel[tm].mean(0)) ** 2).sum() + 1e-12))
        if len(W) < 4:
            continue
        V = np.array(V); dV = np.diff(V, axis=0)
        Vc = dV - dV.mean(0); _, S, _ = np.linalg.svd(Vc, full_matrices=False)
        frac = S ** 2 / (S ** 2).sum()
        PC1v.append(float(frac[0])); PC2v.append(float(frac[:2].sum()))
        R80.append(float(np.searchsorted(np.cumsum(frac), 0.8) + 1))
        Wm = np.array(W); dW = np.diff(Wm, axis=0).reshape(len(W) - 1, -1)
        _, Sw, _ = np.linalg.svd(dW - dW.mean(0), full_matrices=False)
        PC1w.append(float(Sw[0] ** 2 / (Sw ** 2).sum()))
        vol = np.linalg.norm(dV, axis=1)
        VOLTIME.append(sp(np.arange(len(vol)), vol))
        if len(vol) >= 4 and vol[:-1].std() > 0 and vol[1:].std() > 0:
            VOLAR1.append(float(np.corrcoef(vol[:-1], vol[1:])[0, 1]))
        else:
            VOLAR1.append(np.nan)
        r2 = np.array(r2); r2n = r2[1:]                      # align r2 with vol (b->b+1)
        n = min(len(vol), len(r2n))
        if n >= 4:
            LEAD_NEXT.append(sp(vol[:n], r2n[:n]))
            CONTEMP.append(sp(vol[:n], r2[1:][:n]))
            dr2 = np.diff(r2)[:n]
            LEAD_DR2.append(sp(vol[:n], dr2))
        print(f"  {os.path.basename(p).split('ses-')[1].split('_')[0]}: nb={len(W)} "
              f"PC1v={PC1v[-1]:.2f} PC2v={PC2v[-1]:.2f} rank80={R80[-1]:.0f} volAR1={VOLAR1[-1]:+.2f} lead={LEAD_NEXT[-1] if LEAD_NEXT else float('nan'):+.2f}")
    print(f"\nN={len(PC1v)} sessions")
    print(f"(A) LOW-RANK?  velocity-field steps (2K=16 dims): PC1={np.mean(PC1v):.2f}  PC1+2={np.mean(PC2v):.2f}  rank80={np.mean(R80):.1f}/16")
    print(f"               weight-space steps: PC1={np.mean(PC1w):.2f}")
    print(f"(B) VOLATILITY: vs block index rho={np.nanmean(VOLTIME):+.2f} (trend)  lag-1 autocorr={np.nanmean(VOLAR1):+.2f} (predictability)")
    print(f"(C) LEAD/LAG: corr(vol_b, R2_b)   = {np.nanmean(CONTEMP):+.3f}")
    print(f"             corr(vol_b, R2_b+1) = {np.nanmean(LEAD_NEXT):+.3f}  (negative => high volatility -> next R2 falls)")
    print(f"             corr(vol_b, dR2)    = {np.nanmean(LEAD_DR2):+.3f}")
    print(f"\n[resources] {time.time()-t0w:.1f}s peakRSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")


if __name__ == "__main__":
    main()
