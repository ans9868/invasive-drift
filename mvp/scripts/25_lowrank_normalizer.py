#!/usr/bin/env python3
"""25 - LOW-RANK NORMALIZER before the decoder (within session).

Protocol (BCI-style, honest):
  BURN-IN = first 20% of the session -> fixes the scaler (m0,s0), the FROZEN decoder, and the reference
  landmark set C0 (mean state per movement direction) + reference moments.
  ONLINE  = sliding windows over the remaining 80%. Each window's normalizer is estimated LABEL-FREE from
  that window's own structure, applied to the features, then the FROZEN decoder decodes.

Ladder:
  frozen                              (no correction; the thing to beat)
  full-rank landmark Procrustes       (21's failure mode: d-dim solution on K=8 points)
  low-rank landmark Procrustes k=2/3/5/7  (restricted to the landmark span)
  A1 moment z-score match             (label-free, per-unit)
  A2 low-rank covariance match k=5    (label-free, top-k PCs)
  refit                               (fit decoder on the window itself = UPPER BOUND)
Report: mean test R2, headroom recovered = (method - frozen)/(refit - frozen), and a LATENCY variant
(landmarks from the first 25% of each window, decode the remaining 75%).
"""
import argparse, glob, os, sys, time, resource
import numpy as np, h5py
from scipy.signal import lfilter
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


def r2(pred, y):
    return 1.0 - ((pred - y) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-12)


def procrustes(A, B):
    Ac = A - A.mean(0); Bc = B - B.mean(0)
    U, _, Vt = np.linalg.svd(Bc.T @ Ac)
    return U @ Vt, A.mean(0), B.mean(0)


def centroids(Z, hb, kmask):
    C = np.zeros((K, Z.shape[1]), np.float32); ok = np.zeros(K, bool)
    for k in range(K):
        sm = hb & (kmask == k)
        if sm.sum() >= 20:
            C[k] = Z[sm].mean(0); ok[k] = True
    return C, ok


def main():
    t0w = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/perich/sub-C")
    ap.add_argument("--burnin", type=float, default=0.2)
    ap.add_argument("--nwin", type=int, default=8)
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data, "*.nwb")))
    if args.n:
        files = files[:args.n]
    KS = [2, 3, 5, 7]
    METH = ["frozen", "full_rank"] + [f"lr{k}" for k in KS] + ["A1_moment", "A2_lowcov", "refit"]
    ACC = {m: [] for m in METH}; HEAD = {m: [] for m in METH}; LAT = {m: [] for m in ("frozen", "lr5")}
    for p in files:
        try:
            X, vel, ts = load(p)
        except Exception as exc:
            print("SKIP", os.path.basename(p), exc); continue
        X = exp_filt(X); keep = ~np.isnan(vel).any(1); X, vel, ts = X[keep], vel[keep], ts[keep]
        t0, dur = ts[0], ts[-1] - ts[0]
        bm = ts < (t0 + args.burnin * dur)
        if bm.sum() < 1500:
            continue
        m0 = X[bm].mean(0); s0 = X[bm].std(0) + 1e-6
        Z = ((X - m0) / s0).astype(np.float32)
        W = Ridge(alpha=10.0).fit(Z[bm], vel[bm]).coef_
        speed = np.linalg.norm(vel, axis=1); moving = speed > np.nanpercentile(speed, 60)
        kmask = np.full(len(ts), -1)
        ang = np.degrees(np.arctan2(vel[moving, 1], vel[moving, 0]))
        kmask[moving] = np.clip(np.digitize(ang, np.linspace(-180, 180, K + 1)) - 1, 0, K - 1)
        C0, ok0 = centroids(Z, bm, kmask)
        mu0 = Z[bm].mean(0); sd0 = Z[bm].std(0) + 1e-6
        U0 = Z[bm] - mu0; _, _, Vtm = np.linalg.svd(U0, full_matrices=False)
        Pmom = Vtm[:5].T
        U0p = U0 @ Pmom; C0c = np.cov(U0p.T) + 1e-6 * np.eye(5)
        e0, V0 = np.linalg.eigh(C0c)
        test = ~bm; tstart = t0 + args.burnin * dur; tlen = (ts[-1] - tstart) / args.nwin
        acc = {m: [] for m in METH}; latacc = {"frozen": [], "lr5": []}
        for w in range(args.nwin):
            lo = tstart + w * tlen; hi = lo + tlen; sel = test & (ts >= lo) & (ts < hi)
            if sel.sum() < 400:
                continue
            Zw, vw = Z[sel], vel[sel]
            dec = lambda zz: zz @ W.T
            acc["frozen"].append(r2(dec(Zw), vw))
            acc["refit"].append(r2(Ridge(alpha=10.0).fit(Zw, vw).predict(Zw), vw))
            Cw, okw = centroids(Z, sel, kmask)
            com = ok0 & okw
            if com.sum() >= 3:
                A, B = Cw[com], C0[com]
                Rf, af, bf = procrustes(A, B)                    # (d,d) full-rank
                acc["full_rank"].append(r2(dec(Zw @ Rf + bf), vw))
                Ac = B - B.mean(0); _, _, VtA = np.linalg.svd(Ac, full_matrices=False)
                for kk in KS:
                    q = min(kk, com.sum() - 1, VtA.shape[0])
                    if q < 1:
                        continue
                    Plm = VtA[:q].T
                    Rk, ab, bb = procrustes(A @ Plm, B @ Plm)
                    Uz = Zw @ Plm
                    Zk = Zw - Uz @ Plm.T + ((Uz - ab) @ Rk + bb) @ Plm.T
                    acc[f"lr{kk}"].append(r2(dec(Zk), vw))
            Za = (Zw - Zw.mean(0)) / (Zw.std(0) + 1e-6) * sd0 + mu0
            acc["A1_moment"].append(r2(dec(Za), vw))
            Uw = (Zw - mu0) @ Pmom; Cwc = np.cov(Uw.T) + 1e-6 * np.eye(5)
            ew_, Vw_ = np.linalg.eigh(Cwc)
            Mw = (Vw_ @ np.diag(ew_ ** -0.5) @ Vw_.T) @ (V0 @ np.diag(e0 ** 0.5) @ V0.T)
            Ucorr = Uw @ Mw
            Zac = Zw + (Ucorr - Uw) @ Pmom.T
            acc["A2_lowcov"].append(r2(dec(Zac), vw))
            # latency variant: landmarks from first 25% of the window, decode last 75%
            q = int(0.25 * sel.sum()); idx = np.where(sel)[0]; cut = idx[0] + q
            if q >= 200 and (idx[-1] - cut) >= 200:
                lsel = np.zeros(len(ts), bool); lsel[idx[:q]] = True; dsel = np.zeros(len(ts), bool); dsel[idx[q:]] = True
                Zd, vd = Z[dsel], vel[dsel]
                latacc["frozen"].append(r2(dec(Zd), vd))
                Cl, okl = centroids(Z, lsel, kmask); com2 = ok0 & okl
                if com2.sum() >= 3:
                    A2_, B2_ = Cl[com2], C0[com2]; Ac2 = B2_ - B2_.mean(0)
                    _, _, Vt2 = np.linalg.svd(Ac2, full_matrices=False); q2 = min(5, com2.sum() - 1, Vt2.shape[0])
                    P2 = Vt2[:q2].T; R2_, ab2, bb2 = procrustes(A2_ @ P2, B2_ @ P2)
                    U2 = Zd @ P2; Zd2 = Zd - U2 @ P2.T + ((U2 - ab2) @ R2_ + bb2) @ P2.T
                    latacc["lr5"].append(r2(dec(Zd2), vd))
        if not acc["frozen"]:
            continue
        for m in METH:
            if acc[m]:
                ACC[m].append(float(np.nanmean(acc[m])))
                fr = np.nanmean(acc["frozen"]); rf = np.nanmean(acc["refit"])
                HEAD[m].append((np.nanmean(acc[m]) - fr) / (rf - fr + 1e-12))
        for m in ("frozen", "lr5"):
            if latacc[m]:
                LAT[m].append(float(np.nanmean(latacc[m])))
        print(f"  {os.path.basename(p).split('ses-')[1].split('_')[0]}: frozen={np.nanmean(acc['frozen']):.3f} "
              f"refit={np.nanmean(acc['refit']):.3f} | full={np.nanmean(acc['full_rank']) if acc['full_rank'] else float('nan'):.3f} "
              f"lr5={np.nanmean(acc['lr5']) if acc['lr5'] else float('nan'):.3f} A1={np.nanmean(acc['A1_moment']):.3f} A2={np.nanmean(acc['A2_lowcov']):.3f}")
    print(f"\nN={len(ACC['frozen'])} sessions")
    print(f"{'method':10s} {'meanR2':>8s} {'headroomRecov':>14s}")
    for m in METH:
        if ACC[m]:
            print(f"{m:10s} {np.nanmean(ACC[m]):>8.3f} {np.nanmean(HEAD[m]):>14.2f}")
    print(f"\nLATENCY (landmarks from first 25% of window, decode last 75%):")
    for m in ("frozen", "lr5"):
        if LAT[m]:
            print(f"  {m:6s} meanR2={np.nanmean(LAT[m]):.3f}")
    print(f"\n[resources] {time.time()-t0w:.1f}s peakRSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")


if __name__ == "__main__":
    main()
