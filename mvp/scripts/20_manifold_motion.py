#!/usr/bin/env python3
"""20 - Manifold motion (Idea 7). Is the per-block CONDITION manifold moving RIGIDLY (rotation+translation)
or DEFORMING? Does the deformable part LEAD the decoder R2 drop? Reach vs rest subspace drift.

Condition manifold per block = the (K x d) matrix of mean standardised firing per movement-direction bin.
Motion models block b-1 -> b:  identity | rigid (orthogonal Procrustes) | affine (general linear).
Normalised residuals: r_id, r_rigid, r_aff  (rel. to the spread of the target set).
  r_rigid ~ r_aff << r_id  -> RIGID   ;   r_aff << r_rigid -> DEFORMATION.
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


def nrm(a):
    return float(np.linalg.norm(a))


def row_basis(M):
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt.T


def mean_angle(B0, Bb):
    s = np.linalg.svd(B0.T @ Bb, compute_uv=False)
    return float(np.degrees(np.mean(np.arccos(np.clip(s, -1, 1)))))


def procrustes(X, Y):
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    U, _, Vt = np.linalg.svd(Yc.T @ Xc)
    R = U @ Vt
    return R, Y.mean(0) - X.mean(0) @ R


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
    RID, RRIG, RAFF, ANG_R, ANG_S = [], [], [], [], []
    rho_lead = []
    for p in files:
        try:
            X, vel, ts = load(p)
        except Exception as exc:
            print("SKIP", os.path.basename(p), exc); continue
        X = exp_filt(X); keep = ~np.isnan(vel).any(1); X, vel, ts = X[keep], vel[keep], ts[keep]
        Xs = ((X - X.mean(0)) / (X.std(0) + 1e-6)).astype(np.float32)
        speed = np.linalg.norm(vel, axis=1)
        moving = speed > np.nanpercentile(speed, 60); rest = speed < np.nanpercentile(speed, 20)
        dirbin = np.full(len(ts), -1)
        ang = np.degrees(np.arctan2(vel[moving, 1], vel[moving, 0]))
        dirbin[moving] = np.clip(np.digitize(ang, np.linspace(-180, 180, K + 1)) - 1, 0, K - 1)
        t_start = ts[0]; nb = int((ts[-1] - t_start) // args.block_s)
        Cs, Rs, r2s = [], [], []; W0 = None
        for b in range(nb):
            lo = t_start + b * args.block_s; hi = lo + args.block_s; tm = (ts >= lo) & (ts < hi)
            if tm.sum() < 400:
                continue
            Cb = np.zeros((K, Xs.shape[1]), np.float32)
            for k in range(K):
                sm = tm & (dirbin == k)
                if sm.sum() >= 30:
                    Cb[k] = Xs[sm].mean(0)
            if W0 is None:
                W0 = Ridge(alpha=10.0).fit(Xs[tm], vel[tm]).coef_
            pr = Xs[tm] @ W0.T
            r2s.append(1.0 - ((pr - vel[tm]) ** 2).sum() / (((vel[tm] - vel[tm].mean(0)) ** 2).sum() + 1e-12))
            rm = tm & rest; Rb = None
            if rm.sum() >= 200:
                Xr = Xs[rm] - Xs[rm].mean(0); _, _, Vt = np.linalg.svd(Xr, full_matrices=False)
                Rb = Vt[:min(6, Vt.shape[0])].T
            Cs.append(Cb); Rs.append(Rb)
        if len(Cs) < 3:
            continue
        rid_l, rrig_l, raff_l = [], [], []
        for b in range(1, len(Cs)):
            Xc, Yc = Cs[b - 1], Cs[b]; den = nrm(Yc - Yc.mean(0)) + 1e-12
            rid_l.append(nrm(Xc - Yc) / den)
            R, t = procrustes(Xc, Yc); rrig_l.append(nrm(Xc @ R + t - Yc) / den)
            Xa = np.c_[Xc, np.ones(len(Xc))]
            A, _, _, _ = np.linalg.lstsq(Xa, Yc, rcond=None)
            raff_l.append(nrm(Xa @ A - Yc) / den)
        B0r = row_basis(Cs[0]); B0s = row_basis(Rs[0]) if Rs[0] is not None else None
        ang_r = [mean_angle(B0r, row_basis(Cb)) for Cb in Cs]
        ang_s = [mean_angle(B0s, row_basis(Rb)) for Rb in Rs if (Rb is not None and B0s is not None)]
        RID += rid_l; RRIG += rrig_l; RAFF += raff_l
        ANG_R.append(ang_r[-1]); ANG_S.append(ang_s[-1] if ang_s else np.nan)
        if len(rrig_l) >= 4:
            rho_lead.append(spearmanr(rrig_l, np.asarray(r2s[1:])[:len(rrig_l)])[0])
        print(f"  {os.path.basename(p).split('ses-')[1].split('_')[0]}: nb={len(Cs)} "
              f"r_id={np.mean(rid_l):.2f} r_rigid={np.mean(rrig_l):.2f} r_aff={np.mean(raff_l):.2f} "
              f"reachAngle={ang_r[-1]:.0f} restAngle={ang_s[-1] if ang_s else float('nan'):.0f}")
    print(f"\nN={len(ANG_R)} sessions")
    print(f"  motion residuals (mean over blocks): identity={np.mean(RID):.2f}  rigid={np.mean(RRIG):.2f}  affine={np.mean(RAFF):.2f}")
    print(f"    -> rigid explains {100*(1-np.mean(RRIG)/np.mean(RID)):.0f}% of identity error; "
          f"affine adds {100*(1-np.mean(RAFF)/np.mean(RRIG)):.0f}% over rigid")
    print(f"  subspace drift vs block0 (deg): reach={np.nanmean(ANG_R):.1f}  rest={np.nanmean(ANG_S):.1f}")
    if rho_lead:
        print(f"  LEAD/LAG: corr(rigid-residual, decoder R2) rho={np.nanmean(rho_lead):+.3f} "
              f"(negative => more deformation as R2 falls)")
    print(f"\n[resources] {time.time()-t0w:.1f}s peakRSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")


if __name__ == "__main__":
    main()
