#!/usr/bin/env python3
"""21 - Repeated-movement drift (the 'piano'). Units = keys; a movement DIRECTION = a small piece; a
TIME WINDOW = one performance. Track the per-direction population CHORD over windows and ask:
  (1) how far does the chord move?  (2) is the move a SHARED mode (all directions wobble together) or spread?
  (3) can performance w be predicted from the earlier ones (persistence vs rotation)?
  (4) can we RE-ALIGN later performances back to the 1st using the directions as landmarks (unsupervised)?
"""
import argparse, glob, os, sys, time, resource
import numpy as np, h5py
from scipy.signal import lfilter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.linear_model import Ridge

BIN_MS = 20.0; TAU_MS = 240.0; K = 8; NW = 6


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
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def procrustes(X, Y):
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    U, _, Vt = np.linalg.svd(Yc.T @ Xc)
    R = U @ Vt
    return R, Y.mean(0) - X.mean(0) @ R


def main():
    t0w = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/perich/sub-C")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data, "*.nwb")))
    if args.n:
        files = files[:args.n]
    DRIFT, PC1S, CP, CR = [], [], [], []
    R2_UN, R2_AL, R2_REF = [], [], []
    for p in files:
        try:
            X, vel, ts = load(p)
        except Exception as exc:
            print("SKIP", os.path.basename(p), exc); continue
        X = exp_filt(X); keep = ~np.isnan(vel).any(1); X, vel, ts = X[keep], vel[keep], ts[keep]
        Xs = ((X - X.mean(0)) / (X.std(0) + 1e-6)).astype(np.float32)
        speed = np.linalg.norm(vel, axis=1); moving = speed > np.nanpercentile(speed, 60)
        dirbin = np.full(len(ts), -1)
        ang = np.degrees(np.arctan2(vel[moving, 1], vel[moving, 0]))
        dirbin[moving] = np.clip(np.digitize(ang, np.linspace(-180, 180, K + 1)) - 1, 0, K - 1)
        t0 = ts[0]; dur = ts[-1] - t0
        # chords per window (K x d), and whether each direction has enough samples
        Ch, valid, Wd = [], [], []
        W0 = None
        for w in range(NW):
            lo = t0 + w * dur / NW; hi = t0 + (w + 1) * dur / NW; tm = (ts >= lo) & (ts < hi)
            Cw = np.zeros((K, Xs.shape[1]), np.float32); ok = np.zeros(K, bool)
            for k in range(K):
                sm = tm & (dirbin == k)
                if sm.sum() >= 20:
                    Cw[k] = Xs[sm].mean(0); ok[k] = True
            if W0 is None:
                W0 = Ridge(alpha=10.0).fit(Xs[tm], vel[tm]).coef_
            Ch.append(Cw); valid.append(ok); Wd.append((tm, Xs[tm], vel[tm]))
        if sum(v.sum() for v in valid) < 3 * K:
            continue
        # (1) drift magnitude + (2) shared mode of chord changes vs window 0
        dr = []; vecs = []
        for w in range(1, NW):
            m = valid[w] & valid[0]
            d = (Ch[w][m] - Ch[0][m])
            dr.append(float(np.linalg.norm(d) / (np.linalg.norm(Ch[0][m]) + 1e-12)))
            vecs.append(d.ravel())
        Vm = np.array(vecs); Vc = Vm - Vm.mean(0)
        _, S, _ = np.linalg.svd(Vc, full_matrices=False)
        pc1 = float(S[0] ** 2 / (S ** 2).sum())
        # (3) predict performance w from history: persistence vs rotation
        cp, cr = [], []
        for w in range(1, NW):
            cp.append(cosv(Ch[w - 1], Ch[w]))
            if w >= 2:
                Xh = np.vstack([Ch[i] for i in range(0, w - 1)])
                Yh = np.vstack([Ch[i + 1] for i in range(0, w - 1)])
                R, t = procrustes(Xh, Yh); pr = Ch[w - 1] @ R + t
            else:
                pr = Ch[w - 1]
            cr.append(cosv(pr, Ch[w]))
        # (4) unsupervised realignment of later windows onto window 0 (directions as landmarks)
        for w in range(1, NW):
            tm, Xw, Vw = Wd[w]
            pr_un = Xw @ W0.T
            R, t = procrustes(Ch[w], Ch[0])
            pr_al = (Xw @ R + t) @ W0.T
            Wr = Ridge(alpha=10.0).fit(Xw, Vw).coef_
            pr_rf = Xw @ Wr.T
            for pr, acc in ((pr_un, R2_UN), (pr_al, R2_AL), (pr_rf, R2_REF)):
                acc.append(1.0 - ((pr - Vw) ** 2).sum() / (((Vw - Vw.mean(0)) ** 2).sum() + 1e-12))
        DRIFT += dr; PC1S.append(pc1); CP += cp; CR += cr
        print(f"  {os.path.basename(p).split('ses-')[1].split('_')[0]}: drift={np.mean(dr):.2f} PC1shared={pc1:.2f} "
              f"pred persist={np.mean(cp):.3f} rot={np.mean(cr):.3f} | R2 un={np.mean(R2_UN[-5:]):.2f} "
              f"aligned={np.mean(R2_AL[-5:]):.2f} refit={np.mean(R2_REF[-5:]):.2f}")
    print(f"\nN={len(PC1S)} sessions (K={K} dirs, NW={NW} windows)")
    print(f"(1) chord drift per window (rel.)      = {np.mean(DRIFT):.2f}")
    print(f"(2) SHARED mode: PC1 var fraction      = {np.mean(PC1S):.2f}  (>>1/{K} => one shared wobble)")
    print(f"(3) predict next performance:  persistence cos={np.mean(CP):.3f}  rotation cos={np.mean(CR):.3f}")
    print(f"(4) decode: unaligned R2={np.mean(R2_UN):.3f}  landmark-aligned={np.mean(R2_AL):.3f}  refit(upper)={np.mean(R2_REF):.3f}")
    print(f"\n[resources] {time.time()-t0w:.1f}s peakRSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")


if __name__ == "__main__":
    main()
