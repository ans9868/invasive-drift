#!/usr/bin/env python3
"""22 - Pairwise relations: is the GAP between unit i and unit j changing CONSISTENTLY (monotone, same
sign) over repeated performances of the same piece, and can it be EXTRAPOLATED to the next performance?

Per session: split into NW equal windows (performances); per window build each unit's DIRECTION PROFILE
(mean standardised firing for each of K=8 movement directions).
  gap(w,i,j) = ||profile_i - profile_j||   (the 'interval between two keys').
Questions: (1) fraction of pairs with a MONOTONE trend; sign split (widening vs narrowing)?
           (2) can window w's gap be predicted from earlier windows (persistence vs linear extrapolation)?
           (3) same for the TEMPORAL lag between unit pairs (cross-correlation peak lag).
"""
import argparse, glob, os, sys, time, resource
import numpy as np, h5py
from scipy.signal import lfilter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BIN_MS = 20.0; TAU_MS = 240.0; K = 8; NW = 6; MINBIN = 15; LAG = 10


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


def spearman_cols(G):
    """G: (NW, P) -> rho per column (Spearman over windows)."""
    n = G.shape[0]
    rk = np.argsort(np.argsort(G, axis=0), axis=0).astype(float)
    rm = rk - rk.mean(0); x = np.arange(n, dtype=float); xm = x - x.mean()
    return (xm[:, None] * rm).sum(0) / (np.sqrt((xm ** 2).sum()) * np.sqrt((rm ** 2).sum(0)) + 1e-12)


def main():
    t0w = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/perich/sub-C")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--maxpairs", type=int, default=400)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data, "*.nwb")))
    if args.n:
        files = files[:args.n]
    MONO, WIDE, RELCHG, EXT_SK, EXT_R = [], [], [], [], []
    TLAG_MONO, TLAG_ABS = [], []
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
        t0 = ts[0]; dur = ts[-1] - t0; d = Xs.shape[1]
        Pw = np.zeros((NW, K, d), np.float32); ok = np.zeros((NW, K), bool); masks = []
        okunit = np.ones(d, bool)
        for w in range(NW):
            lo = t0 + w * dur / NW; hi = t0 + (w + 1) * dur / NW; tm = (ts >= lo) & (ts < hi)
            masks.append(tm)
            for k in range(K):
                sm = tm & (dirbin == k)
                if sm.sum() >= MINBIN:
                    Pw[w, k] = Xs[sm].mean(0); ok[w, k] = True
            okunit &= ok[w].all()   # keep units with a valid profile in ALL windows & dirs
        if okunit.sum() < 10:
            print("SKIP(units)", os.path.basename(p)); continue
        Pp = Pw[:, :, okunit]                      # (NW, K, du)
        du = Pp.shape[2]
        iu = np.triu_indices(du, 1)
        G = np.empty((NW, len(iu[0])), np.float32)
        for w in range(NW):
            A = Pp[w].T                            # (du, K) unit profiles
            dif = A[:, None, :] - A[None, :, :]
            Dw = np.sqrt((dif ** 2).sum(-1))
            G[w] = Dw[iu]
        rho = spearman_cols(G)
        mono = np.abs(rho) >= 0.8
        MONO.append(float(mono.mean()))
        WIDE.append(float((rho[mono] > 0).mean()) if mono.any() else np.nan)
        RELCHG.append(float(np.median((G[-1] - G[0]) / (G[0] + 1e-12))))
        # extrapolation vs persistence for windows 2..NW-1
        se, sp, ok_r = 0.0, 0.0, []
        for w in range(2, NW):
            true = G[w]; per = G[w - 1]; ext = 2 * G[w - 1] - G[w - 2]
            se += float(((ext - true) ** 2).sum()); sp += float(((per - true) ** 2).sum())
            cc = np.corrcoef(ext, true)[0, 1]
            if np.isfinite(cc):
                ok_r.append(cc)
        EXT_SK.append(1.0 - se / (sp + 1e-12)); EXT_R.append(float(np.mean(ok_r)) if ok_r else np.nan)
        # (3) temporal lag on a subset of pairs (among the top-rate units)
        rate = Xs[moving].mean(0); top = np.argsort(-rate)[:20]
        tiu = np.triu_indices(len(top), 1)
        lagG = np.full((NW, len(tiu[0])), np.nan, np.float32)
        for w in range(NW):
            sub = Xs[masks[w] & moving]
            if len(sub) < 50:
                continue
            sub = sub - sub.mean(0)
            for pi, (a, b) in enumerate(zip(tiu[0], tiu[1])):
                ia, ib = top[a], top[b]
                best, bl = -2.0, 0
                for l in range(-LAG, LAG + 1):
                    if l >= 0:
                        x1, x2 = sub[l:, ia], sub[:len(sub) - l, ib] if l else sub[:, ib]
                    else:
                        x1, x2 = sub[:len(sub) + l, ia], sub[-l:, ib]
                    if len(x1) < 30:
                        continue
                    c = float(x1 @ x2 / (np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-12))
                    if c > best:
                        best, bl = c, l
                lagG[w, pi] = bl
        m = ~np.isnan(lagG).any(0)
        if m.sum() > 5:
            rl = spearman_cols(lagG[:, m])
            TLAG_MONO.append(float((np.abs(rl) >= 0.8).mean()))
            TLAG_ABS.append(float(np.median(np.abs(lagG[-1, m] - lagG[0, m]))))
        print(f"  {os.path.basename(p).split('ses-')[1].split('_')[0]}: du={du} pairs={len(iu[0])} "
              f"monoFrac={MONO[-1]:.2f} wideFrac={WIDE[-1]:.2f} relChg={RELCHG[-1]:+.2f} extSkill={EXT_SK[-1]:+.3f}")
    print(f"\nN={len(MONO)} sessions (K={K} dirs, NW={NW} performances)")
    print(f"(1) monotone pairs (|rho|>=0.8)      = {np.nanmean(MONO):.2f}")
    print(f"    of monotone, fraction WIDENING    = {np.nanmean(WIDE):.2f}  (>0.5 => gaps mostly grow)")
    print(f"    median rel. change gap(w1->wNW)   = {np.nanmean(RELCHG):+.2f}")
    print(f"(2) extrapolate next gap: skill vs persistence = {np.nanmean(EXT_SK):+.3f} (corr {np.nanmean(EXT_R):+.2f})")
    print(f"(3) temporal lag: monotone pairs      = {np.nanmean(TLAG_MONO):.2f} ; median |lag change| = {np.nanmean(TLAG_ABS):.1f} bins")
    print(f"\n[resources] {time.time()-t0w:.1f}s peakRSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")


if __name__ == "__main__":
    main()
