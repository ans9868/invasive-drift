#!/usr/bin/env python3
"""19 - Drift axis, unit-FREE (measured in velocity space).

PRE-REGISTERED plan (see drafts/ideas.md Ideas 7/9 + bonus). Question: is the within-session
drift "the same turn" every session, and is it one shared mode?

Design choice: per-block decoders are fit in the SESSION-standardised feature basis, so every
block weight lives in ONE basis and the drift is not confounded by per-block rescalers. We probe
a FIXED set of per-direction mean states with each block's decoder -> a per-direction VELOCITY
FIELD V_b (a 2K vector). Drift = how V_b moves over blocks. Because V lives in velocity space,
it is comparable across sessions WITHOUT unit matching.

  (1) Axis consistency: per-session drift direction = V_last - V_first (and PC1 of V).
      Report mean pairwise cosine across sessions (random-direction null ~ 0).
  (2) Shared mode: SVD of block-to-block changes dV. Mean PC1 variance fraction, and the
      cross-session cosine of the PC1 direction.
  (3) Drift RATE vs PRE-REGISTERED covariates only: n_units, duration_min, session_index, start_hour.

Resource use is reported in-script (peak RSS + per-stage time); NO squeue/sacct polling.
"""
import argparse, glob, os, sys, time, resource, datetime as dt
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


def unit(v):
    v = np.asarray(v, float).ravel(); return v / (np.linalg.norm(v) + 1e-12)


def start_hour(path):
    try:
        with h5py.File(path, "r") as h:
            if "session_start_time" in h:
                v = h["session_start_time"][()]
                s = v.decode() if isinstance(v, bytes) else str(v)
                if "T" in s:
                    t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
                    return t.hour + t.minute / 60.0
    except Exception:
        pass
    return np.nan


def mean_pairwise_cos(vecs):
    A = np.array([unit(v) for v in vecs])
    n = len(A)
    if n < 3:
        return np.nan, np.nan, 0
    C = A @ A.T; iu = np.triu_indices(n, 1)
    return float(np.mean(C[iu])), float(np.std(C[iu])), n


def main():
    t_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/perich/sub-C")
    ap.add_argument("--block-s", type=float, default=120.0)
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data, "*.nwb")))
    if args.n:
        files = files[:args.n]
    axes_end, axes_pc, pc1_traj, pc1_chg, pc1dir_chg = [], [], [], [], []
    rows = []; rate = []; cov = {"n_units": [], "duration_min": [], "session_index": [], "start_hour": []}
    t_loop = time.time()
    for idx, p in enumerate(files):
        try:
            X, vel, ts = load(p)
        except Exception as exc:
            print("SKIP", os.path.basename(p), exc); continue
        X = exp_filt(X); keep = ~np.isnan(vel).any(1); X, vel, ts = X[keep], vel[keep], ts[keep]
        m_s = X.mean(0); s_s = X.std(0) + 1e-6
        Xs = ((X - m_s) / s_s).astype(np.float32)                      # session basis
        speed = np.linalg.norm(vel, axis=1); moving = speed > np.nanpercentile(speed, 60)
        ang = np.degrees(np.arctan2(vel[moving, 1], vel[moving, 0]))
        bidx = np.clip(np.digitize(ang, np.linspace(-180, 180, K + 1)) - 1, 0, K - 1)
        Xm = Xs[moving]; state = np.zeros((K, Xs.shape[1]), np.float32); nk = 0
        for k in range(K):
            sel = bidx == k
            if sel.sum() >= 100:
                state[k] = Xm[sel].mean(0); nk += 1
        if nk < 4:
            print("SKIP(empty dirs)", os.path.basename(p)); continue
        t0 = ts[0]; nb = int((ts[-1] - t0) // args.block_s)
        W = []
        for b in range(nb):
            lo = t0 + b * args.block_s; hi = lo + args.block_s; sel = (ts >= lo) & (ts < hi)
            if sel.sum() < 400:
                continue
            W.append(Ridge(alpha=10.0).fit(Xs[sel], vel[sel]).coef_)   # (2,d) in session basis
        if len(W) < 3:
            continue
        V = np.array([(W[b] @ state.T).ravel() for b in range(len(W))])  # (nb, 2K)
        d = V[-1] - V[0]; dur = (ts[-1] - t0) / 60.0
        Vc = V - V.mean(0); _, S, Vt = np.linalg.svd(Vc, full_matrices=False)
        dV = np.diff(V, axis=0); dVc = dV - dV.mean(0); _, S2, Vt2 = np.linalg.svd(dVc, full_matrices=False)
        axes_end.append(unit(d)); axes_pc.append(unit(Vt[0])); pc1dir_chg.append(unit(Vt2[0]))
        pc1_traj.append(float(S[0] ** 2 / (S ** 2).sum())); pc1_chg.append(float(S2[0] ** 2 / (S2 ** 2).sum()))
        rate.append(float(np.linalg.norm(d)) / max(dur, 1e-6))
        cov["n_units"].append(X.shape[1]); cov["duration_min"].append(dur)
        cov["session_index"].append(idx); cov["start_hour"].append(start_hour(p))
        rows.append((os.path.basename(p).split("ses-")[1].split("_")[0], X.shape[1], len(W), dur,
                     float(np.linalg.norm(d)), rate[-1], pc1_traj[-1], pc1_chg[-1]))
        print(f"  {rows[-1][0]}: units={X.shape[1]:3d} nb={len(W):2d} dur={dur:5.1f}m "
              f"|driftV|={rows[-1][4]:5.2f} rate={rate[-1]:.3f}/m PC1traj={pc1_traj[-1]:.2f} PC1chg={pc1_chg[-1]:.2f}")
    t_loop = time.time() - t_loop

    print(f"\nN={len(rate)} sessions with >=3 blocks")
    me, se, n = mean_pairwise_cos(axes_end)
    mp, sp, _ = mean_pairwise_cos(axes_pc)
    mc, sc, _ = mean_pairwise_cos(pc1dir_chg)
    print(f"(1) AXIS CONSISTENCY (mean pairwise cosine; 2K={2*K} dim -> random ~0, sd~{1/np.sqrt(2*K):.2f})")
    print(f"    end-to-end drift dir : mean_cos={me:+.3f} (sd {se:.2f}, n={n})")
    print(f"    PC1 of V trajectory  : mean_cos={mp:+.3f} (sd {sp:.2f})")
    print(f"(2) SHARED MODE")
    print(f"    PC1 var fraction  trajectory={np.nanmean(pc1_traj):.2f}  step-changes={np.nanmean(pc1_chg):.2f}")
    print(f"    cross-session cos of step-PC1 dir: mean_cos={mc:+.3f} (sd {sc:.2f})")
    print(f"(3) DRIFT RATE vs PRE-REGISTERED covariates (Spearman over {len(rate)} sessions)")
    for k_, v_ in cov.items():
        v_ = np.asarray(v_, float); msk = ~np.isnan(v_)
        if msk.sum() < 5:
            print(f"    {k_:14s}: n/a"); continue
        r_, p_ = spearmanr(v_[msk], np.asarray(rate)[msk])
        print(f"    {k_:14s}: rho={r_:+.3f} p={p_:.3f} (n={msk.sum()})")
    dd = np.array([rw[4] for rw in rows]); durs = np.array([rw[3] for rw in rows]); nu = np.array([rw[1] for rw in rows])
    print(f"    [diag] |driftV| vs duration: rho={spearmanr(durs, dd)[0]:+.3f}"
          f" | vs n_units: rho={spearmanr(nu, dd)[0]:+.3f}")
    print(f"\n[resources] loop={t_loop:.1f}s total={time.time()-t_start:.1f}s "
          f"peakRSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")


if __name__ == "__main__":
    main()
