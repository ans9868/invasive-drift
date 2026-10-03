#!/usr/bin/env python3
"""24 - Same as 23 (low-rank gate + volatility + lead/lag) but with an MLP decoder, head-to-head with a
linear one. Rationale: the low-rank result could be an artefact of a LINEAR readout, so swap in a
nonlinear one. Feature set is IDENTICAL to 23 (instantaneous session-standardised state, no lags), so the
only change is linear -> nonlinear.

Per block: fit a decoder on that block; probe it on the FIXED per-direction mean states -> velocity
field V_b (2K=16). Drift step dV_b = V_{b+1}-V_b. Report (A) low-rank, (B) volatility, (C) lead/lag and
R2 series -- for RIDGE and MLP side by side.
"""
import argparse, glob, os, sys, time, resource, warnings
import numpy as np, h5py
from scipy.signal import lfilter
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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


def fit_ridge(X, y):
    m = X.mean(0); s = X.std(0) + 1e-6
    return m, s, Ridge(alpha=10.0).fit((X - m) / s, y)


def fit_mlp(X, y, hidden=(64, 64), max_iter=200):
    m = X.mean(0); s = X.std(0) + 1e-6
    mdl = MLPRegressor(hidden_layer_sizes=hidden, max_iter=max_iter, early_stopping=True,
                       random_state=0, alpha=1e-3)
    mdl.fit((X - m) / s, y)
    return m, s, mdl


def r2of(pred, y):
    return 1.0 - ((pred - y) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-12)


def analyze(V, r2):
    V = np.array(V); dV = np.diff(V, axis=0)
    Vc = dV - dV.mean(0); _, S, _ = np.linalg.svd(Vc, full_matrices=False)
    frac = S ** 2 / (S ** 2).sum()
    vol = np.linalg.norm(dV, axis=1)
    ar1 = float(np.corrcoef(vol[:-1], vol[1:])[0, 1]) \
        if (len(vol) >= 4 and vol[:-1].std() > 0 and vol[1:].std() > 0) else np.nan
    r2 = np.array(r2); n = min(len(vol), len(r2) - 1)
    return dict(pc1=float(frac[0]), pc12=float(frac[:2].sum()),
                r80=float(np.searchsorted(np.cumsum(frac), 0.8) + 1), vtime=sp(np.arange(len(vol)), vol),
                ar1=ar1, cont=sp(vol[:n], r2[:n]), lead=sp(vol[:n], r2[1:][:n]),
                ldr=sp(vol[:n], np.diff(r2)[:n]), r2=float(np.mean(r2)))


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
    res = {"ridge": [], "mlp": []}
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
        blocks = []
        for b in range(nb):
            lo = t0 + b * args.block_s; hi = lo + args.block_s
            tm = (ts >= lo) & (ts < hi)
            if tm.sum() >= 400:
                blocks.append((Xs[tm], vel[tm]))
        if len(blocks) < 4:
            continue
        (X0, y0) = blocks[0]
        mrg, srg, crg = fit_ridge(X0, y0)                      # block-0 ridge (for R2 series)
        mml, sml, cml = fit_mlp(X0, y0)                        # block-0 MLP   (for R2 series)
        Vr, Vm, r2r, r2m = [], [], [], []
        for (Xb, yb) in blocks:
            mb, sb, cb = fit_ridge(Xb, yb); Vr.append((cb.coef_ @ state.T).ravel())
            mn, sn, cn = fit_mlp(Xb, yb); Vm.append(cn.predict((state - mn) / sn).ravel())
            r2r.append(r2of(crg.predict((Xb - mrg) / srg), yb))
            r2m.append(r2of(cml.predict((Xb - mml) / sml), yb))
        a_r = analyze(Vr, r2r); a_m = analyze(Vm, r2m)
        res["ridge"].append(a_r); res["mlp"].append(a_m)
        print(f"  {os.path.basename(p).split('ses-')[1].split('_')[0]}: nb={len(blocks)} | ridge PC1={a_r['pc1']:.2f} PC12={a_r['pc12']:.2f} R2={a_r['r2']:.2f} "
              f"| mlp PC1={a_m['pc1']:.2f} PC12={a_m['pc12']:.2f} R2={a_m['r2']:.2f}")
    for mdl in ("ridge", "mlp"):
        R = res[mdl]
        if not R:
            continue
        g = lambda k: np.nanmean([x[k] for x in R])
        print(f"\n=== {mdl.upper()} ({len(R)} sessions) ===")
        print(f"  (A) LOW-RANK: PC1={g('pc1'):.2f}  PC1+2={g('pc12'):.2f}  rank80={g('r80'):.1f}/16")
        print(f"  (B) VOLATILITY: vs block rho={g('vtime'):+.2f}  lag-1 autocorr={g('ar1'):+.2f}")
        print(f"  (C) LEAD/LAG: corr(vol,R2)={g('cont'):+.3f}  corr(vol,R2_next)={g('lead'):+.3f}  corr(vol,dR2)={g('ldr'):+.3f}")
        print(f"  decoder R2 (block-0 decoder): {g('r2'):.3f}")
    print(f"\n[resources] {time.time()-t0w:.1f}s peakRSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")


if __name__ == "__main__":
    main()
