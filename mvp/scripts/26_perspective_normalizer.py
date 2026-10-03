#!/usr/bin/env python3
"""26 - Put the normalizer 'win' into perspective OBJECTIVELY.

A1 = per-unit re-centre+re-scale of each window to the burn-in reference (label-free). Diagnostics:
  (a) DISTRIBUTION: per-session (A1 - frozen); mean, SD, fraction improving, paired sign test.
  (b) DECOMPOSITION: mean-only vs std-only vs both -> which part drives the gain?
  (c) UNIT-SHUFFLED reference: use the correct mu0/sigma0 but permuted across units -> kills per-unit
      correspondence. If the gain survives, it is not about per-unit identity.
  (d) GLOBAL (scalar) moments: one mean/std for all units -> tests whether per-unit granularity matters.
  (e) BEHAVIOR control: correlate the A1 gain with the window's behavior change (mean speed) vs burn-in.
"""
import argparse, glob, os, sys, time, resource
import numpy as np, h5py
from scipy.signal import lfilter
from scipy.stats import spearmanr, binomtest
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


def r2(pred, y):
    return 1.0 - ((pred - y) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-12)


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
    M = ["frozen", "A1", "mean_only", "std_only", "A1_shufref", "A1_global"]
    per = {m: [] for m in M}; beh = []
    rng = np.random.default_rng(0)
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
        mu0 = Z[bm].mean(0); sd0 = Z[bm].std(0) + 1e-6
        perm = rng.permutation(Z.shape[1]); d = Z.shape[1]
        sp0 = float(np.linalg.norm(vel[bm], axis=1).mean())
        test = ~bm; tstart = t0 + args.burnin * dur; tlen = (ts[-1] - tstart) / args.nwin
        acc = {m: [] for m in M}; gains = []; bdelta = []
        for w in range(args.nwin):
            lo = tstart + w * tlen; hi = lo + tlen; sel = test & (ts >= lo) & (ts < hi)
            if sel.sum() < 400:
                continue
            Zw, vw = Z[sel], vel[sel]
            dec = lambda zz: zz @ W.T
            acc["frozen"].append(r2(dec(Zw), vw))
            mw = Zw.mean(0); sw = Zw.std(0) + 1e-6
            acc["A1"].append(r2(dec((Zw - mw) / sw * sd0 + mu0), vw))
            acc["mean_only"].append(r2(dec(Zw - mw + mu0), vw))
            acc["std_only"].append(r2(dec(Zw * (sd0 / sw)), vw))
            acc["A1_shufref"].append(r2(dec((Zw - mw) / sw * sd0[perm] + mu0[perm]), vw))
            gmw, gsw = float(Zw.mean()), float(Zw.std()) + 1e-6
            acc["A1_global"].append(r2(dec((Zw - gmw) / gsw * float(sd0.mean()) + float(mu0.mean())), vw))
            gains.append(acc["A1"][-1] - acc["frozen"][-1])
            bdelta.append(float(np.linalg.norm(vw, axis=1).mean()) - sp0)
        if not acc["frozen"]:
            continue
        for m in M:
            if acc[m]:
                per[m].append(float(np.nanmean(acc[m])))
        per["frozen"] = per["frozen"]
        beh.append(float(spearmanr(gains, bdelta)[0]) if len(set(bdelta)) > 1 else np.nan)
        print(f"  {os.path.basename(p).split('ses-')[1].split('_')[0]}: frozen={np.nanmean(acc['frozen']):.3f} A1={np.nanmean(acc['A1']):.3f} "
              f"mean={np.nanmean(acc['mean_only']):.3f} std={np.nanmean(acc['std_only']):.3f} shuf={np.nanmean(acc['A1_shufref']):.3f} glob={np.nanmean(acc['A1_global']):.3f}")
    print(f"\nN={len(per['frozen'])} sessions")
    d = np.array(per["A1"]) - np.array(per["frozen"])
    print(f"(a) A1 - frozen per session: mean={d.mean():+.4f}  sd={d.std():.4f}  median={np.median(d):+.4f}  "
          f"frac>0={np.mean(d > 0):.2f}  signTest p={binomtest(int((d > 0).sum()), len(d)).pvalue:.3g}")
    print(f"(b) decomposition (mean R2): frozen={np.mean(per['frozen']):.3f}  "
          f"mean_only={np.mean(per['mean_only']):.3f}  std_only={np.mean(per['std_only']):.3f}  A1(both)={np.mean(per['A1']):.3f}")
    print(f"(c) unit-shuffled reference: {np.mean(per['A1_shufref']):.3f}  (vs A1 {np.mean(per['A1']):.3f})")
    print(f"(d) global scalar moments:   {np.mean(per['A1_global']):.3f}")
    print(f"(e) behavior control: mean spearman(gain, speed delta) = {np.nanmean(beh):+.3f}")
    print(f"\n[resources] {time.time()-t0w:.1f}s peakRSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")


if __name__ == "__main__":
    main()
