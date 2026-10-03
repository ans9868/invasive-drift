#!/usr/bin/env python3
"""Drift cause: (1) link drift metrics to decoder decay across sessions;
(2) recalibratability. Metrics per block: frozen R2, refit R2 (large expanding window),
weight-drift, FUNCTIONAL drift (prediction-space), behavior drift (mean speed)."""
import argparse, glob, os, sys
import numpy as np, h5py
from scipy.signal import lfilter
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


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()


def fit(X, y, a=10.0):
    m, s = X.mean(0), X.std(0) + 1e-6
    return Ridge(alpha=a).fit((X - m) / s, y), m, s


def pred(model_ms, X):
    model, m, s = model_ms
    return model.predict((X - m) / s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/perich/sub-C")
    ap.add_argument("--block-s", type=float, default=120.0)
    ap.add_argument("--ref-frac", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data, "*.nwb")))
    if args.n:
        files = files[:args.n]
    # per-session summary rows
    summ = []
    for p in files:
        try:
            X, vel, ts = load(p)
        except Exception as exc:
            print("SKIP", os.path.basename(p), exc); continue
        X = exp_filt(X); keep = ~np.isnan(vel).any(1); X, vel, ts = X[keep], vel[keep], ts[keep]
        nref = int(args.ref_frac * len(vel))
        frozen = fit(X[:nref], vel[:nref])
        bl0 = None; w0 = None
        t0 = ts[0]; nb = int((ts[-1] - t0) // args.block_s)
        fr, rr, wd, fd, bd = [], [], [], [], []
        for b in range(nb):
            lo = t0 + b * args.block_s; hi = lo + args.block_s; sel = (ts >= lo) & (ts < hi)
            if sel.sum() < 400:
                continue
            Xb, vb = X[sel], vel[sel]
            fr.append(r2(vb, pred(frozen, Xb)))
            # refit on expanding window [0, lo)
            pre = ts < lo
            if pre.sum() > 2000:
                rf = fit(X[pre], vel[pre])
                rr.append(r2(vb, pred(rf, Xb)))
            else:
                rr.append(np.nan)
            # block-0 encoder for weight + functional drift
            if bl0 is None:
                bl0 = fit(Xb, vb); w0 = bl0[0].coef_.ravel()
            blb = fit(Xb, vb); wb = blb[0].coef_.ravel()
            wd.append(float(np.corrcoef(wb, w0)[0, 1]))
            fd.append(float(np.corrcoef(pred(bl0, Xb).ravel(), pred(blb, Xb).ravel())[0, 1]))
            bd.append(float(np.abs(vb).mean()))
        fr, rr, wd, fd, bd = map(np.array, (fr, rr, wd, fd, bd))
        sess = os.path.basename(p).split("ses-")[1].split("_")[0]
        x = np.arange(len(fr)) * (args.block_s / 60.0)
        dec = float(np.polyfit(x, fr, 1)[0])   # decoder slope (decay) per min
        wslope = float(np.polyfit(x, wd, 1)[0]) if len(wd) > 2 else np.nan
        fslope = float(np.polyfit(x, fd, 1)[0]) if len(fd) > 2 else np.nan
        bslope = float(np.polyfit(x, bd, 1)[0]) if len(bd) > 2 else np.nan
        summ.append(dict(sess=sess, nb=len(fr), dec=dec, wslope=wslope, fslope=fslope, bslope=bslope,
                         frozen_mean=float(np.nanmean(fr)), refit_mean=float(np.nanmean(rr))))
        print(f"  {sess}: nb={len(fr)} frozenR2={np.nanmean(fr):.2f} refitR2={np.nanmean(rr):.2f} "
              f"decay={dec:+.4f}/min wdrift={wslope:+.4f} fdrift={fslope:+.4f} bdrift={bslope:+.4f}")
    import pandas as pd
    df = pd.DataFrame(summ)
    print(f"\nN={len(df)} sessions")
    for k in ["wslope", "fslope", "bslope"]:
        c = df[["dec", k]].dropna()
        r = np.corrcoef(c.dec, c[k])[0, 1] if len(c) > 2 else np.nan
        print(f"corr(decay, {k:7s}) = {r:+.3f}   (n={len(c)})")
    print(f"mean frozenR2={df.frozen_mean.mean():.3f}  mean refitR2={df.refit_mean.mean():.3f}")


if __name__ == "__main__":
    main()
