#!/usr/bin/env python3
"""Within-session mechanism: WHAT drives the decoder decay?
Frozen (day-ref) decoder vs Refit (recent) decoder; tuning-weight drift; firing-rate (gain) trend.
If refit >> frozen -> representational drift (recalibratable). If both decay -> intrinsic."""
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


def zn(X, m, s):
    return (X - m) / (s + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/perich/sub-C")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--block-s", type=float, default=120.0)
    ap.add_argument("--ref-frac", type=float, default=0.2)
    ap.add_argument("--min-block", type=int, default=4)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data, "*.nwb")))[:args.n]
    for p in files:
        try:
            X, vel, ts = load(p)
        except Exception as exc:
            print("SKIP", os.path.basename(p), exc); continue
        X = exp_filt(X); keep = ~np.isnan(vel).any(1); X, vel, ts = X[keep], vel[keep], ts[keep]
        nref = int(args.ref_frac * len(vel))
        mr, sr = X[:nref].mean(0), X[:nref].std(0)
        frozen = Ridge(alpha=10.0).fit(zn(X[:nref], mr, sr), vel[:nref])
        wref = frozen.coef_.ravel()
        t0 = ts[0]; nb = int((ts[-1] - t0) // args.block_s)
        sess = os.path.basename(p).split("ses-")[1].split("_")[0]
        print(f"\n=== {sess}: X{X.shape} blocks={nb} ===")
        print(f"{'b':>3s} {'t(min)':>7s} {'frozenR2':>9s} {'refitR2':>8s} {'rate':>6s} {'wcorr':>6s}")
        for b in range(nb):
            lo = t0 + b * args.block_s; hi = lo + args.block_s; sel = (ts >= lo) & (ts < hi)
            if sel.sum() < 400:
                continue
            Xb, vb = X[sel], vel[sel]
            fr = r2(vb, frozen.predict(zn(Xb, mr, sr)))
            # refit: train on first 70% of block, test last 30%
            k = int(0.7 * len(vb))
            m2, s2 = Xb[:k].mean(0), Xb[:k].std(0)
            rf = Ridge(alpha=10.0).fit(zn(Xb[:k], m2, s2), vb[:k])
            rr = r2(vb[k:], rf.predict(zn(Xb[k:], m2, s2)))
            rate = float(Xb.mean())
            w = rf.coef_.ravel()
            cc = float(np.corrcoef(w, wref)[0, 1])
            print(f"{b:3d} {(b*args.block_s+args.block_s/2)/60:7.1f} {fr:9.3f} {rr:8.3f} {rate:6.2f} {cc:6.3f}")


if __name__ == "__main__":
    main()
