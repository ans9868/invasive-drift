#!/usr/bin/env python3
"""Benchmark a zoo of decoders intra-session (Perich cursor velocity), across ALL sessions."""
import argparse, glob, os, sys
import numpy as np, h5py
from scipy.signal import lfilter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decoders as D

BIN_MS = 20.0; TAU_MS = 240.0


def load(path):
    with h5py.File(path, "r") as h:
        st = h["units/spike_times"][:]; idx = h["units/spike_times_index"][:]
        b = np.concatenate([[0], idx]); n = len(idx)
        spies = [st[b[u]:b[u + 1]] for u in range(n)]
        posg = h["processing/behavior/Position/cursor_pos"]; ts = posg["timestamps"][:]
        pos = posg["data"][:].astype(np.float32)
        vel = h["processing/behavior/Velocity/cursor_vel/data"][:].astype(np.float32)
    bs = BIN_MS / 1000.0; edges = np.concatenate([[ts[0] - bs], ts])
    X = np.zeros((len(ts), n), np.float32)
    for u in range(n):
        X[:, u] = np.histogram(spies[u], bins=edges)[0]
    return X, pos, vel


def exp_filt(x, tau_ms=TAU_MS, bin_ms=BIN_MS):
    t = np.arange(0.0, tau_ms, bin_ms); k = np.exp(-t / tau_ms); k /= k.sum()
    return lfilter(k, [1.0], x, axis=0).astype(np.float32)


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()


def specs():
    s = [("ridge", D.RidgeDec), ("wiener_L5", D.WienerDec), ("mlp_L3", D.MLPDec),
         ("kf_vel", lambda: D.KalmanDec("vel")), ("kf_posvel", lambda: D.KalmanDec("posvel"))]
    if getattr(D, "_HAS_TORCH", False):
        s.append(("gru_L10", lambda: D.GRUDec(10)))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/perich/sub-C")
    ap.add_argument("--n", type=int, default=0)   # 0 = all
    ap.add_argument("--sub", type=int, default=60000)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data, "*.nwb")))
    if args.n:
        files = files[:args.n]
    sp = specs(); names = [n for n, _ in sp]
    print(f"{'session':16s} " + " ".join(f"{n:>10s}" for n in names))
    acc = {n: [] for n in names}
    for p in files:
        try:
            X, pos, vel = load(p)
        except Exception as exc:
            print("SKIP", os.path.basename(p), exc); continue
        X = exp_filt(X); keep = ~np.isnan(vel).any(1); X, pos, vel = X[keep], pos[keep], vel[keep]
        if args.sub and len(vel) > args.sub:
            X, pos, vel = X[:args.sub], pos[:args.sub], vel[:args.sub]
        ntr = int(0.8 * len(vel))
        sess = os.path.basename(p).split("ses-")[1].split("_")[0]
        row = []
        for name, ctor in sp:
            d = ctor()
            try:
                if isinstance(d, D.KalmanDec):
                    d.fit(X[:ntr], pos[:ntr], vel[:ntr])
                else:
                    d.fit(X[:ntr], vel[:ntr])
                P = d.predict(X[ntr:]); yt = vel[ntr:][-len(P):]
                v = r2(yt, P); acc[name].append(v); row.append(v)
            except Exception as exc:
                row.append(np.nan)
        print(f"{sess:16s} " + " ".join(f"{v:10.3f}" for v in row))
    print("-" * (16 + 11 * len(names)))
    print(f"{'MEAN':16s} " + " ".join(f"{np.nanmean(acc[n]):10.3f}" for n in names))
    print(f"{'STD':16s} " + " ".join(f"{np.nanstd(acc[n]):10.3f}" for n in names))
    print(f"{'N':16s} " + " ".join(f"{len(acc[n]):10d}" for n in names))


if __name__ == "__main__":
    main()
