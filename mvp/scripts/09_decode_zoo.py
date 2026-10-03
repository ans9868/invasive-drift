#!/usr/bin/env python3
"""Benchmark a zoo of decoders intra-session (Perich cursor velocity)."""
import argparse, glob, os, sys
import numpy as np, h5py
from scipy.signal import lfilter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decoders import RidgeDec, WienerDec, MLPDec, KalmanDec

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/perich/sub-C")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--sub", type=int, default=60000)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data, "*.nwb")))[:args.n]
    specs = [("ridge", RidgeDec), ("wiener_L5", WienerDec), ("mlp_L3", MLPDec),
             ("kf_vel", lambda: KalmanDec("vel")), ("kf_posvel", lambda: KalmanDec("posvel"))]
    names = [n for n, _ in specs]
    print(f"{'session':16s} " + " ".join(f"{n:>10s}" for n in names))
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
        outs = []
        for name, ctor in specs:
            d = ctor()
            try:
                if isinstance(d, KalmanDec):
                    d.fit(X[:ntr], pos[:ntr], vel[:ntr])
                else:
                    d.fit(X[:ntr], vel[:ntr])
                P = d.predict(X[ntr:]); yt = vel[ntr:][-len(P):]
                outs.append(r2(yt, P))
            except Exception as exc:
                outs.append(np.nan)
        print(f"{sess:16s} " + " ".join(f"{v:10.3f}" for v in outs))


if __name__ == "__main__":
    main()
