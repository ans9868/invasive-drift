#!/usr/bin/env python3
"""IDEA 16 — the 80/20 SANITY CHECK (known-answer test).

Replicates the decoder zoo's number through THIS pipeline: fit a decoder on the first 80% of the
session, test on the last 20%. Expected (from `mvp/scripts/09_decode_zoo.py`, --sub 60000):
    ridge ≈ 0.357   wiener ≈ 0.404   kf_posvel ≈ 0.393

If we reproduce those, then loading / binning / standardisation / row-alignment / R2 are all confirmed
end-to-end against an independent implementation. If not -> we have a real bug that outranks everything.

Reads the cached artifacts (Z is already session-standardised; ridge/wiener/KF are affine-invariant
to that, so the numbers must match).
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "mvp"))
import decoders as D          # noqa: E402
from common import align_tail  # noqa: E402


def r2(p, y):
    return float(1.0 - ((p - y) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "artifacts/perich_subC"))
    ap.add_argument("--sub", type=int, default=60000)   # match the zoo's default
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    specs = [("ridge", D.RidgeDec), ("wiener", D.WienerDec), ("kf_posvel", lambda: D.KalmanDec("posvel"))]
    acc = {n: [] for n, _ in specs}
    files = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    if args.n:
        files = files[:args.n]

    for p in files:
        try:
            z = np.load(p, allow_pickle=True)
            Z = z["Z"]; vel = z["vel"]; pos = z["pos"]
        except Exception as exc:  # noqa: BLE001
            print("SKIP", os.path.basename(p), exc); continue
        if args.sub and len(vel) > args.sub:
            Z, vel, pos = Z[:args.sub], vel[:args.sub], pos[:args.sub]
        ntr = int(0.8 * len(vel))
        row = []
        for nm, ctor in specs:
            d = ctor()
            if nm.startswith("kf"):
                d.fit(Z[:ntr], pos[:ntr], vel[:ntr])
            else:
                d.fit(Z[:ntr], vel[:ntr])
            pred = np.asarray(d.predict(Z[ntr:]))
            v = r2(pred, align_tail(vel[ntr:], pred))
            acc[nm].append(v); row.append(v)
        print(f"  {os.path.basename(p).split('.')[0]:12s} " + "  ".join(f"{n}={v:.3f}" for n, v in zip(acc, row)))

    print(f"\nN = {len(acc['ridge'])} sessions   (zoo reference: ridge 0.357  wiener 0.404  kf_posvel 0.393)")
    for nm, _ in specs:
        print(f"  {nm:10s} mean R2 = {np.nanmean(acc[nm]):.3f}")
    print("\nSANITY_DONE")


if __name__ == "__main__":
    main()
