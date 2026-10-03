#!/usr/bin/env python3
"""Summarise the cached artifacts: durations, window counts, short/long split. Read-only."""
import glob
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "artifacts/perich_subC")
    files = sorted(glob.glob(os.path.join(d, "*.npz")))
    rows = []
    for p in files:
        try:
            z = np.load(p, allow_pickle=True)
            rows.append((os.path.basename(p).split(".")[0], float(z["session_minutes"]),
                         int(z["n_windows"]), bool(z["short_recording"]), int(z["n_units"])))
        except Exception as exc:  # noqa: BLE001
            print("SKIP", os.path.basename(p), exc)
    if not rows:
        print("no artifacts in", d)
        return
    mins = sorted(r[1] for r in rows)
    wins = [r[2] for r in rows]
    print(f"sessions = {len(rows)}   dir = {d}")
    print(f"minutes : min {mins[0]:.1f}  p25 {mins[len(mins)//4]:.1f}  median {mins[len(mins)//2]:.1f}"
          f"  p75 {mins[3*len(mins)//4]:.1f}  max {mins[-1]:.1f}")
    print("window counts (n_windows -> #sessions):", dict(sorted(Counter(wins).items())))
    print(f"short (n_windows < 5) = {sum(r[3] for r in rows)} / {len(rows)}")
    print(f"  -> usable for staleness (n_windows >= 5) = {sum(w >= 5 for w in wins)} sessions")
    print("\nper session (minutes, windows, short, units):")
    for r in sorted(rows, key=lambda x: x[1]):
        print(f"  {r[0]:12s} {r[1]:5.1f}min  W={r[2]:2d}  short={str(r[3]):5s} units={r[4]}")


if __name__ == "__main__":
    main()
