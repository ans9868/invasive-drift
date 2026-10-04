#!/usr/bin/env python3
"""L0 selftest for cross_session/match_units.py -- synthetic, no data, no pytest.

Run:  python cross_session/selftest_match.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import match_units as MU  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAILS.append(name)


def template(rng, n, W=48):
    """n distinct spike-ish waveforms: biphasic, unit-specific width/position/amplitude."""
    t = np.linspace(-1, 1, W)
    out = np.zeros((n, W), np.float32)
    for i in range(n):
        a = 1.0 + 0.4 * rng.normal()
        c = rng.uniform(-0.4, 0.4)
        w = 0.15 + 0.08 * abs(rng.normal())
        out[i] = (a * np.exp(-((t - c) ** 2) / (2 * w ** 2))
                  - 0.5 * a * np.exp(-((t - c - 0.25) ** 2) / (2 * (w * 1.5) ** 2)))
    return out


def main():
    rng = np.random.default_rng(0)
    W = 48
    n = 40
    shared = 20

    A = template(rng, n, W)
    fresh = template(rng, n - shared, W)
    perm = rng.permutation(n)
    B = np.zeros((n, W), np.float32)
    B[:shared] = A[:shared][perm[:shared]]      # the SAME units, permuted position
    B[shared:] = fresh                          # units that appear fresh / have no partner
    truth = {i: int(perm[i]) for i in range(shared)}     # A[i] -> B position

    # realistic day-to-day perturbation: jitter + amplitude change (amplitude was ~-7%/session)
    Bjit = B.copy()
    Bjit += rng.normal(scale=0.02, size=B.shape).astype(np.float32)
    Bjit *= rng.uniform(0.8, 1.25, size=(n, 1)).astype(np.float32)

    print("partial-overlap recovery (half persist, half are fresh):")
    iA, iB, C = MU.match(A, Bjit, 0.90)
    got = dict(zip(iA.tolist(), iB.tolist()))
    true_hits = sum(1 for a, b in got.items() if truth.get(a) == b)
    check("matched_count_close_to_shared", abs(len(iA) - shared) <= 2, f"matched={len(iA)} vs {shared}")
    check("all_matched_are_true_correspondences", true_hits == len(iA), f"{true_hits}/{len(iA)}")
    check("mutual_exclusion", len(set(iA.tolist())) == len(iA) and len(set(iB.tolist())) == len(iB))
    check("corr_matrix_shape", C.shape == (n, n))

    print("\nscale invariance (correlation is scale-invariant after de-meaning):")
    iA2, iB2, _ = MU.match(A, Bjit * 100.0, 0.90)
    check("amplitude_x100_same_match_count", len(iA2) == len(iA), f"{len(iA2)} vs {len(iA)}")

    print("\nunrelated units do NOT match:")
    noise = rng.normal(size=(n, W)).astype(np.float32)
    iAn, iBn, _ = MU.match(A, noise, 0.90)
    check("noise_matches_nothing", len(iAn) == 0, f"matched={len(iAn)}")

    print("\nthreshold is monotone and excludes the null:")
    counts = [len(MU.match(A, Bjit, t)[0]) for t in (0.5, 0.9, 0.95, 0.99)]
    check("match_count_decreases_with_threshold",
          all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), f"{counts}")

    print("\nheavy noise degrades gracefully:")
    Bn = (B + rng.normal(scale=0.5, size=B.shape).astype(np.float32))
    iA3, iB3, _ = MU.match(A, Bn, 0.90)
    check("heavy_noise_matches_fewer", len(iA3) < len(iA), f"{len(iA3)} < {len(iA)}")

    print("\ndegenerate input:")
    z = np.zeros((3, W), np.float32)
    try:
        iAz, iBz, _ = MU.match(z, z, 0.90)
        check("all_zero_waveforms_no_crash", np.all(np.isfinite(iAz)), f"matched={len(iAz)}")
    except Exception as exc:  # noqa: BLE001
        check("all_zero_waveforms_no_crash", False, str(exc))

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
