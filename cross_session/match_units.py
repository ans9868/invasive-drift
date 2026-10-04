#!/usr/bin/env python3
"""Cross-session step 0: build unit-CORRESPONDENCE tables for CONSECUTIVE session pairs.

WHY. Cross-session decoder transfer needs to know which unit on day N is which unit on day N+1.
The existing prototype (`mvp/scripts/14_waveform_crosssession.py`) matched consecutive pairs but only
PRINTED the result. This saves the tables.

KEY FACT (verified in `grid_within_session_27/cache.py::load_session`): the artifact's `Z` column
order IS the NWB unit order -- `X[:, u]` is built by iterating `units/spike_times_index`. So a table
of raw NWB unit indices re-indexes straight into the existing artifacts. **No re-cache needed.**

METHOD (identical algorithm to `14`, so the numbers are comparable and cross-checkable): per-unit mean
waveform over channel 0, L2-normalised, then a GREEDY max-correlation assignment above `--thresh` with
mutual exclusion. Greedy, not optimal (Hungarian) -- flagged here, not hidden.

Outputs, per consecutive pair k -> k+1:
  artifacts/cross_session/matches/<A>__<B>.npz    idxA, idxB, corr, nA, nB, thresh, wall_s
  artifacts/cross_session/summary.csv             one row per pair, append-safe
"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import h5py

W_DEFAULT = 48


def load_waveforms(path):
    """-> means (n,48), amp (n,), nspk (n,), diag.

    Unit order == `units/spike_times_index` order == artifact column order. We ASSERT that the
    waveforms ragged index agrees with it, because a silent mismatch here would make every
    downstream match table wrong while still looking plausible.
    """
    with h5py.File(path, "r") as h:
        sidx = h["units/spike_times_index"][:]
        wf = h["units/waveforms"][:, 0]
        wiidx = h["units/waveforms_index_index"][:]
    raw = wf if wf.ndim == 2 else wf.reshape(-1, W_DEFAULT)
    n_samp = int(raw.shape[1])
    bounds = np.concatenate([[0], wiidx.astype(np.int64)])
    diag = dict(n_units_spike_index=int(len(sidx)), n_units_wfm_index=int(len(wiidx)),
                rows_wfm=int(len(raw)), last_bound=int(bounds[-1]), n_samp=n_samp)
    if len(wiidx) != len(sidx):
        raise ValueError(f"unit-count mismatch between spike_times_index and waveforms index: {diag}")
    if bounds[-1] > len(raw):
        raise ValueError(f"ragged index runs past the waveform array: {diag}")
    means = np.zeros((len(wiidx), n_samp), np.float32)
    amp = np.zeros(len(wiidx), np.float64)
    nspk = np.zeros(len(wiidx), np.int64)
    for u in range(len(wiidx)):
        lo, hi = int(bounds[u]), int(bounds[u + 1])
        nspk[u] = hi - lo
        if hi <= lo:
            continue
        m = raw[lo:hi].mean(0)
        means[u] = m
        amp[u] = float(m.max() - m.min())
    diag["units_with_no_waveform"] = int((nspk == 0).sum())
    return means, amp, nspk, diag


def unit_norm(X):
    X = X - X.mean(1, keepdims=True)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def match(A, B, thresh):
    """Greedy max-correlation assignment with mutual exclusion. Returns (idxA, idxB) and the
    full correlation matrix (so callers can build a null / check symmetry without recomputing)."""
    C = unit_norm(A) @ unit_norm(B).T
    Cc = C.copy()
    ii, jj = [], []
    while Cc.size:
        i, j = np.unravel_index(int(np.argmax(Cc)), Cc.shape)
        if Cc[i, j] < thresh:
            break
        ii.append(int(i))
        jj.append(int(j))
        Cc[i, :] = -1.0
        Cc[:, j] = -1.0
    return np.array(ii, int), np.array(jj, int), C


def sess_id(path):
    return os.path.basename(path).split("ses-")[1].split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/perich/sub-C")
    ap.add_argument("--out", default="artifacts/cross_session")
    ap.add_argument("--thresh", type=float, default=0.90)
    ap.add_argument("--n", type=int, default=0, help="limit to the first N sessions (0 = all)")
    ap.add_argument("--pairs", type=int, default=0, help="limit to the first N pairs (0 = all)")
    ap.add_argument("--diag", action="store_true", help="print ragged-index diagnostics per session")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "*.nwb")))
    if args.n:
        files = files[:args.n]
    if not files:
        print(f"FATAL: no .nwb under {args.data}")
        return 1
    mdir = os.path.join(args.out, "matches")
    os.makedirs(mdir, exist_ok=True)

    t0 = time.time()
    means, amps, ids, diags = [], [], [], []
    for p in files:
        try:
            M, A, S, d = load_waveforms(p)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {sess_id(p)}: {exc}")
            continue
        means.append(M)
        amps.append(A)
        ids.append(sess_id(p))
        diags.append(d)
        if args.diag:
            print(f"  diag {ids[-1]}: {d}")
    print(f"loaded {len(means)} sessions in {time.time() - t0:.0f}s")
    if len(means) < 2:
        print("FATAL: need >= 2 sessions")
        return 1

    hdr = ("pair", "nA", "nB", "matched", "fracA", "med_corr", "p10_corr", "sym_frac", "wall_s")
    print("\n%-24s %5s %5s %8s %7s %9s %9s %8s %7s" % hdr)
    rows = []
    npairs = len(means) - 1 if not args.pairs else min(args.pairs, len(means) - 1)
    for k in range(npairs):
        ta = time.time()
        A, B = means[k], means[k + 1]
        iA, iB, C = match(A, B, args.thresh)
        iB2, iA2, _ = match(B, A, args.thresh)          # symmetry check (greedy is NOT symmetric)
        sAB = set(zip(iA.tolist(), iB.tolist()))
        sBA = set(zip(iA2.tolist(), iB2.tolist()))
        sym = len(sAB & sBA) / max(1, len(sAB | sBA))
        cs = C[iA, iB] if len(iA) else np.array([])
        wall = time.time() - ta
        name = f"{ids[k]}__{ids[k + 1]}"
        np.savez_compressed(os.path.join(mdir, name + ".npz"), idxA=iA, idxB=iB, corr=cs,
                            nA=A.shape[0], nB=B.shape[0], thresh=args.thresh, wall_s=wall)
        row = dict(pair=name, A=ids[k], B=ids[k + 1], nA=A.shape[0], nB=B.shape[0],
                   matched=len(iA), fracA=len(iA) / max(1, A.shape[0]),
                   med_corr=float(np.median(cs)) if len(cs) else 0.0,
                   p10_corr=float(np.percentile(cs, 10)) if len(cs) else 0.0,
                   sym_frac=sym, wall_s=wall)
        rows.append(row)
        print("%-24s %5d %5d %8d %7.2f %9.3f %9.3f %8.3f %7.1f" % (
            name, row["nA"], row["nB"], row["matched"], row["fracA"],
            row["med_corr"], row["p10_corr"], sym, wall))

    # INDICATIVE forward chain: follow each session-0 unit forward through the per-pair matches.
    # Greedy composition (not a proper union-find), so a unit whose target is already claimed dies.
    # Reported ONLY as an indicative survivor count.
    alive = {u: u for u in range(means[0].shape[0])}
    for k in range(len(means) - 1):
        A, B = means[k], means[k + 1]
        iA, iB, _ = match(A, B, args.thresh)
        fwd = dict(zip(iA.tolist(), iB.tolist()))
        taken = set()
        nxt = {}
        for u0, cur in alive.items():
            tgt = fwd.get(cur)
            if tgt is not None and tgt not in taken:
                taken.add(tgt)
                nxt[u0] = tgt
        alive = nxt
    print(f"\nINDICATIVE chain: {len(alive)} of {means[0].shape[0]} units from {ids[0]} "
          f"survive greedily to {ids[-1]}")

    sp = os.path.join(args.out, "summary.csv")
    with open(sp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} match tables -> {mdir}\n      summary -> {sp}")
    print(f"[resources] total={time.time() - t0:.1f}s")
    print("MATCH_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
