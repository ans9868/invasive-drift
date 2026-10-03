#!/usr/bin/env python3
"""P2 cache — build per-session artifacts (features, splits, reference) for the exp-27 grid.

Everything is derived from BURN-IN ONLY for the scaler/reference (no leakage).

Tracer usage:
    python grid_within_session_27/cache.py --n 1
Outputs <artifact_dir>/<session>.npz and prints a summary + timing + peak RSS.
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import h5py
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_cfg(path):
    with open(path) as fh:
        return json.load(fh)


def load_session(path, bin_ms, tau_ms):
    with h5py.File(path, "r") as h:
        st = h["units/spike_times"][:]
        idx = h["units/spike_times_index"][:]
        b = np.concatenate([[0], idx])
        n = len(idx)
        spies = [st[b[u]:b[u + 1]] for u in range(n)]
        vg = h["processing/behavior/Velocity/cursor_vel"]
        ts = vg["timestamps"][:]
        vel = vg["data"][:].astype(np.float32)
        pg = h["processing/behavior/Position/cursor_pos"]
        pos = pg["data"][:].astype(np.float32)
    bs = bin_ms / 1000.0
    edges = np.concatenate([[ts[0] - bs], ts])
    X = np.zeros((len(ts), n), np.float32)
    for u in range(n):
        X[:, u] = np.histogram(spies[u], bins=edges)[0]
    t = np.arange(0.0, tau_ms, bin_ms)
    k = np.exp(-t / tau_ms); k /= k.sum()
    X = lfilter(k, [1.0], X, axis=0).astype(np.float32)
    return X, vel, pos, ts


def direction_bins(vel, nbins):
    speed = np.linalg.norm(vel, axis=1)
    moving = speed > np.nanpercentile(speed, 60)
    ang = np.degrees(np.arctan2(vel[moving, 1], vel[moving, 0]))
    d = np.full(len(vel), -1)
    d[moving] = np.clip(np.digitize(ang, np.linspace(-180, 180, nbins + 1)) - 1, 0, nbins - 1)
    return d


def build(path, cfg):
    X, vel, pos, ts = load_session(path, cfg["bin_ms"], cfg["tau_ms"])
    keep = ~np.isnan(vel).any(1)
    X, vel, pos, ts = X[keep], vel[keep], pos[keep], ts[keep]
    t0, dur = ts[0], ts[-1] - ts[0]
    bm = ts < (t0 + cfg["burnin_frac"] * dur)
    m0 = X[bm].mean(0); s0 = X[bm].std(0) + 1e-6
    Z = ((X - m0) / s0).astype(np.float32)
    d = Z.shape[1]
    k = min(cfg["pca_k"], d)
    Zb = Z[bm] - Z[bm].mean(0)
    Vt = np.linalg.svd(Zb, full_matrices=False)[2]
    P = Vt[:k].T
    Vz = Zb @ P
    C0k = np.cov(Vz, rowvar=False) + 1e-6 * np.eye(k)
    C0 = np.cov(Zb, rowvar=False) + 1e-6 * np.eye(d)
    dirbin = direction_bins(vel, cfg["dir_bins"])
    K = cfg["dir_bins"]
    dirC = np.zeros((K, d), np.float32); dirOK = np.zeros(K, bool)
    for kk in range(K):
        sm = bm & (dirbin == kk)
        if sm.sum() >= cfg["min_dir_samples"]:
            dirC[kk] = Z[sm].mean(0); dirOK[kk] = True
    vbm = vel[bm]
    W = cfg["n_windows"]
    tstart = t0 + cfg["burnin_frac"] * dur
    tlen = (ts[-1] - tstart) / W
    fit_mask = np.zeros((W, len(ts)), bool); eval_mask = np.zeros((W, len(ts)), bool)
    for w in range(W):
        lo = tstart + w * tlen; hi = lo + tlen
        inw = (ts >= lo) & (ts < hi)
        ii = np.where(inw)[0]
        if len(ii) < cfg["min_window_samples"]:
            continue
        cut = int(len(ii) * cfg["pool_frac"])
        fit_mask[w, ii[:cut]] = True
        eval_mask[w, ii[cut:]] = True
    valid_w = (fit_mask.sum(1) >= cfg["min_fit_samples"]) & (eval_mask.sum(1) >= 100)
    # ---- per-window CONTEXT (computed ONCE per window; broadcast to every cell) ----
    speed = np.linalg.norm(vel, axis=1)
    moving = speed > np.nanpercentile(speed, 60)
    bin_s = cfg["bin_ms"] / 1000.0
    ctx_names = ["rate_mean", "rate_median", "frac_silent", "active_units", "mean_pairwise_corr",
                 "pc1_var", "eff_dim", "subspace_angle_deg", "speed_mean", "moving_frac", "dir_coverage"]
    ctx = np.full((W, len(ctx_names)), np.nan, np.float32)
    rngc = np.random.default_rng(cfg.get("seed", 0))
    for w in range(W):
        rows = np.where(fit_mask[w])[0]
        if len(rows) < cfg["min_fit_samples"]:
            continue
        rate = X[rows].mean(0) / bin_s                       # Hz per unit
        Zw = Z[rows] - Z[rows].mean(0)
        _, Sw, Vh = np.linalg.svd(Zw, full_matrices=False)
        e = Sw ** 2
        Pw = Vh[:min(k, Vh.shape[0])].T
        cosang = np.linalg.svd(Pw.T @ P, compute_uv=False)
        sub = Zw[rngc.choice(len(Zw), size=min(1500, len(Zw)), replace=False)]
        C = np.corrcoef(sub, rowvar=False)
        off = C[np.triu_indices_from(C, 1)]
        sp = speed[rows]; db = dirbin[rows]
        ctx[w] = [rate.mean(), np.median(rate), float((rate < 0.5).mean()), float((rate >= 0.5).sum()),
                  float(np.nanmean(off)), float(e[0] / e.sum()), float((e.sum() ** 2) / (e ** 2).sum()),
                  float(np.degrees(np.mean(np.arccos(np.clip(cosang, -1, 1))))),
                  float(sp.mean()), float(moving[rows].mean()),
                  float(sum((db == kk).sum() >= cfg["min_dir_samples"] for kk in range(K)))]
    return dict(Z=Z, vel=vel, pos=pos, ts=ts, burnin=bm, dirbin=dirbin.astype(np.int16),
                mu0=Z[bm].mean(0), sd0=Z[bm].std(0) + 1e-6, C0=C0, C0k=C0k, P=P, Zref=Z[bm],
                dirC=dirC, dirOK=dirOK, v_mu0=vbm.mean(0), v_cov0=np.cov(vbm, rowvar=False) + 1e-6 * np.eye(2),
                m0=m0, s0=s0, fit_mask=fit_mask, eval_mask=eval_mask, valid_w=valid_w,
                ctx=ctx, ctx_names=np.array(ctx_names),
                n_units=d, dur=dur), (W, int(valid_w.sum()))


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    data_dir = args.data or os.path.join(ROOT, cfg["data_dir"])
    out_dir = args.out or os.path.join(ROOT, cfg["artifact_dir"])
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".nwb"))
    if args.n:
        files = files[:args.n]
    print(f"config: burnin={cfg['burnin_frac']} windows={cfg['n_windows']} pca_k={cfg['pca_k']}")
    print(f"{len(files)} session(s) -> {out_dir}")
    for f in files:
        t1 = time.time()
        try:
            art, (W, nv) = build(os.path.join(data_dir, f), cfg)
        except Exception as exc:  # noqa: BLE001
            print("SKIP", f, exc); continue
        sess = f.split("ses-")[1].split("_")[0]
        np.savez_compressed(os.path.join(out_dir, f"{sess}.npz"), **art)
        print(f"  {sess}: units={art['n_units']} T={len(art['ts'])} dur={art['dur']/60:.1f}min "
              f"burnin={int(art['burnin'].sum())} windows_valid={nv}/{W} "
              f"dirs_ok={int(art['dirOK'].sum())} ({time.time()-t1:.1f}s)")
        wv = np.where(art["valid_w"])[0]
        if len(wv):
            i0 = int(wv[0])
            print("    ctx[%d]: " % i0 + "  ".join(
                f"{nm}={art['ctx'][i0, i]:.3g}" for i, nm in enumerate(art["ctx_names"])))
    print(f"[resources] total={time.time()-t0:.1f}s "
          f"peakRSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")
    print("CACHE_DONE")


if __name__ == "__main__":
    main()
