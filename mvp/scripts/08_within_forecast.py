#!/usr/bin/env python3
"""Within-session forecast - RICH features (plan option #1). Same bias-guarded ladder.

Extra per-block features: mean pairwise spike-count correlation, decoder MSE on the block,
rate CV, and L1 per-unit rate change vs previous block (plus rate/act/rate_std).
"""
import argparse, glob, os
import numpy as np
import pandas as pd
import h5py
from scipy.signal import lfilter
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

BIN_MS = 20.0; TAU_MS = 240.0
ALPHAS = np.logspace(-4, 3, 15)
FEATS = ["rate", "act", "rate_std", "corr", "mse", "rate_cv", "drate"]


def load(path):
    with h5py.File(path, "r") as h:
        st = h["units/spike_times"][:]; idx = h["units/spike_times_index"][:]
        b = np.concatenate([[0], idx]); n = len(idx)
        spies = [st[b[u]:b[u + 1]] for u in range(n)]
        vg = h["processing/behavior/Velocity/cursor_vel"]
        ts = vg["timestamps"][:]; y = vg["data"][:].astype(np.float32)
    bs = BIN_MS / 1000.0; edges = np.concatenate([[ts[0] - bs], ts])
    X = np.zeros((len(ts), n), np.float32)
    for u in range(n):
        X[:, u] = np.histogram(spies[u], bins=edges)[0]
    return X, y, ts


def exp_filt(x, tau_ms=TAU_MS, bin_ms=BIN_MS):
    t = np.arange(0.0, tau_ms, bin_ms); k = np.exp(-t / tau_ms); k /= k.sum()
    return lfilter(k, [1.0], x, axis=0).astype(np.float32)


def session_series(path, block_s, ref_frac):
    X, y, ts = load(path); X = exp_filt(X)
    keep = ~np.isnan(y).any(1); X, y, ts = X[keep], y[keep], ts[keep]
    nref = int(ref_frac * len(y))
    m, s = X[:nref].mean(0), X[:nref].std(0); s[s == 0] = 1; Xs = (X - m) / s
    dec = GridSearchCV(Ridge(), {"alpha": ALPHAS}, cv=3).fit(Xs[:nref], y[:nref])
    t0 = ts[0]; nb = int((ts[-1] - t0) // block_s)
    H, F, prev = [], [], None
    for bi in range(nb):
        lo = t0 + bi * block_s; hi = lo + block_s; sel = (ts >= lo) & (ts < hi)
        if sel.sum() < 200:
            continue
        Xb = X[sel]; rates = Xb.mean(0)
        corr = np.nan
        if Xb.shape[0] > 10 and Xb.shape[1] > 1:
            C = np.corrcoef(Xb.T); iu = np.triu_indices_from(C, 1)
            corr = float(np.nanmean(C[iu]))
        mse = float(np.mean((dec.predict(Xs[sel]) - y[sel]) ** 2))
        rate = float(rates.mean()); rate_std = float(rates.std())
        rate_cv = float(rate_std / (rate + 1e-9))
        act = int((rates > 0.05).sum())
        drate = float(np.abs(rates - prev).sum()) if prev is not None else np.nan
        prev = rates
        H.append(dec.score(Xs[sel], y[sel]))
        F.append([rate, act, rate_std, corr, mse, rate_cv, drate])
    return np.array(H), np.array(F)


def r2(y, p):
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/perich/sub-C/*.nwb")
    ap.add_argument("--block-s", type=float, default=300.0)
    ap.add_argument("--ref-frac", type=float, default=0.2)
    args = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(args.glob)):
        try:
            H, F = session_series(p, args.block_s, args.ref_frac)
        except Exception as exc:
            print("SKIP", os.path.basename(p), exc); continue
        if len(H) < 4:
            continue
        sess = os.path.basename(p).split("ses-")[1].split("_")[0]
        print(f"  {sess}: blocks={len(H)} R2 [{H.min():.2f},{H.max():.2f}] corr_med={np.nanmedian(F[:,3]):.2f}")
        for bi in range(len(H) - 1):
            rows.append(dict(sess=sess, h=H[bi], hnext=H[bi + 1], **{k: F[bi, i] for i, k in enumerate(FEATS)}))
    df = pd.DataFrame(rows).dropna()
    sessions = sorted(df.sess.unique())
    print(f"\nrows={len(df)} sessions={len(sessions)} feats={FEATS}")
    y = df.hnext.to_numpy(); h = df.h.to_numpy(); Xf = df[FEATS].to_numpy()
    preds = {k: np.zeros(len(df)) for k in ["chance", "persistence", "trend", "panel", "panel_detr"]}
    for sess in sessions:
        te = (df.sess == sess).to_numpy(); tr = ~te
        preds["chance"][te] = y[tr].mean()
        preds["persistence"][te] = h[te]
        delta = (y[tr] - h[tr]).mean(); preds["trend"][te] = h[te] + delta
        Xp = np.column_stack([Xf, h]); mp, sp = Xp[tr].mean(0), Xp[tr].std(0) + 1e-9
        dec = Ridge(alpha=1.0).fit((Xp[tr] - mp) / sp, y[tr]); preds["panel"][te] = dec.predict((Xp[te] - mp) / sp)
        rr = y[tr] - preds["trend"][tr]; mm, ss = Xf[tr].mean(0), Xf[tr].std(0) + 1e-9
        dec2 = Ridge(alpha=1.0).fit((Xf[tr] - mm) / ss, rr); preds["panel_detr"][te] = preds["trend"][te] + dec2.predict((Xf[te] - mm) / ss)
    print(f"\n{'model':14s} {'R2':>8s} {'MAE':>8s} {'corr':>7s}")
    for k, p in preds.items():
        c = np.corrcoef(y, p)[0, 1] if p.std() > 0 else np.nan
        print(f"{k:14s} {r2(y, p):8.3f} {np.abs(y - p).mean():8.3f} {c:7.3f}")
    print(f"\nheadline: dR2 panel over trend      = {r2(y, preds['panel']) - r2(y, preds['trend']):+.3f}")
    print(f"          dR2 panel_detr over trend = {r2(y, preds['panel_detr']) - r2(y, preds['trend']):+.3f}")


if __name__ == "__main__":
    main()
