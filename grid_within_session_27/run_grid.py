#!/usr/bin/env python3
"""run_grid.py — the adapter grid (P3+P4 merged). NO windows (Idea 19).

Split (PLAN §3): one 20/80 for the DECODER (burn-in -> online) and one 80/20 for the ADAPTER
(first 80% of online -> last 20%), with N = a prefix of the adapter's fit pool (causal).

Per session x N x adapter x decoder:
  fit adapter on the fit-pool prefix -> apply -> decode the EVAL -> full metric row (long format).

Frozen decoders come from the pickle cache (config-hash checked); refit on miss.
Writes one CSV per session.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)                      # adapters/ lives at the repo root
sys.path.insert(0, os.path.join(ROOT, "mvp"))
import adapters as A      # noqa: E402
import metrics as M       # noqa: E402
import common as C        # noqa: E402
import cache as CACHE     # noqa: E402
from common import align_tail  # noqa: E402

AGREE_KEYS = ("pred_corr_mean", "ens_disagree", "err_corr_mean", "r2_consensus")


def r2(p, y):
    return float(1.0 - ((p - y) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-12))


def fit_dec(name, obj, Z, y, pos):
    (obj.fit(Z, pos, y) if name.startswith("kf") else obj.fit(Z, y))
    return obj


def build_ref(z):
    return dict(mu0=z["mu0"], sd0=z["sd0"], C0=z["C0"], C0k=z["C0k"], P=z["P"], Zref=z["Zref"],
                dirC=z["dirC"], dirOK=z["dirOK"], v_mu0=z["v_mu0"], v_cov0=z["v_cov0"])


def adapter_meta(ad_name):
    """Family metadata, so cells can be grouped at ANALYSIS time (see findings/..._adapter_grid_framing.md).

    REQUIRED pre-run: the grid discards the adapter object, so without these columns P1/P2/P3
    (moment family vs direction-only family vs shuffled control) cannot be evaluated from the CSV.
    """
    if ad_name == "none":
        return dict(form="frozen", label_use="none", aligned=False, causal=True,
                    uses_targets=False, uses_decoder=False, is_trainable=False)
    d = A.get(ad_name).describe()
    if d["uses_labels"]:
        lu = "labeled"
    elif d["uses_targets"]:
        lu = "gray"                      # target identity, not velocity (PLAN §2)
    else:
        lu = "unlabeled"
    return dict(form=d["stage"], label_use=lu, aligned=bool(d["uses_decoder"]),
                causal=bool(d["causal"]), uses_targets=bool(d["uses_targets"]),
                uses_decoder=bool(d["uses_decoder"]), is_trainable=bool(d["is_trainable"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--data", default=None)
    ap.add_argument("--nwb", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--session", default=None)
    ap.add_argument("--session-index", type=int, default=0,
                    help="1-based index into the SORTED artifact list (for SLURM job arrays). "
                         "Preferred over --session: substring matching can select >1 artifact.")
    ap.add_argument("--decoders", default=None)
    ap.add_argument("--adapters", default=None)
    ap.add_argument("--N", default=None)
    ap.add_argument("--sessions", type=int, default=0)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    art_dir = args.data or os.path.join(ROOT, cfg["artifact_dir"])
    nwb_dir = args.nwb or os.path.join(ROOT, cfg["data_dir"])
    out_dir = args.out or os.path.join(ROOT, cfg["results_dir"])
    os.makedirs(out_dir, exist_ok=True)
    dec_names = (args.decoders or ",".join(cfg["active_decoders"])).split(",")
    ad_names = ["none"] + (args.adapters.split(",") if args.adapters else list(cfg["adapters"]))
    meta_map = {nm: adapter_meta(nm) for nm in ad_names}
    Ns = [float(x) for x in (args.N.split(",") if args.N else cfg["N_fracs"])]
    refit = set(cfg.get("refit_decoders", []))
    K = cfg["dir_bins"]
    files = sorted(glob.glob(os.path.join(art_dir, "*.npz")))
    if args.session:
        files = [f for f in files if args.session in f]
        if len(files) != 1:
            print(f"  WARN: --session '{args.session}' matched {len(files)} artifacts "
                  f"({[os.path.basename(f) for f in files]}) -> use --session-index")
    if args.session_index:
        n_all = len(files)
        if not 1 <= args.session_index <= n_all:
            print(f"  FATAL: --session-index {args.session_index} out of range 1..{n_all}")
            return 1
        files = [files[args.session_index - 1]]
    if args.sessions:
        files = files[:args.sessions]
    spec_map = dict(CACHE.decoder_specs(dec_names))
    print(f"grid: {len(files)} sessions | decoders={dec_names} | adapters={len(ad_names)} | N={Ns}",
          flush=True)
    t0 = time.time()
    for p in files:
        sess = os.path.basename(p).split(".")[0]
        z = np.load(p, allow_pickle=True)
        Z, vel, pos = z["Z"], z["vel"], z["pos"]
        dirbin = z["dirbin"].astype(int)
        gfit, geval = z["gfit_mask"], z["geval_mask"]
        ref = build_ref(z)
        speed = np.linalg.norm(vel, axis=1)
        moving = speed > np.nanpercentile(speed, 60)
        # ---- frozen decoders: pickle cache (hash-checked) else refit ----
        cands = glob.glob(os.path.join(nwb_dir, f"*{sess}*.nwb"))
        decs, bmeta = None, None
        if cands:
            obj, st = CACHE.load_decoders(os.path.join(art_dir, f"{sess}.decoders.pkl"),
                                          CACHE.config_hash(cfg, cands[0]))
            if st == "ok":
                decs = {n: obj["decoders"][n] for n in dec_names if n in obj["decoders"]}
                bmeta = obj
        if decs is None:
            decs, r2in, r2out = CACHE.fit_decoders(CACHE.decoder_specs(dec_names),
                                                   Z, vel, pos, z["burnin"])
            bmeta = {"r2_burnin_in": r2in, "r2_burnin_out": r2out}
        bin_ = bmeta.get("r2_burnin_in", {}) or {}
        bout_ = bmeta.get("r2_burnin_out", {}) or {}
        # ---- rows ----
        ei = np.where(geval)[0]
        tri_all = np.where(gfit)[0]
        eval_contiguous = bool(len(ei) > 1 and np.all(np.diff(ei) == 1))   # PLAN §5: lag_bins validity
        Zte, yte, mte = Z[ei], vel[ei], moving[ei]
        refdv = np.zeros((K, 2), np.float32)
        for k in range(K):
            m = gfit & (dirbin == k)
            if m.sum() > 20:
                refdv[k] = vel[m].mean(0)
        base = M.baseline_metrics(yte, dirbin[ei], refdv, ref_mean=vel[tri_all].mean(0))
        ctxs = {str(nm): float(v) for nm, v in zip(z["ctx_names"], z["ctx_sess"])}
        # ---- refit ORACLE (in-sample on the eval; per session x decoder) ----
        orac = {}
        for dn in dec_names:
            if dn in refit:
                d = fit_dec(dn, spec_map[dn](), Zte, yte, pos[ei])
                pp = np.asarray(d.predict(Zte))
                orac[dn] = r2(pp, align_tail(yte, pp))
            else:
                orac[dn] = np.nan
        rows = []
        for f in Ns:
            n = max(int(len(tri_all) * f), 1)
            tri = tri_all[:n]
            Ztr, ytr, dtr = Z[tri], vel[tri], dirbin[tri]
            outr = {}
            for dn in dec_names:
                if dn in refit:
                    d = fit_dec(dn, spec_map[dn](), Ztr, ytr, pos[tri])
                    pp = np.asarray(d.predict(Zte))
                    outr[dn] = r2(pp, align_tail(yte, pp))
                else:
                    outr[dn] = np.nan
            for ad_name in ad_names:
                preds, md = {}, {}
                for dn in dec_names:
                    ad = None if ad_name == "none" else A.get(ad_name)
                    m = C.evaluate_cell(decs[dn], Ztr, ytr, Zte, yte, ref=ref, adapter=ad,
                                        dirbin_tr=dtr, moving_te=mte)
                    preds[dn] = m.pop("_pred")
                    md[dn] = m
                mn = min(len(v) for v in preds.values())
                if len(preds) >= 2:
                    P = np.stack([preds[d][len(preds[d]) - mn:] for d in dec_names])
                    agree = M.agreement_metrics(P, yte[len(yte) - mn:])
                else:
                    agree = {k: np.nan for k in AGREE_KEYS}
                for dn in dec_names:
                    rows.append(dict(session_id=sess, split="grid", block_idx=-1,
                                     t_start_min=np.nan, ctx_scope="session",
                                     session_minutes=float(z["session_minutes"]),
                                     n_blocks=int(z["n_blocks"]),
                                     short_recording=bool(z["short_recording"]),
                                     n_units=int(z["n_units"]), decoder=dn, objective=ad_name,
                                     N_frac=f, N_samples=n, seed=cfg.get("seed", 0),
                                     r2_refit_oracle=orac[dn], r2_refit_out=outr[dn],
                                     refit_available=bool(dn in refit),
                                     r2_burnin_in=float(bin_.get(dn, np.nan)),
                                     r2_burnin_out=float(bout_.get(dn, np.nan)),
                                     eval_contiguous=eval_contiguous,
                                     **meta_map[ad_name], **md[dn], **agree, **base, **ctxs))
        keys = sorted({k for r in rows for k in r})
        with open(os.path.join(out_dir, f"{sess}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"  {sess}: {len(rows)} rows -> {sess}.csv  ({time.time()-t0:.1f}s)", flush=True)
    print("GRID_DONE")


if __name__ == "__main__":
    raise SystemExit(main())

