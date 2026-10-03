#!/usr/bin/env python3
"""L1b selftest for the DECODER CACHE: fit -> save -> load -> identical predictions, plus the guards.

Tests the REAL save/load code in cache.py (not a copy).
Run:  python grid_within_session_27/selftest_cache.py
"""
import os
import sys
import tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cache as C  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAILS.append(name)


def main():
    rng = np.random.default_rng(0)
    n, d = 1500, 12
    Z = rng.normal(size=(n, d))
    W = rng.normal(size=(d, 2))
    vel = Z @ W + rng.normal(scale=0.5, size=(n, 2))     # real signal so R2 is not ~0
    pos = np.cumsum(vel, axis=0) * 0.02
    bm = np.zeros(n, bool); bm[:700] = True

    specs = C.decoder_specs(["ridge", "wiener", "kf_posvel", "mlp"])
    try:
        import torch  # noqa: F401
        import decoders as D          # mvp/ is on sys.path after the call above
        specs.append(("gru", lambda: D.GRUDec(10, hidden=16, epochs=1, bs=1024)))
    except Exception as exc:  # noqa: BLE001
        print("  (gru round-trip SKIPPED:", exc, ")")

    print("fit on burn-in:")
    decs, r2in, r2out = C.fit_decoders(specs, Z, vel, pos, bm)
    check("all_decoders_fitted", len(decs) == len(specs), f"{list(decs)}")
    check("r2_burnin_in_finite", all(np.isfinite(v) for v in r2in.values()),
          " ".join(f"{k}={v:.3f}" for k, v in r2in.items()))
    check("r2_burnin_out_finite", all(np.isfinite(v) for v in r2out.values()),
          " ".join(f"{k}={v:.3f}" for k, v in r2out.items()))
    check("in_ge_out_all_decoders", all(r2in[k] >= r2out[k] - 1e-6 for k in r2in),
          "gaps=" + " ".join(f"{k}:{r2in[k]-r2out[k]:+.3f}" for k in r2in))
    predA = {nm: np.asarray(decs[nm].predict(Z[bm])) for nm in decs}

    print("\nrow alignment (lagged decoders drop L rows):")
    check("align_tail_trims", C.align_tail(np.arange(10), np.zeros((7, 2))).tolist() == [3, 4, 5, 6, 7, 8, 9])
    for nm in decs:
        check(f"pred_len_le_input:{nm}", len(predA[nm]) <= bm.sum(),
              f"len={len(predA[nm])} vs {int(bm.sum())}")
    if "ridge" in predA:
        check("ridge_keeps_all_rows", len(predA["ridge"]) == int(bm.sum()))
    if "wiener" in predA:
        check("wiener_drops_5_rows", len(predA["wiener"]) == int(bm.sum()) - 5,
              f"len={len(predA['wiener'])}")

    print("\nround-trip:")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.decoders.pkl")
        C.save_decoders(p, decs, r2in, r2out, {"config_hash": "abc123", "env": C.env_versions()})
        obj, st = C.load_decoders(p, "abc123")
        check("status_ok", st == "ok", f"status={st}")
        for nm in decs:
            predB = np.asarray(obj["decoders"][nm].predict(Z[bm]))
            check(f"pred_identical:{nm}", predA[nm].shape == predB.shape and
                  np.allclose(predA[nm], predB, atol=1e-6),
                  f"maxdiff={np.abs(predA[nm]-predB).max():.2e}")
        check("r2_burnin_in_preserved", obj["r2_burnin_in"] == r2in)
        check("r2_burnin_out_preserved", obj["r2_burnin_out"] == r2out)
        check("env_versions_present", "numpy" in obj["meta"]["env"])

    print("\nguards:")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.decoders.pkl")
        C.save_decoders(p, decs, r2in, r2out, {"config_hash": "abc123"})
        _, st2 = C.load_decoders(p, "DIFFERENT")
        check("hash_mismatch_detected", st2 == "hash_mismatch", f"status={st2}")
        _, st3 = C.load_decoders(os.path.join(td, "nope.pkl"), "abc123")
        check("missing_detected", st3 == "missing", f"status={st3}")
        with open(p, "wb") as fh:
            fh.write(b"not a pickle")
        _, st4 = C.load_decoders(p, "abc123")
        check("corrupt_detected", st4.startswith("unreadable"), f"status={st4[:24]}")

    print("\nconfig hash:")
    h1 = C.config_hash({"burnin_frac": 0.2, "bin_ms": 20.0, "tau_ms": 240.0, "dir_bins": 8,
                        "pca_k": 5, "seed": 0, "decoders": ["ridge"]}, __file__)
    h2 = C.config_hash({"burnin_frac": 0.3, "bin_ms": 20.0, "tau_ms": 240.0, "dir_bins": 8,
                        "pca_k": 5, "seed": 0, "decoders": ["ridge"]}, __file__)
    check("hash_stable", h1 == C.config_hash({"burnin_frac": 0.2, "bin_ms": 20.0, "tau_ms": 240.0,
                                             "dir_bins": 8, "pca_k": 5, "seed": 0,
                                             "decoders": ["ridge"]}, __file__))
    check("hash_sensitive_to_config", h1 != h2, f"{h1} vs {h2}")

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
