#!/usr/bin/env python3
"""L1c selftest for common.evaluate_cell — the shared cell logic (synthetic, no pytest).

Run:  python grid_within_session_27/selftest_common.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C      # noqa: E402
import metrics as M     # noqa: E402
import adapters as A    # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAILS.append(name)


class LinDec:
    def __init__(self, W):
        self.coef_ = np.atleast_2d(W)
        self.intercept_ = np.zeros(2)

    def predict(self, X):
        return np.asarray(X) @ self.coef_.T + self.intercept_


class LagDec(LinDec):
    L = 5

    def predict(self, X):
        return np.asarray(X)[self.L:] @ self.coef_.T + self.intercept_


class NoopAd(A.Adapter):
    name = "noop_test"
    stage = "feature"

    def fit(self, *a, **k):
        self.noop = True
        return self

    def apply(self, X):
        return X


def main():
    rng = np.random.default_rng(0)
    n, d = 800, 12
    W = rng.normal(size=(2, d))
    dec = LinDec(W)
    Z = rng.normal(size=(n, d))
    y = Z @ W.T + rng.normal(scale=0.2, size=(n, 2))
    ref = dict(mu0=Z.mean(0), sd0=Z.std(0) + 1e-6, C0=np.cov(Z, rowvar=False) + 1e-6 * np.eye(d),
               P=np.linalg.svd(Z - Z.mean(0), full_matrices=False)[2][:5].T, Zref=Z,
               dirC=np.zeros((8, d)), dirOK=np.zeros(8, bool), v_mu0=y.mean(0),
               v_cov0=np.cov(y, rowvar=False) + 1e-6 * np.eye(2))
    tr, te = slice(0, 500), slice(500, 800)
    Ztr, ytr, Zte, yte = Z[tr], y[tr], Z[te], y[te]

    print("no adapter:")
    m0 = C.evaluate_cell(dec, Ztr, ytr, Zte, yte, ref=ref)
    check("no_adapter_flagged_noop", m0["noop"] is True)
    check("no_adapter_r2_high", m0["r2_all"] > 0.9, f"r2={m0['r2_all']:.3f}")
    check("n_eval_matches", m0["n_eval"] == len(Zte))
    check("has_all_metric_keys", all(k in m0 for k in ("r2_vx", "ang_bias_deg", "lag_bins", "mse_var")))

    print("\nidentity control:")
    mi = C.evaluate_cell(dec, Ztr, ytr, Zte, yte, ref=ref, adapter=A.get("identity"))
    check("identity_predictions_identical", abs(mi["r2_all"] - m0["r2_all"]) < 1e-12)
    check("identity_not_flagged_noop", mi["noop"] is False)

    print("\nfeature-stage adapter:")
    mf = C.evaluate_cell(dec, Ztr, ytr, Zte, yte, ref=ref, adapter=A.get("mom_diag"))
    check("mom_diag_finite", np.isfinite(mf["r2_all"]), f"r2={mf['r2_all']:.3f}")
    check("mom_diag_flagged_noop_false", mf["noop"] is False)
    check("corr_size_identity_zero", C.correction_size(A.get("identity"), dec, Zte)["corr_rel"] < 1e-9)

    print("\noutput-stage adapter:")
    mo = C.evaluate_cell(dec, Ztr, ytr, Zte, yte, ref=ref, adapter=A.get("out_mom"))
    check("out_mom_finite", np.isfinite(mo["r2_all"]), f"r2={mo['r2_all']:.3f}")
    check("out_mom_n_eval_full", mo["n_eval"] == len(Zte))

    print("\nlagged-decoder alignment:")
    decL = LagDec(W)
    mL = C.evaluate_cell(decL, Ztr, ytr, Zte, yte, ref=ref)
    check("lagged_n_eval_shorter", mL["n_eval"] == len(Zte) - decL.L, f"{mL['n_eval']}")
    pL = decL.predict(Zte)
    manual = M.velocity_metrics(yte[-len(pL):], pL)["r2_all"]
    check("lagged_r2_matches_manual_tail_align", abs(mL["r2_all"] - manual) < 1e-12)
    check("align_tail_trims", C.align_tail(np.arange(10), np.zeros(7))[0] == 3)

    print("\nno-op detection:")
    mn = C.evaluate_cell(dec, Ztr, ytr, Zte, yte, ref=ref, adapter=NoopAd())
    check("noop_adapter_flagged", mn["noop"] is True)
    check("noop_equals_base", abs(mn["r2_all"] - m0["r2_all"]) < 1e-12)
    check("corr_size_noop_zero", C.correction_size(NoopAd().fit(), dec, Zte)["corr_rel"] == 0.0)

    print("\ncontext helper:")
    ctx = np.arange(22, dtype=float).reshape(2, 11)
    d0 = C.ctx_row_to_dict(ctx, ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"], 1)
    check("ctx_row_to_dict", d0["a"] == 11.0 and len(d0) == 11)

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
