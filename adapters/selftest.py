#!/usr/bin/env python3
"""Selftest for the adapters library — L0 synthetic checks (no data, no pytest).

Run:  python adapters/selftest.py
Exits non-zero on any failure.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import adapters as A  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAILS.append(name)


def make_ref(rng, d=12, k=4, K=8, n=800):
    Z = rng.normal(size=(n, d)) @ (rng.normal(size=(d, d)) * 0.5) + rng.normal(size=d) * 0.3
    mu0, sd0 = Z.mean(0), Z.std(0) + 1e-6
    C0 = np.cov(Z, rowvar=False) + 1e-6 * np.eye(d)
    Vt = np.linalg.svd(Z - mu0, full_matrices=False)[2]
    P = Vt[:k].T
    C0k = np.cov((Z - mu0) @ P, rowvar=False) + 1e-6 * np.eye(k)
    v = rng.normal(size=(n, 2))
    return dict(mu0=mu0, sd0=sd0, C0=C0, P=P, C0k=C0k, Zref=Z, dirC=rng.normal(size=(K, d)),
                dirOK=np.ones(K, bool), v_mu0=v.mean(0), v_cov0=np.cov(v, rowvar=False) + 1e-6 * np.eye(2))


def topk(X, k):
    return np.linalg.svd(X - X.mean(0), full_matrices=False)[2][:k].T


def mean_angle(P, Q):
    s = np.linalg.svd(P.T @ Q, compute_uv=False)
    return float(np.mean(np.arccos(np.clip(s, -1, 1))))


class LinDec:
    def __init__(self, W):
        self.coef_ = W
        self.intercept_ = np.zeros(2)

    def predict(self, X):
        return X @ self.coef_.T + self.intercept_


def main():
    rng = np.random.default_rng(0)
    ref = make_ref(rng)
    d = ref["mu0"].size
    Zf = rng.normal(size=(600, d)) * rng.normal(size=d).clip(0.5, 2) + 3.0   # deliberately shifted
    dec = LinDec(rng.normal(size=(2, d)))

    print("registry:")
    names = set(A.list_adapters())
    for nm in ["identity", "shuffled_ref", "mom_global", "mom_diag", "mom_diag_self", "zca",
               "cov_lowrank", "subspace", "centroid_proc", "null_proj", "out_mom", "out_affine"]:
        check(f"registered:{nm}", nm in names)
    check("all_have_stage", all(a["stage"] in ("feature", "output") for a in A.describe_all()))

    print("\nidentity / controls:")
    ad = A.get("identity").fit(Zf, ref)
    check("identity_is_exact", np.allclose(ad.apply(Zf), Zf))
    s1 = A.get("shuffled_ref").fit(Zf, ref)
    s2 = A.get("shuffled_ref").fit(Zf, ref)
    check("shuffled_ref_deterministic", np.allclose(s1.perm, s2.perm))
    check("shuffled_ref_differs_from_mom_diag",
          not np.allclose(s1.apply(Zf), A.get("mom_diag").fit(Zf, ref).apply(Zf)))

    print("\nmoment matching:")
    md = A.get("mom_diag").fit(Zf, ref)
    out = md.apply(Zf)
    check("mom_diag_matches_mean", np.allclose(out.mean(0), ref["mu0"], atol=1e-8))
    check("mom_diag_matches_std", np.allclose(out.std(0), ref["sd0"], atol=1e-6))
    mds = A.get("mom_diag_self").fit(Zf, ref)
    out2 = mds.apply(rng.normal(size=(300, d)) * 5 + 7)
    check("mom_diag_self_matches_mean", np.allclose(out2.mean(0), ref["mu0"], atol=1e-8))
    check("mom_diag_self_is_non_causal", mds.causal is False)

    print("\ncovariance alignment:")
    zc = A.get("zca").fit(Zf, ref)
    Cz = np.cov(zc.apply(Zf), rowvar=False)
    check("zca_matches_cov", np.allclose(Cz, ref["C0"], atol=2e-2), f"maxerr={np.abs(Cz-ref['C0']).max():.3f}")
    cl = A.get("cov_lowrank").fit(Zf, ref)
    Pl = cl.P
    Cl = np.cov((cl.apply(Zf) - ref["mu0"]) @ Pl, rowvar=False)
    check("cov_lowrank_matches_cov_k", np.allclose(Cl, ref["C0k"], atol=3e-2))
    cc = A.get("subspace").fit(Zf, ref)
    check("subspace_shape", cc.apply(Zf).shape == Zf.shape)
    k5 = 5
    ang_raw = mean_angle(topk(Zf, k5), topk(ref["Zref"], k5))
    ang_aln = mean_angle(topk(cc.apply(Zf), k5), topk(ref["Zref"], k5))
    check("subspace_reduces_subspace_angle", ang_aln < ang_raw,
          f"raw={np.degrees(ang_raw):.1f}deg -> aligned={np.degrees(ang_aln):.1f}deg")

    print("\ngray landmark / aligned:")
    dirbin = rng.integers(0, 8, size=len(Zf))
    ref["dirC"] = np.array([Zf[dirbin == k].mean(0) for k in range(8)])
    cp = A.get("centroid_proc").fit(Zf, ref, dirbin=dirbin)
    check("centroid_proc_shape", cp.apply(Zf).shape == Zf.shape)
    check("centroid_proc_is_gray", cp.uses_targets is True)
    npj = A.get("null_proj").fit(Zf, ref, decoder=dec)
    delta = npj.apply(Zf) - Zf
    check("null_proj_lives_in_rowspace", np.allclose(delta @ npj.PW, delta, atol=1e-8))
    nop = A.get("null_proj").fit(Zf, ref, decoder=object())
    check("null_proj_noop_without_coef", np.allclose(nop.apply(Zf), Zf))

    print("\noutput-stage (decoder-aligned):")
    om = A.get("out_mom").fit(Zf, ref, decoder=dec)
    Vc = om.apply(dec.predict(Zf))
    check("out_mom_matches_v_mu0", np.allclose(Vc.mean(0), ref["v_mu0"], atol=1e-6))
    check("out_mom_stage_output", om.stage == "output")
    oa = A.get("out_affine").fit(Zf, ref, decoder=dec, y=dec.predict(Zf))
    check("out_affine_recovers_identity", np.allclose(oa.apply(dec.predict(Zf)), dec.predict(Zf), atol=1e-6))
    check("out_affine_needs_labels", oa.uses_labels is True)

    print("\ncontract:")
    for nm in A.list_adapters():
        a = A.get(nm)
        try:
            a.fit(Zf, ref, decoder=dec, y=None, dirbin=dirbin)
            inp = dec.predict(Zf) if a.stage == "output" else Zf
            ok = a.apply(inp).shape[0] == Zf.shape[0]
        except Exception as exc:  # noqa: BLE001
            ok = False
            print("        err:", nm, exc)
        check(f"contract:{nm}", ok)

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
