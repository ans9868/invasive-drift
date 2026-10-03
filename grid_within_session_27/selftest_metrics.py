#!/usr/bin/env python3
"""L0 selftest for grid_within_session_27/metrics.py — synthetic, no pytest.

Run:  python grid_within_session_27/selftest_metrics.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAILS.append(name)


def main():
    rng = np.random.default_rng(0)
    n = 600
    v = np.cumsum(rng.normal(size=(n, 2)), axis=0)
    v = (v - v.mean(0)) / v.std(0)
    moving = np.ones(n, bool)

    print("velocity:")
    check("perfect_predictor_r2_is_1", abs(M.velocity_metrics(v, v)["r2_all"] - 1.0) < 1e-9)
    vm = M.velocity_metrics(v, np.broadcast_to(v.mean(0), v.shape))
    check("mean_predictor_r2_is_0", abs(vm["r2_all"]) < 1e-9, f"r2={vm['r2_all']:.2e}")
    p = v + rng.normal(scale=0.3, size=v.shape) + np.array([0.5, -0.25])
    m = M.velocity_metrics(v, p)
    check("mse = bias2 + var", abs(m["mse"] - (m["mse_bias2"] + m["mse_var"])) < 1e-9,
          f"mse={m['mse']:.6f}")
    check("bias_recovered", abs(m["bias_vx"] - 0.5) < 0.02 and abs(m["bias_vy"] + 0.25) < 0.02)
    check("slope_halved_when_gain_is_2x",
          abs(M.velocity_metrics(v, 2 * v)["slope_vx"] - 0.5) < 0.02)

    print("\nlag:")
    L = 7
    pl = np.zeros_like(v); pl[L:] = v[:-L]; pl[:L] = v[0]
    check("lag_recovers_delay", M.lag_bins(v, pl) == L, f"lag={M.lag_bins(v, pl)}")
    check("lag_zero_when_aligned", abs(M.lag_bins(v, v)) == 0)

    print("\ndirection / speed:")
    spd = 1.0 + rng.random(n)
    y = np.c_[spd * np.cos(np.radians(45)), spd * np.sin(np.radians(45))]
    prot = np.c_[spd * np.cos(np.radians(75)), spd * np.sin(np.radians(75))]
    d = M.direction_metrics(y, prot, moving)
    check("ang_bias_recovered", abs(d["ang_bias_deg"] - 30.0) < 0.5, f"bias={d['ang_bias_deg']:.2f}")
    check("resultant_R_is_1_for_constant_bias", abs(d["ang_resultant_R"] - 1.0) < 1e-6)
    check("circ_std_is_0_for_constant_bias", d["circ_std_deg"] < 0.01)
    check("speed_ratio_is_2_when_doubled", abs(M.direction_metrics(y, 2 * y, moving)["speed_ratio"] - 2.0) < 1e-6)
    check("dir_acc8_is_1_when_identical", abs(M.direction_metrics(y, y, moving)["dir_acc8"] - 1.0) < 1e-9)
    check("dir_acc8_drops_under_30deg_rotation", d["dir_acc8"] < 0.5, f"acc8={d['dir_acc8']:.2f}")
    check("wrap180_bounds", np.all(np.abs(M.wrap180([-540, 181, -181, 359])) <= 180))

    print("\nagreement / reliability / kf:")
    a = v + rng.normal(scale=0.2, size=v.shape)
    b = v + rng.normal(scale=0.2, size=v.shape)
    ag = M.agreement_metrics([a, b], v)
    check("ensemble_disagree_positive", ag["ens_disagree"] > 0)
    ag2 = M.agreement_metrics([a, a], v)
    check("identical_decoders_err_corr_1", abs(ag2["err_corr_mean"] - 1.0) < 1e-9)
    check("identical_decoders_disagree_0", abs(ag2["ens_disagree"]) < 1e-12)
    check("consensus_ge_worst", ag["r2_consensus"] >= min(M.velocity_metrics(v, a)["r2_all"],
                                                          M.velocity_metrics(v, b)["r2_all"]) - 1e-9)
    check("reliability_of_identical_is_1", abs(M.reliability(a, a) - 1.0) < 1e-9)
    check("reliability_of_independent_is_low", abs(M.reliability(a, rng.normal(size=v.shape))) < 0.3)
    k = M.kf_uncertainty(np.linspace(1.0, 3.0, 100))
    check("kf_growth_positive", k["kf_post_var_growth"] > 0, f"growth={k['kf_post_var_growth']:.3f}")

    print("\nbaselines:")
    bl = M.baseline_metrics(v, lags=(1, 12))
    check("baseline_mean_r2_is_0", abs(bl["r2_mean"]) < 1e-9)
    check("persistence_lag1_is_high_on_smooth", bl["r2_persist_lag1"] > 0.9,
          f"lag1={bl['r2_persist_lag1']:.3f}")
    check("persistence_lag12_finite", np.isfinite(bl["r2_persist_lag12"]),
          f"lag12={bl['r2_persist_lag12']:.3f}")
    dirbin = rng.integers(0, 8, size=n)
    refdv = np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)])
    check("r2_target_finite", np.isfinite(M.baseline_metrics(v, dirbin, refdv)["r2_target"]))

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
