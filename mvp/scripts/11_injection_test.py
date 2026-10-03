#!/usr/bin/env python3
"""Injection + coverage test: can a const-accel smoother recover known slope/accel/step at n~15,
and does it beat the historical-slope baseline? (Cheap confirmatory test, plan step 2.)"""
import argparse
import numpy as np


def kalman_smooth_known(z, t, sig, q=(1e-4, 1e-5, 1e-6)):
    """Const-accel model with IRREGULAR dt and KNOWN per-point obs noise sig. Returns (xs, Ps)."""
    T = len(z); n = 3
    xs = np.zeros((T, n)); Ps = np.zeros((T, n, n))
    x = np.array([z[0], 0.0, 0.0]); P = np.eye(n) * 1.0
    F = np.eye(n); Fs = []
    for i in range(T):
        if i > 0:
            dt = t[i] - t[i - 1]
            F = np.array([[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]], float)
            x = F @ x; P = F @ P @ F.T + np.diag(q)
        H = np.array([[1.0, 0, 0]]); R = np.array([[sig[i] ** 2]])
        S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
        x = x + (K @ (z[i] - H @ x)).ravel(); P = (np.eye(n) - K @ H) @ P
        xs[i] = x; Ps[i] = P; Fs.append(F)
    xs_s = xs.copy(); Ps_s = Ps.copy()
    for i in range(T - 2, -1, -1):
        F = Fs[i + 1]; Pp = F @ Ps[i] @ F.T + np.diag(q)
        C = Ps[i] @ F.T @ np.linalg.inv(Pp)
        xs_s[i] = xs[i] + C @ (xs_s[i + 1] - F @ xs[i])
    return xs_s, Ps_s


def make_series(nb, block_min, h0, slope, accel, step, step_frac, sigma, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(nb) * block_min
    mu = h0 + slope * t + 0.5 * accel * t * t
    if step != 0:
        mu = mu + step * (t > step_frac * t[-1])
    z = mu + rng.normal(0, sigma, nb)
    return t, mu, z


def coverage(nb=15, block_min=2.0, h0=0.6, sigma=0.05, nrep=400, slope=-0.02, accel=0.0,
             step=0.0, step_frac=0.5):
    cov_s = cov_a = 0.0; es = ea = 0.0
    for r in range(nrep):
        t, mu, z = make_series(nb, block_min, h0, slope, accel, step, step_frac, sigma, r)
        xs, Ps = kalman_smooth_known(z, t, np.full(nb, sigma))
        m = nb - 1  # end of session
        se = 1.96 * np.sqrt(Ps[m, 1, 1]); cov_s += (abs(xs[m, 1] - slope) <= se); es += xs[m, 1] - slope
        ae = 1.96 * np.sqrt(Ps[m, 2, 2]); cov_a += (abs(xs[m, 2] - accel) <= ae); ea += xs[m, 2] - accel
    return cov_s / nrep, es / nrep, cov_a / nrep, ea / nrep


def walkforward(nb=15, block_min=2.0, h0=0.6, sigma=0.05, nrep=400, slope=-0.02, accel=0.0,
                step=0.0, step_frac=0.5):
    """1-block-ahead MSE ratio vs persistence; hist-slope = h(t)+median past delta; model = smoother pred."""
    mse = {"persistence": 0.0, "hist_slope": 0.0, "model": 0.0}
    cnt = 0
    for r in range(nrep):
        t, mu, z = make_series(nb, block_min, h0, slope, accel, step, step_frac, sigma, r)
        for i in range(4, nb - 1):
            hist = z[:i + 1]
            d = np.median(np.diff(hist))
            p_pers = z[i]
            p_hist = z[i] + d
            xs, Ps = kalman_smooth_known(hist, t[:i + 1], np.full(i + 1, sigma))
            p_model = xs[i, 0] + xs[i, 1] * block_min + 0.5 * xs[i, 2] * block_min * block_min
            y = z[i + 1]
            mse["persistence"] += (y - p_pers) ** 2
            mse["hist_slope"] += (y - p_hist) ** 2
            mse["model"] += (y - p_model) ** 2
            cnt += 1
    base = mse["persistence"]
    return {k: 1 - v / base for k, v in mse.items()}, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", type=int, default=15)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--nrep", type=int, default=400)
    a = ap.parse_args()
    print(f"=== INJECTION COVERAGE (n={a.nb} blocks of 2min, sigma={a.sigma}, target 95%) ===")
    print(f"{'config':18s} {'slope_cov':>9s} {'slope_bias':>11s} {'accel_cov':>9s} {'accel_bias':>11s}")
    cases = [("flat", -0.0, 0.0, 0.0), ("slope=-0.02/min", -0.02, 0.0, 0.0),
             ("slope=-0.05/min", -0.05, 0.0, 0.0), ("accel=-0.002/min2", -0.01, -0.002, 0.0),
             ("step=-0.30", 0.0, 0.0, -0.30)]
    for name, s, ac, st in cases:
        cs, bs, ca, ba = coverage(a.nb, 2.0, 0.6, a.sigma, a.nrep, s, ac, st)
        print(f"{name:18s} {cs:9.2f} {bs:11.4f} {ca:9.2f} {ba:11.4f}")
    print(f"\n=== WALK-FORWARD 1-block-ahead skill (1 - MSE/MSE_persist), n={a.nb} ===")
    print(f"{'config':18s} {'persist':>9s} {'hist_slope':>11s} {'model':>9s}")
    for name, s, ac, st in cases:
        sk, cnt = walkforward(a.nb, 2.0, 0.6, a.sigma, a.nrep, s, ac, st)
        print(f"{name:18s} {sk['persistence']:9.2f} {sk['hist_slope']:11.3f} {sk['model']:9.3f}  (nscore={cnt})")


if __name__ == "__main__":
    main()
