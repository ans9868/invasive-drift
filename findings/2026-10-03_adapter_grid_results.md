# 2026-10-03 — The adapter grid: RESULTS (within-session, 53 sessions)

**Framing and the pre-registered predictions P1–P5: `2026-10-03_adapter_grid_framing.md`.** This note is the
outcome. It reports the **std-floor-fixed re-run** (`grid_within_session_27/README.md` logs 7–8).

**Scale.** 53 sessions × 5 decoders × **12 objectives** × 4 N = **12,720 rows × 73 cols**, every task
`rc=0`. Within-session only. Continuous, causal, trial-label-free.

**Baseline:** `none` = `identity` = **median R² 0.3171** at N=1.0, and their paired Δ is **exactly
0.0000** across all 1,060 cells — the control is exact.

**Statistics convention (locked):** median, IQR, % improved, and a **paired sign test**. *Not* mean±std —
see §7. All numbers below are **N=1.0** unless stated.

---

## 1. The headline

> **Unsupervised within-session adaptation is a near-null result (+0.008 on a 0.317 baseline = +2.7%
> relative), the one family that tries to fix the *geometry* actively destroys decoding, and the only
> thing that clearly helps is *supervised*.**

And the headline *finding* is P3: **the entire unsupervised gain is generic, not identity-specific.**

## 2. The table

| objective | family | Δ median | IQR | % improved | verdict |
|---|---|---|---|---|---|
| `identity` | control | **+0.0000** | [0, 0] | 0.0% | exact control ✅ |
| `out_affine` *(labeled)* | output | **+0.0146** | — | **90.2%** | **best — but supervised** |
| `mom_diag_self` | moment (2-pass) | +0.0085 | — | 80.8% | tiny |
| **`shuffled_ref`** *(negative control)* | control | **+0.0082** | [+0.0014, +0.0207] | 79.2% | **matches the real thing** |
| `mom_diag` | moment (causal) | +0.0065 | [+0.0004, +0.0145] | 78.1% | tiny |
| `cov_lowrank` | moment | +0.0039 | — | 74.7% | negligible |
| `mom_global` | moment | −0.0009 | — | 44.5% | nothing |
| `zca` | moment | −0.0015 | — | 46.4% | **negative** |
| `centroid_proc` *(gray)* | direction-only | −0.0128 | [−0.0294, −0.0036] | 15.8% | **negative** |
| `out_mom` | output | −0.0655 | — | 15.1% | **harmful** |
| **`subspace`** | direction-only | **−0.1711** | [−0.2942, −0.1095] | **3.8%** | **catastrophic** |

## 3. P1 — the moment ladder is NOT monotone → **FAILS**

| rung (paired, N=1.0) | Δ median | % improved | sign p |
|---|---|---|---|
| `mom_global` − `identity` | −0.0009 | 44.5% | 0.075 |
| **`mom_diag` − `mom_global`** | **+0.0074** | **76.6%** | **≈0** |
| `cov_lowrank` − `mom_diag` | −0.0007 | 47.2% | 0.36 |
| **`zca` − `cov_lowrank`** | **−0.0044** | 36.2% | **7.3e-06** |
| **`zca` − `mom_diag`** | **−0.0058** | 30.2% | **1.1e-10** |

Ladder vs `identity`: `mom_global −0.0009` → `mom_diag +0.0065` → `cov_lowrank +0.0039` → `zca −0.0015`.

**Not monotone.** But there **is one real rung**: **per-unit beats global** (`mom_diag` vs `mom_global`
+0.0074, 76.6% improved, p≈0). Beyond that, higher moment order buys **nothing** — and full-rank
covariance (`zca`) is **significantly worse than plain per-unit moments** (p=1.1e-10). d²≈10,000 params
overfits. **Resolution matters; moment order does not.**

## 4. P2 — direction-only alignment is harmful → **CONFIRMED, emphatically**

| paired vs `identity`, N=1.0 | Δ median | IQR | % improved | sign p |
|---|---|---|---|---|
| **`subspace`** (CLEAN) | **−0.1711** | [−0.2942, −0.1095] | **3.8%** | ≈0 |
| `centroid_proc` (GRAY) | −0.0128 | [−0.0294, −0.0036] | 15.8% | ≈0 |

Harmful in **every** decoder and **every** N:

```
subspace   ridge -0.1374  wiener -0.1519  kf -0.1504  mlp -0.2512  gru -0.1997
by N:      -0.1744       -0.1635        -0.1711      -0.1711
```

Only **3.8%** of cells improved — i.e. **96% got worse**. Note `subspace` is flat across N: it does not
depend on adaptation data, it depends on the *geometry*, and the geometry is wrong.

### 4.1 The sanity check (this was the strongest and biggest number — it deserved one)

Diagnostics at N=1.0 (medians):

| objective | r2_all | corr_vx | corr_vy | ang_err_deg | **speed_ratio** | **corr_rel** | **corr_rank** |
|---|---|---|---|---|---|---|---|
| `none` | 0.317 | 0.585 | 0.592 | 40.98 | **0.582** | 0.000 | 0 |
| `subspace` | **0.118** | **0.345** | **0.280** | **58.45** | **0.287** | **0.892** | **61** |
| `centroid_proc` | 0.297 | 0.563 | 0.579 | 42.07 | 0.581 | 0.168 | **7** |
| `mom_diag_self` | 0.331 | 0.577 | 0.601 | 39.80 | 0.579 | 0.273 | 61 |

`corr_rel = ‖Δ‖/‖base‖`. **Verdict: REAL, not a numerical artifact.**
- `subspace` changes the features by **89% of their own magnitude**, **full-rank (61 units)** — nearly as
  large as the signal itself.
- It **halves the output speed** (0.582 → 0.287) and **decorrelates** (0.585 → 0.345 for vx).
- All magnitudes sane, all finite, no blow-up. `corr_rel < 1`.
- Contrast `centroid_proc`: a **confined** correction (rank **7**, corr_rel 0.168) → far milder damage
  (−0.013 vs −0.171). **Confining the correction limits the damage.**

→ A large full-rank subspace rotation, applied to features whose relationship to the decoder has already
drifted, scrambles the readout. This is exactly what *"the drift is a deformation, not a rotation"*
predicts, and it is the strongest single result in the grid.

## 5. P3 — the per-unit gain is NOT identity-specific → **REFUTED** (the headline finding)

| paired, N=1.0 | Δ median | IQR | % improved | sign p |
|---|---|---|---|---|
| `shuffled_ref` − `identity` *(the CONTROL)* | **+0.0082** | [+0.0014, +0.0207] | 79.2% | ≈0 |
| `mom_diag` − `identity` *(the REAL one)* | +0.0065 | [+0.0004, +0.0145] | 78.1% | ≈0 |
| **`mom_diag` − `shuffled_ref`** | **−0.0024** | [−0.0085, +0.0017] | **34.7%** | **6.5e-07** |
| `mom_diag_self` − `shuffled_ref` | **+0.0000** | [0, 0] | 58.9% | 3.9e-03 |

`shuffled_ref` is the **negative control**: identical math, but the reference moments are **permuted across
units**. It scores **+0.0082**, essentially the same as the real per-unit match (**+0.0085** for
`mom_diag_self`, +0.0065 for `mom_diag`), by decoder too:

```
shuffled_ref   ridge +0.0102  wiener +0.0133  kf +0.0045  mlp +0.0121  gru +0.0033
mom_diag_self  ridge +0.0102  wiener +0.0133  kf +0.0045  mlp +0.0129  gru +0.0037
```

**And it goes further than "no better":** the *correct* correspondence (`mom_diag`) is **significantly
worse** than its own shuffled control (−0.0024, only 34.7% of cells better, p=6.5e-07).

> **Which unit gets which target moments is irrelevant. The gain is generic re-standardisation, and
> imposing the real (noisy) 20 ms per-unit correspondence is marginally *harmful* relative to randomising
> it.**

This independently reproduces `2026-10-03_normalizer_perspective.md` (+0.015, unit-shuffled control
indistinguishable) at **grid scale** — 53 sessions × 5 decoders × 4 N, paired, p=6.5e-07.

**Consequence:** the "per-unit moment matching" family — which is what **NoMAD's per-channel z-score** is —
is not doing what it is believed to do. It is a **global re-scaling effect**, not drift realignment.

## 6. What actually helps: `out_affine`, and it is supervised

| | Δ median | % improved | labeled? |
|---|---|---|---|
| **`out_affine`** | **+0.0146** | **90.2%** | **YES** (uses velocity) |
| `out_mom` | −0.0655 | 15.1% | no |

Both are **output-stage** (2-D velocity space), so the *stage* is not the discriminator — the
**objective** is. `out_affine` fits a 2×2 affine to minimise decode error: that is **cheap supervised
recalibration**, not unsupervised adaptation. `out_mom` — the label-free twin that matches the predicted
velocity's moments — **hurts badly** (−0.0655, and it is the only objective with a large failure count,
105 rows below −1).

**So the honest reading: unsupervised feature adaptation ≈ nothing; unsupervised output-moment matching
hurts; only supervised recalibration helps.**

## 7. Ceilings and context (why "tiny" understates it in one way and overstates it in another)

| anchor | median |
|---|---|
| `r2_persist_lag1` (20 ms persistence) | **0.9941** |
| `r2_target` (direction-condition mean) | **0.5726** |
| `r2_persist_lag12` (240 ms persistence) | 0.4278 |
| `r2_mean` (constant mean) | −0.0000 ✅ sanity |
| **decoder baseline (`none`)** | **0.3171** |

**Headroom framing:** the gap from the 0.317 baseline to the 0.573 target ceiling is **0.256**.
`out_affine` captures **5.7%** of it; `mom_diag_self` **3.3%**; `mom_diag` **2.5%**. That is the honest
size of these effects — and **no** adapter comes close to the ceiling, let alone to persistence.

**Statistics convention (locked, and it matters):** median / IQR / % improved / paired sign test.
**Never mean±std.** Reason: see `grid_within_session_27/README.md` log 7 — a single adapter's std-floor bug
produced 28 rows down to −7.8e9. A **median over 1,060 cells is robust to 2.6% outliers, which is the only
reason it corrupted the tail and not the answer.** A mean-based pipeline would have reported garbage
undetected.

## 8. Caveats

- **Effect sizes are tiny.** +0.008 on 0.317 = **+2.7% relative**, ~3–6% of available headroom. "Beats"
  is weak language for this.
- **P4 is not testable here** — the grid has no time axis (`block_idx=-1`). It belongs to `staleness.py`.
- **Register 2 (`SS_free`, innovation R²) is still undefined** and not in this note.
- **`subspace` ≠ Aligned FA exactly.** `subspace` is *unlabeled orthogonal subspace alignment*; Aligned FA
  uses stable-loading **landmarks**. Treated as one family here — an assumption, explicitly flagged.
- **`out_mom`'s 105 failure rows** are a behavioural signal worth their own look; not investigated here.
- 137 rows remain below −1 (ordinary failures, worst −5.13). No longer astronomical, but they exist.
- The **NoMAD comparison is indicative**: different dataset, different output (EMG vs kinematics).

## 9. Provenance

- **Framing / pre-registered P1–P5:** `2026-10-03_adapter_grid_framing.md`
- **Run + the std-floor fix + verification:** `../grid_within_session_27/README.md` logs 6–8
- **The fix:** `adapters/base.py::safe_scale`; guards in `adapters/selftest.py`, `smoke.sbatch`
- **Data:** Torch `$SCRATCH/invasive-drift/results/raw/` — 53 CSVs, 12,720 rows, 73 cols.
  Grid job `19125433` (53/53 `rc=0`). Pre-fix CSVs at `trash/prev-run-1791065039` + `temp-analysis/`.
- **Analysis:** stdlib-only scripts over the CSVs (median/IQR/paired sign test); no numpy needed.

