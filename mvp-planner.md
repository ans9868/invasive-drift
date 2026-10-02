# mvp-planner.md — plan for getting the MVP to work on Torch

Companion to `drafts/10_mvp_hackathon.md` (the concept) and `mvp/README.md` (the runbook).

---

## ⚠️ SAFETY RULES ON TORCH (read first)

**Never run destructive commands against `$SCRATCH` as a whole.** `$SCRATCH` contains much more than
this project (Conn2Conn, SLN_reruns, eeg-*, conda_storage, ...). Rules:

- **Do NOT** run `rm -rf $SCRATCH`, `rm -rf /scratch/ans9868/*`, or any wildcard delete at that level.
- **Do NOT** `scancel -u $USER` (kills *all* your jobs). Cancel jobs by explicit id only.
- All work for this project lives strictly inside **`$SCRATCH/invasive-drift/`**.
- To reset the project dir, operate *inside* it and only on its own contents
  (e.g. `cd $SCRATCH/invasive-drift && rm -rf mvp/out`), never on a parent path.
- Data goes in `$SCRATCH/invasive-drift/data/` (gitignored), outputs in `mvp/out/` (gitignored).
- When in doubt, **don't delete — rename or move aside**.

**Audit of what Cline has done on Torch so far:** only `mkdir -p $SCRATCH`,
`git clone ... $SCRATCH/invasive-drift`, a single pre-clone `rm -rf invasive-drift` (relative path,
inside `$SCRATCH`, the project dir only), and `git pull`. **No other deletions.**

---

## Objective (one sentence)

A chronic same-probe recording → cheap spike-derived features from session *N* predict session *N+1*'s
decoder drop **better than persistence**; report it as one figure + one number.

---

## Dataset sizes (for the MVP)

| Dataset | Total | Per session | Role |
|---|---|---|---|
| **FALCON M1-A** `000941` | **~0.30 GB** (11 assets) | ~28 MB | Phase-0 sanity (smallest) |
| FALCON M1-B `001209` | ~0.22 GB | ~19 MB | 2nd monkey (optional) |
| **Perich `000688`** | **~12.3 GiB** (111 sessions) | **~119 MB** | Phase-1 spine (one subject ~30 sessions ≈ 3.5 GB) |
| FALCON H2 `000950` | ~1.14 GiB | — | human (optional later) |

**MVP budget:** Phase 0 ≈ **0.3 GB**; Phase 1 ≈ **3.5 GB** (one subject); everything ≈ **12.3 GiB**.

---

## Environment (one-time)

`$SCRATCH/conda_storage/kalman` already has numpy/scipy/sklearn/matplotlib/pandas (no pynwb/dandi).
Either reuse it (`pip install pynwb dandi`) or make a fresh env:

```bash
conda create -y -p $SCRATCH/conda_storage/invasive-drift python=3.11
conda activate $SCRATCH/conda_storage/invasive-drift
pip install -r $SCRATCH/invasive-drift/mvp/requirements.txt
python $SCRATCH/invasive-drift/mvp/scripts/00_check_env.py
```

---

## Build steps (checklist)

- [ ] **S0 — env.** `00_check_env.py` prints `ENV OK`.
- [ ] **S1 — list data.** `01_list_data.py` (DANDI metadata only, cheap).
- [ ] **S2 — grab Phase-0 data.** FALCON M1-A (~0.3 GB) → `data/falcon/m1a/`.
- [ ] **S3 — validate I/O.** `02_inspect_nwb.py <one>.nwb` → confirm `units` (spike times) + `trials`/cursor.
- [ ] **S4 — decoder + health(t).** `03_decode_health.py`: ridge decode velocity; frozen day-1 → `health(t)`.
      **CHECKPOINT:** does `health(t)` decay? If not → stop (wrong data/decoder).
- [ ] **S5 — features + forecast.** `04_forecast.py`: reduced panel (rate, unit count, match-survival)
      predict `ΔR²(t+1)`; **persistence** baseline; AUROC/corr.
- [ ] **S6 — Perich spine.** One-subject chain (~30 sessions, ~3.5 GB); rerun S4–S5.
- [ ] **S7 — figure + number.** `out/forecast.png` + one sentence.

**Kill criteria:** no decay in `health(t)` (S4) or panel ≤ persistence (S5) → ship the honest null.

---

## Sync workflow (git)

Local is source of truth; Torch pulls.

```bash
# local
cd /Volumes/CrucialX6/Home/projects/invasive-drift && git add -A && git commit -m "..." && git push
# torch
cd $SCRATCH/invasive-drift && git pull --ff-only
```

---

## Deliverable

`mvp/out/forecast.png` (health(t) + panel-vs-persistence) and one number
("panel AUROC ___ vs persistence ___").

---

## Open choices (from `drafts/09` §10)

1. Decoder: **ridge** (default) vs Kalman.
2. Horizon: **t+1** (default).
3. Crash threshold τ: blind peek at `health(t)` marginal, pre-registered.
4. Matching: spike-train only (no waveforms).
5. Task: center-out (`CO`) first.
