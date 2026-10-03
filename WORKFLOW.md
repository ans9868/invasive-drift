# WORKFLOW — operational rules for this project

Cross-cutting rules. Applies to local (macOS) **and** Torch (HPC).

## 1. Git discipline
- The repo `ans9868/invasive-drift` is the **source of truth**; **local = author**, **Torch = clone**.
- **Push a backup snapshot before starting any new implementation** (`git add -A && commit && push`).
- After every change: **push locally → `git pull --ff-only` on Torch** (`$SCRATCH/invasive-drift`).
- Small, human-readable outputs (summary tables, figures, findings) → **committed**.
  Bulk/derived (data, `.npz`, `.parquet`, raw logs) → **scratch, never committed**.

## 2. Filesystem safety — **NO `rm`**
- **Never `rm` / `rm -rf`.** If something must go, **move it to `trash/`**:
  - local: `invasive-drift/trash/`  ·  Torch: `$SCRATCH/invasive-drift/trash/`
  - `mkdir -p trash && mv <thing> trash/` (timestamp the name if needed).
- All work stays **inside** `$SCRATCH/invasive-drift/`. Touch nothing else in `$SCRATCH`.
- **Login-node `/tmp` is often FULL** — never write there; use `$SCRATCH/invasive-drift/tmp/`.
- Jobs: `scancel <id>` only, by explicit id. Never broad-scope anything.

## 3. SLURM etiquette
- **NEVER poll `squeue`/`sacct` in a loop** — it spams the controller. Use **file sentinels** +
  `tail` on the job's `.out`/`.err`, and a long `sleep` between checks.
- Partition `cpu_short` (idle, no walltime cap). **`--mem=4G` is plenty** (measured ≤1.3 GB) — do not
  over-request; the earlier 48 G→16 G right-sizing was still generous.
- Queue backing up is fine: jobs are independent and will drain one at a time.
- Scripts print **peak RSS + per-stage wall time** themselves (no external polling).

## 4. Reproducibility
- Fixed seeds everywhere (deterministic). Cached artifacts so re-analysis never re-fits.
- Every job writes **append-only, crash-safe output** (one row/file per task).

## 5. Scratch layout
```
$SCRATCH/invasive-drift/
├── repo/  (git clone)  · data/ (read-only) · artifacts/ (derived, regenerable)
├── results/raw/ (per-task) · results/merged/ · logs/ · tmp/ · trash/
```
