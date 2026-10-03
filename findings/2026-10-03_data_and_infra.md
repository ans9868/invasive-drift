# Finding — Data, environment, and infrastructure

**Date:** 2026-10-03

## Datasets
- **Perich & Miller**, DANDI **`000688`** — macaque M1/PMd, chronic arrays, cursor pos/vel, **spike
  waveforms (48 samples/spike)**, 111 sessions. **sub-C = 53 center-out sessions (2013–2016), ~7 GB** on
  Torch `$SCRATCH/invasive-drift/data/perich/`.
- **FALCON M1-A**, DANDI **`000941`** — macaque M1 reach-to-grasp, **EMG** (16 muscles) + spikes;
  11 files, 298 MB. (M1 is an *EMG-decoding* task, not classification.)
- Both are public (DANDI API). Data downloaded directly on the Torch login node.

## Environment
- Env `$SCRATCH/conda_storage/kalman` (python 3.11): numpy/scipy/sklearn/pandas/matplotlib + **h5py**
  (installed) ; **torch** for the GRU. No pynwb needed (h5py reads the NWB directly).

## Compute notes (Torch)
- **Partitions idle**: `cpu_short` 253/260 nodes idle; plenty of capacity; no walltime cap.
- **RAM was over-requested** (jobs used ~1.6 GB, requested 48 GB) → sbatch `--mem` right-sized to **16 G**.
  **Still generous:** `19` (53 sessions, full loop) peaked at **1.14 GB** and ran in ~50 s on 4 CPUs →
  jobs could safely drop to **4 G**. Scripts print `peakRSS` + stage timing themselves (no `squeue` polls).
- **Never poll `squeue`/`sacct` in a loop** (spams the SLURM controller) — use file sentinels + `tail`.
- Login node `/tmp` is often **full** (6 GB tmpfs) → write everything under `$SCRATCH`.
- Jobs run on compute nodes (cs6xx); login node python import is slow (NFS).

## Repo / workflow
- Source of truth: git repo **`ans9868/invasive-drift`** (public), cloned on Torch at
  `$SCRATCH/invasive-drift`. Sync: **push local → pull on Torch**.
- `mvp/decoders.py` — decoder zoo; `mvp/scripts/01..16` — data→decode→drift→forecast pipeline.
- **No deletions** in `$SCRATCH` beyond the project dir; `scancel` only by explicit job id.
