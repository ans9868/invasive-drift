#!/bin/bash
#SBATCH --job-name=id_grid
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --array=1-53
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/ans9868/invasive-drift/logs/id_grid_%A_%a.out
#SBATCH --error=/scratch/ans9868/invasive-drift/logs/id_grid_%A_%a.err
# Run the adapter grid for ONE session -> results/raw/<sess>.csv (one file per task, so a
# failure is isolated and re-runnable). Requires the v3 cache to exist first (cache.sbatch).
#   --session-index N : 1-based index into the SORTED artifact list
#   (preferred over --session, whose substring matching can select >1 artifact and race)
set -uo pipefail
export PYTHONUNBUFFERED=1
# Point ALL temp/scratch usage at $SCRATCH. The login-node /tmp is a 6 GB tmpfs that is 100% FULL
# (df: 6.0G/6.0G/0), and /scratch/ans9868/export_paths.sh is NOT sourced in non-interactive shells
# (bash skips .bashrc), so TMPDIR arrives UNSET and numpy/LAPACK/ssh default into that full tmpfs.
export TMPDIR="${SCRATCH:-/scratch/${USER}}/invasive-drift/tmp"
mkdir -p "$TMPDIR"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
cd /scratch/ans9868/invasive-drift
E=/scratch/ans9868/conda_storage/kalman/bin/python
echo "node=$(hostname) task=${SLURM_ARRAY_TASK_ID} cpus=${SLURM_CPUS_PER_TASK} TMPDIR=${TMPDIR}"
"$E" grid_within_session_27/run_grid.py --session-index "$SLURM_ARRAY_TASK_ID"
rc=$?
echo "GRID_TASK_DONE $SLURM_ARRAY_TASK_ID rc=$rc"
