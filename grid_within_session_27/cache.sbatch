#!/bin/bash
#SBATCH --job-name=id_cache
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --array=1-53
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ans9868/invasive-drift/logs/id_cache_%A_%a.out
#SBATCH --error=/scratch/ans9868/invasive-drift/logs/id_cache_%A_%a.err
# Build ONE session of the v3 cache (npz artifact + frozen-decoder pickle).
#   --index N        : 1-based index into the SORTED .nwb list
#   --skip-existing  : resume a partially-completed array without redoing work
# NOTE: the 53 npz now in temp-trash/ are the OLD window-based (cache_version 2) artifacts
# and are INCOMPATIBLE with run_grid (no gfit_mask/geval_mask). They must NOT be restored.
set -uo pipefail
export PYTHONUNBUFFERED=1
# Point ALL temp/scratch usage at $SCRATCH — the login-node /tmp is a 6 GB tmpfs that is 100% FULL,
# and export_paths.sh is NOT sourced in non-interactive shells, so TMPDIR arrives UNSET.
export TMPDIR="${SCRATCH:-/scratch/${USER}}/invasive-drift/tmp"
mkdir -p "$TMPDIR"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
cd /scratch/ans9868/invasive-drift
E=/scratch/ans9868/conda_storage/kalman/bin/python
echo "node=$(hostname) task=${SLURM_ARRAY_TASK_ID} cpus=${SLURM_CPUS_PER_TASK} TMPDIR=${TMPDIR}"
"$E" grid_within_session_27/cache.py --index "$SLURM_ARRAY_TASK_ID" --with-decoders --skip-existing
rc=$?
echo "CACHE_TASK_DONE $SLURM_ARRAY_TASK_ID rc=$rc"
