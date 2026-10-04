#!/bin/bash
#SBATCH --job-name=id_match
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ans9868/invasive-drift/logs/id_match_%j.out
#SBATCH --error=/scratch/ans9868/invasive-drift/logs/id_match_%j.err
# Cross-session Step 0: unit-correspondence tables for ALL 53 consecutive session pairs.
# Writes artifacts/cross_session/matches/<A>__<B>.npz + summary.csv
# These tables are the gate for any cross-session transfer: without them there is no notion of
# "the same unit on two days", and cross-session decoding is undefined.
set -uo pipefail
export PYTHONUNBUFFERED=1
export TMPDIR="${SCRATCH:-/scratch/${USER}}/invasive-drift/tmp"
mkdir -p "$TMPDIR"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
cd /scratch/ans9868/invasive-drift
E=/scratch/ans9868/conda_storage/kalman/bin/python
echo "node=$(hostname) TMPDIR=${TMPDIR}"
"$E" cross_session/match_units.py --out artifacts/cross_session
echo MATCH_TASK_DONE rc=$?
