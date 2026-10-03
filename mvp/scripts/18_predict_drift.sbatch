#!/bin/bash
#SBATCH --job-name=id_pd
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/ans9868/invasive-drift/logs/id_pd_%j.out
#SBATCH --error=/scratch/ans9868/invasive-drift/logs/id_pd_%j.err
set -uo pipefail
export PYTHONUNBUFFERED=1
cd /scratch/ans9868/invasive-drift
E=/scratch/ans9868/conda_storage/kalman/bin/python
"$E" mvp/scripts/18_predict_drift.py "$@"
echo ID_PD_DONE
