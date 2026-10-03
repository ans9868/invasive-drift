#!/bin/bash
#SBATCH --job-name=id_fc2
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ans9868/invasive-drift/logs/id_fc2_%j.out
#SBATCH --error=/scratch/ans9868/invasive-drift/logs/id_fc2_%j.err
set -uo pipefail
export PYTHONUNBUFFERED=1
cd /scratch/ans9868/invasive-drift
E=/scratch/ans9868/conda_storage/kalman/bin/python
"$E" mvp/scripts/08_within_forecast.py "$@"
echo FC2_DONE
