#!/bin/bash
#SBATCH --job-name=id_perich
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ans9868/invasive-drift/logs/id_perich_%j.out
#SBATCH --error=/scratch/ans9868/invasive-drift/logs/id_perich_%j.err
set -uo pipefail
cd /scratch/ans9868/invasive-drift
E=/scratch/ans9868/conda_storage/kalman/bin/python
"$E" mvp/scripts/04_perich_health.py "$@"
echo "PERICH_DONE"
