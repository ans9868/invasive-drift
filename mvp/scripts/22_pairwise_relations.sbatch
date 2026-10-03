#!/bin/bash
#SBATCH --job-name=id_pwr
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/ans9868/invasive-drift/logs/id_pwr_%j.out
#SBATCH --error=/scratch/ans9868/invasive-drift/logs/id_pwr_%j.err
set -uo pipefail
export PYTHONUNBUFFERED=1
cd /scratch/ans9868/invasive-drift
E=/scratch/ans9868/conda_storage/kalman/bin/python
"$E" mvp/scripts/22_pairwise_relations.py "$@"
echo ID_PWR_DONE
