#!/bin/bash
#SBATCH --job-name=id_inj
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ans9868/invasive-drift/logs/id_inj_%j.out
#SBATCH --error=/scratch/ans9868/invasive-drift/logs/id_inj_%j.err
set -uo pipefail
export PYTHONUNBUFFERED=1
cd /scratch/ans9868/invasive-drift
E=/scratch/ans9868/conda_storage/kalman/bin/python
"$E" mvp/scripts/11_injection_test.py "$@"
echo INJ_DONE
