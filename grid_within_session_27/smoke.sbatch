#!/bin/bash
#SBATCH --job-name=id_smoke
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ans9868/invasive-drift/logs/id_smoke_%j.out
#SBATCH --error=/scratch/ans9868/invasive-drift/logs/id_smoke_%j.err
set -uo pipefail
export PYTHONUNBUFFERED=1
cd /scratch/ans9868/invasive-drift
E=/scratch/ans9868/conda_storage/kalman/bin/python
echo "=== L0 selftest (adapters) ==="
"$E" adapters/selftest.py
echo "=== L2 tracer (cache 1 session) ==="
"$E" grid_within_session_27/cache.py --n 1
echo ID_SMOKE_DONE
