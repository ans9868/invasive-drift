#!/bin/bash
#SBATCH --job-name=id_wf
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ans9868/invasive-drift/logs/id_wf_%j.out
#SBATCH --error=/scratch/ans9868/invasive-drift/logs/id_wf_%j.err
set -uo pipefail
export PYTHONUNBUFFERED=1
cd /scratch/ans9868/invasive-drift
E=/scratch/ans9868/conda_storage/kalman/bin/python
"$E" mvp/scripts/12_waveform_degradation.py "$@"
echo WF_DONE
