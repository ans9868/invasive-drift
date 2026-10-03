#!/bin/bash
#SBATCH --job-name=id_inspect
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/ans9868/invasive-drift/logs/id_inspect_%j.out
#SBATCH --error=/scratch/ans9868/invasive-drift/logs/id_inspect_%j.err
set -uo pipefail
export PYTHONUNBUFFERED=1
cd /scratch/ans9868/invasive-drift
E=/scratch/ans9868/conda_storage/kalman/bin/python
"$E" - <<'PYE'
import glob, h5py
p = sorted(glob.glob("data/perich/sub-C/*.nwb"))[0]
print("file:", p)
with h5py.File(p, "r") as h:
    print("top:", list(h.keys()))
    for grp in ("acquisition", "processing", "intervals", "stimulus"):
        if grp in h:
            print(f"{grp}:", list(h[grp].keys()))
    if "processing" in h:
        for k in h["processing"]:
            print("  processing/", k, list(h["processing"][k].keys()))
    if "intervals" in h:
        for k in h["intervals"]:
            g = h["intervals"][k]
            cols = list(g.keys())
            print(f"  intervals/{k}: n={len(g[cols[0]])} cols={cols}")
    if "units" in h:
        print("units cols:", list(h["units"].keys()))
PYE
echo ID_INSPECT_DONE
