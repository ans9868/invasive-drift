#!/bin/bash
#SBATCH --job-name=id_smoke
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ans9868/invasive-drift/logs/id_smoke_%j.out
#SBATCH --error=/scratch/ans9868/invasive-drift/logs/id_smoke_%j.err
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
echo "=== L0 selftest (adapters) ==="
"$E" adapters/selftest.py
echo "=== L0 selftest (metrics) ==="
"$E" grid_within_session_27/selftest_metrics.py
echo "=== L1b selftest (decoder cache round-trip) ==="
"$E" grid_within_session_27/selftest_cache.py
echo "=== L1c selftest (common.evaluate_cell) ==="
"$E" grid_within_session_27/selftest_common.py
echo "=== L2 tracer (cache 1 session + decoders) ==="
"$E" grid_within_session_27/cache.py --n 1 --with-decoders
echo "=== L3 tracer: CANARY session CO-20131220 (the one that exercised the std-floor blow-up) ==="
"$E" grid_within_session_27/run_grid.py --session CO-20131220
echo "=== L3c: columns + VALUE-RANGE GUARD ==="
"$E" - <<'PY'
import csv, os, time
p = "results/raw/CO-20131220.csv"
if not os.path.exists(p):
    print("  *** canary CSV MISSING -> L3 crashed ***")
    print("  L3_CHECK: FAIL")
else:
    age = time.time() - os.path.getmtime(p)
    fh = open(p); r = next(csv.DictReader(fh))
    want = ["r2_vw", "r2_all", "form", "label_use", "aligned", "causal",
            "refit_available", "r2_burnin_in", "r2_burnin_out", "eval_contiguous"]
    bad = [k for k in want if r.get(k) is None]
    n = 0; mx = 0.0
    for row in csv.DictReader(open(p)):
        n += 1
        mx = max(mx, abs(float(row["r2_all"])))
    print("  file=%s  age=%.0fs  rows=%d  cols=%d" % (p, age, n, len(r)))
    print("  max|r2_all| = %.4g     (GUARD: must be < 1e4; blow-ups were ~1e6-1e9)" % mx)
    for k in bad:
        print("     *** MISSING column:", k)
    ok = (not bad) and age <= 900 and mx < 1e4
    print("  L3_CHECK:", "PASS" if ok else "FAIL")
PY
echo ID_SMOKE_DONE
