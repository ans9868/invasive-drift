#!/bin/bash
#SBATCH --job-name=id_smoke
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
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
echo "=== L3 tracer (run_grid 1 session x 5 decoders x 13 adapters x 4 N) ==="
"$E" grid_within_session_27/run_grid.py --sessions 1
echo "=== L3c: new pre-run columns present? ==="
"$E" - <<'PY'
import csv, glob, os, time
fs = sorted(glob.glob("results/raw/*.csv"), key=os.path.getmtime)
if not fs:
    print("  *** NO tracer CSV WRITTEN ***")
    print("  L3_COL_CHECK: FAIL")
else:
    f = fs[-1]
    age = time.time() - os.path.getmtime(f)
    stale = age > 300
    print("  file=%s  age=%.0fs  %s" % (f, age,
          "*** STALE -> L3 CRASHED; this is an OLD file ***" if stale else "(fresh)"))
    r = next(csv.DictReader(open(f)))
    want = ["r2_vw", "r2_all", "form", "label_use", "aligned", "causal",
            "refit_available", "r2_burnin_in", "r2_burnin_out", "eval_contiguous"]
    bad = [k for k in want if r.get(k) is None]
    for k in want:
        v = r.get(k, None)
        print("  %-18s = %s" % (k, "*** MISSING ***" if v is None else v))
    print("  n_cols = %d" % len(r))
    print("  L3_COL_CHECK:", "PASS" if (not bad and not stale) else "FAIL")
PY
echo ID_SMOKE_DONE
