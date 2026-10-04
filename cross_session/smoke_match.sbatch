#!/bin/bash
#SBATCH --job-name=id_match_smoke
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ans9868/invasive-drift/logs/id_match_smoke_%j.out
#SBATCH --error=/scratch/ans9868/invasive-drift/logs/id_match_smoke_%j.err
# Cross-session Step 0 SMOKE.
#   (a) synthetic selftest of the matcher (partial-overlap recovery, scale invariance, nulls)
#   (b) the REAL loader on 3 sessions with ragged-index diagnostics
# CROSS-CHECK: the published `14` prototype gave median matched corr ~= 0.996 and 63-100% of units
# matching between consecutive sessions. This run must reproduce that ballpark or something is wrong.
set -uo pipefail
export PYTHONUNBUFFERED=1
# Point ALL temp/scratch usage at $SCRATCH (login-node /tmp is a full 6 GB tmpfs; see grid README log 7)
export TMPDIR="${SCRATCH:-/scratch/${USER}}/invasive-drift/tmp"
mkdir -p "$TMPDIR"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
cd /scratch/ans9868/invasive-drift
E=/scratch/ans9868/conda_storage/kalman/bin/python
echo "node=$(hostname) TMPDIR=${TMPDIR}"
FAILS=0
echo "=== L0 selftest (cross_session/match_units) ==="
"$E" cross_session/selftest_match.py || FAILS=$((FAILS + 1))
echo "=== L1 real loader: 3 sessions, ragged-index DIAGNOSTICS ==="
"$E" cross_session/match_units.py --n 3 --diag --out artifacts/cross_session_smoke || FAILS=$((FAILS + 1))
echo "=== L1c summary.csv ==="
cat artifacts/cross_session_smoke/summary.csv 2>/dev/null || { echo "(no summary)"; FAILS=$((FAILS + 1)); }
echo "=== VERDICT ==="
if [ "$FAILS" -eq 0 ]; then echo "MATCH_SMOKE: PASS"; else echo "MATCH_SMOKE: FAIL ($FAILS failing stage(s))"; fi
echo MATCH_SMOKE_DONE
