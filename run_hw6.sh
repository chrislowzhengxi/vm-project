#!/usr/bin/env bash
# Homework 6 refinements (stages 8-9) on top of the finalized pipeline. Needs data/processed/features.parquet and
# data/processed/final/policy_actions_*.parquet (from ./run_all.sh and ./run_final.sh); does NOT rerun stages 1-7 and
# never writes their outputs. About 5 minutes on a 10-core laptop. IAI (for the OCT) is optional.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for c in python3 /usr/local/bin/python3 python; do
    if command -v "$c" >/dev/null && "$c" -c "import pandas, sklearn, pyarrow" 2>/dev/null; then PY=$c; break; fi
  done
fi
[ -n "$PY" ] || { echo "no Python with pandas/scikit-learn/pyarrow found; set PYTHON"; exit 1; }
for f in data/processed/features.parquet data/processed/final/policy_actions_test.parquet; do
  [ -f "$f" ] || { echo "missing $f: run ./run_all.sh and ./run_final.sh first"; exit 1; }
done
mkdir -p results/hw6
"$PY" -m pytest -q tests/test_hw6.py
"$PY" src/08_hw6_weighted_prescriptions.py | tee results/hw6/hw6_08_log.txt
"$PY" src/09_hw6_clustering.py | tee results/hw6/hw6_09_log.txt
