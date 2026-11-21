#!/usr/bin/env bash
# Final-project experiments (stages 5-7) on top of the feasibility pipeline (stages 1-2 must have been run:
# data/processed/features.parquet and vmtable.parquet). About 25-35 min on a 10-core laptop.
# Optional: Gurobi (gurobipy + license) for the MIO cross-check; Interpretable AI system image + license for OPT-IAI
# (see README). Set PYTHON=/path/to/python to choose the interpreter.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for c in python3 /usr/local/bin/python3 python; do
    if command -v "$c" >/dev/null && "$c" -c "import pandas, sklearn, pyarrow" 2>/dev/null; then PY=$c; break; fi
  done
fi
[ -n "$PY" ] || { echo "no Python with pandas/scikit-learn/pyarrow found; set PYTHON"; exit 1; }
[ -f data/processed/features.parquet ] || { echo "run ./run_all.sh first (stages 1-2)"; exit 1; }
"$PY" src/final_env.py
"$PY" -m pytest -q tests/
"$PY" src/05_opt_policy_trees.py | tee results/final_05_log.txt
if "$PY" -c "import gurobipy" 2>/dev/null; then
  "$PY" src/05b_mio_crosscheck.py | tee results/final_05b_log.txt
else
  echo "gurobipy not available: skipping the MIO cross-check"
fi
"$PY" src/06_final_evaluation.py | tee results/final_06_log.txt
"$PY" src/07_robustness.py | tee results/final_07_log.txt
