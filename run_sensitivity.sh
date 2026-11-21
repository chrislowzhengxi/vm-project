#!/usr/bin/env bash
# Beta-sensitivity and OPT-complexity study (stages 10-11) on top of the finalized pipeline and HW6 outputs.
# Default (~20-25 min): tests + E1-E4 (stage 10) + E5 frontier + E6 leaf-penalty path (stage 11).
# Exact depth-4 OPT (E7) is expensive and runs only with --depth4 (each run has a wall-clock budget).
#   ./run_sensitivity.sh                 # E1-E6
#   ./run_sensitivity.sh --reuse-e3      # reuse cached E3 change-point paths
#   ./run_sensitivity.sh --depth4        # + E7: beta=5 probe (32 bins, 30 min), then beta 2/5/10 at min leaf 1%
#   DEPTH4_BUDGET=30 DEPTH4_BUDGET_B=75 ./run_sensitivity.sh --depth4   (minutes; probe / stage-B budgets)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for c in python3 /usr/local/bin/python3 python; do
    if command -v "$c" >/dev/null && "$c" -c "import pandas, sklearn, pyarrow" 2>/dev/null; then PY=$c; break; fi
  done
fi
[ -n "$PY" ] || { echo "no Python with pandas/scikit-learn/pyarrow found; set PYTHON"; exit 1; }
for f in data/processed/features.parquet data/processed/final/Xb_opt.npy data/processed/hw6/forest_actions_test.parquet; do
  [ -f "$f" ] || { echo "missing $f: run run_all.sh, run_final.sh and run_hw6.sh first"; exit 1; }
done
DEPTH4=0; REUSE=""
for a in "$@"; do
  case "$a" in --depth4) DEPTH4=1 ;; --reuse-e3) REUSE="--reuse-e3" ;; *) echo "unknown flag $a"; exit 1 ;; esac
done
BUDGET=${DEPTH4_BUDGET:-30}
BUDGET_B=${DEPTH4_BUDGET_B:-75}
mkdir -p results/sensitivity
"$PY" -m pytest -q tests/
"$PY" src/10_beta_sensitivity.py $REUSE | tee results/sensitivity/s10_log.txt
"$PY" src/11_opt_complexity.py --part frontier | tee results/sensitivity/s11_frontier_log.txt
"$PY" src/11_opt_complexity.py --part penalty $([ -n "$REUSE" ] && echo --reuse) | tee results/sensitivity/s11_penalty_log.txt
if [ "$DEPTH4" = 1 ]; then
  # Stage A probe (primary configuration), then Stage B in parallel (single-threaded solves)
  "$PY" src/11_opt_complexity.py --part depth4 --beta 5 --min-leaf-frac 0.05 --bins 32 --budget-min "$BUDGET"
  "$PY" src/11_opt_complexity.py --part depth4 --beta 5 --min-leaf-frac 0.01 --bins 32 --budget-min "$BUDGET_B" &
  "$PY" src/11_opt_complexity.py --part depth4 --beta 2 --min-leaf-frac 0.01 --bins 32 --budget-min "$BUDGET_B" &
  "$PY" src/11_opt_complexity.py --part depth4 --beta 10 --min-leaf-frac 0.01 --bins 32 --budget-min "$BUDGET_B" &
  wait
  "$PY" src/11_opt_complexity.py --part depth4-summary | tee results/sensitivity/s11_depth4_log.txt
fi
