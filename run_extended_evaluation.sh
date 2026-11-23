#!/usr/bin/env bash
# Extended evaluation beyond modeled cost (stage 12): operational risk, savings mechanism, cost-vs-risk frontiers,
# customer heterogeneity, workload segments, predictive quality, oracle regret, disagreement, focused bootstrap.
# Reads saved actions / predictions / trees from stages 5-11 and HW6; no policy is retrained (~30 s + tests).
#   ./run_extended_evaluation.sh                  # full test suite, then stage 12
#   ./run_extended_evaluation.sh --skip-tests     # stage 12 only
#   N_BOOT=1000 MIN_SUB_VMS=5 ./run_extended_evaluation.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for c in python3 /usr/local/bin/python3 python; do
    if command -v "$c" >/dev/null && "$c" -c "import pandas, sklearn, pyarrow" 2>/dev/null; then PY=$c; break; fi
  done
fi
[ -n "$PY" ] || { echo "no Python with pandas/scikit-learn/pyarrow found; set PYTHON"; exit 1; }
for f in data/processed/features.parquet data/processed/final/Xb_opt.npy data/processed/final/policy_actions_test.parquet \
         data/processed/hw6/forest_actions_test.parquet data/processed/sensitivity/forest_msl50_preds.npz \
         data/processed/sensitivity/e2_policies.pkl results/sensitivity/s11_e6_selection.csv; do
  [ -f "$f" ] || { echo "missing $f: run run_all.sh, run_final.sh, run_hw6.sh and run_sensitivity.sh (--depth4) first"; exit 1; }
done
ls results/sensitivity/depth4/run_*.json >/dev/null 2>&1 || { echo "missing depth-4 runs: run ./run_sensitivity.sh --depth4"; exit 1; }
SKIP=0
for a in "$@"; do
  case "$a" in --skip-tests) SKIP=1 ;; *) echo "unknown flag $a"; exit 1 ;; esac
done
mkdir -p results/extended_evaluation
[ "$SKIP" = 1 ] || "$PY" -m pytest -q tests/
"$PY" src/12_extended_evaluation.py --n-boot "${N_BOOT:-1000}" --min-sub-vms "${MIN_SUB_VMS:-5}" \
  | tee results/extended_evaluation/ext_log.txt
