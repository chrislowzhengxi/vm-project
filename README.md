# VM right-sizing feasibility study (Azure Public Dataset V2)

Question: *given only information available when a VM is created, is there enough signal to predict its
P95 CPU demand and make better individualized CPU reservations than a fixed rule?*

The answer, with all supporting tables and plots, is in **[REPORT.md](REPORT.md)**.

## Data

Only the small per-VM summary tables from the official
[Azure Public Dataset V2](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md)
release are used (no 5-minute telemetry):

| file | compressed | uncompressed | rows |
|---|---|---|---|
| `trace_data_vmtable_vmtable.csv.gz` | 438 MB | 821 MB | 2,695,548 |
| `trace_data_deployments_deployments.csv.gz` | 2.0 MB | 3.1 MB | 33,205 |
| `trace_data_subscriptions_subscriptions.csv.gz` | 0.35 MB | 0.5 MB | 6,687 |

`run_all.sh` downloads them into `data/raw/` if they are missing. `data/raw_docs/` holds copies of the official V2
documentation and link list consulted before downloading.

## How to run

Requirements: Python ≥ 3.10 with `pandas`, `pyarrow`, `numpy`, `scikit-learn`, `matplotlib`, `tabulate`.

```bash
pip install pandas pyarrow numpy scikit-learn matplotlib tabulate
./run_all.sh            # downloads data if needed, then runs the four stages (~15-25 min, ~6 GB RAM)
```

Or run the stages one at a time (each reads the previous stage's output from `data/processed/`):

| stage | script | what it does | runtime |
|---|---|---|---|
| 1 | `src/01_build_profile.py` | load VM table → `vmtable.parquet`; dataset stats, column profile, leakage checks, fig 1-3 | ~15 s |
| 2 | `src/02_features.py` | leakage-safe creation-time + historical features, chronological split → `features.parquet` | ~35 s |
| 3 | `src/03_models.py` | baselines, Ridge, CART, Random Forest; A/B/C ablation; segments; importance; stress tests; fig 4-6 | ~6-15 min |
| 4 | `src/04_policies.py` | reservation tiers, fixed vs. learned policies, λ-costs, policy tree; fig 7-8 | ~2 min |

`src/common.py` holds paths, constants, feature-set definitions and the (training-only) imputation.

## Outputs

* `data/processed/features.parquet` - **the reusable modelling table** (2.47 M rows, one per VM created during the
  trace). Columns prefixed `sub_`, `dep_` and the raw size/time columns are features; `y_p95` is the target;
  `LEAKY_*` and `post_*` columns are kept *only* for leakage/stress checks and must never be used as features;
  `key_sub` / `key_dep` are integer group codes (grouping only, never features); `split` is train/val/test.
* `data/processed/predictions.parquet` - validation/test predictions of the stage-3 models.
* `results/*.md|csv` - every table quoted in the report (prefix = stage).
* `figures/*.png` - the 8 plots used in the report.

## Leakage rules (enforced in `src/02_features.py`)

* Target: `p95maxcpu`. Never used as a feature: `p95maxcpu`, `avgcpu`, `maxcpu`, `vmdeleted`, lifetime, `vmcategory`
  (only assigned to VMs observed for ≥ 5 days), and the deployment table's `deploymentsize` (a whole-trace aggregate).
* A previous VM's `p95maxcpu` only enters a history feature once that VM has been **deleted** before the current VM
  is created (`vmdeleted < vmcreated`, strict), because P95 summarizes the whole lifetime.
* Train = days 0-20, validation = days 21-24, test = days 25-29 (by `vmcreated`); imputation values come from training rows only.

---

## Final project: Optimal Policy Trees (stages 5-7)

Findings: **[FINAL_FINDINGS.md](FINAL_FINDINGS.md)**. The feasibility study above (stages 1-4, `REPORT.md`) is
unchanged; the final stages reuse `data/processed/features.parquet` and write `results/final_*`, `figures/final_*`,
`results/final_models/` and `data/processed/final/`.

```bash
./run_final.sh          # tests + stages 5, 5b, 6, 7 (~40 min on a 10-core laptop); PYTHON=... to pick Python
python -m pytest -q tests/
```

| stage | script | what it does |
|---|---|---|
| lib | `src/final_lib.py` | costs, oracle, evaluation, binning, exact DP policy trees, greedy cost tree, Gurobi MIO, IAI bridge, cluster bootstrap |
| lib | `src/final_setup.py` | strict label-available train/val masks, training-only imputation, sampling, evaluation weights |
| 5 | `src/05_opt_policy_trees.py` | cost-aware RF + validation feature selection; CART-MSE / greedy / OPT-exact / OPT-IAI grids; validation selection |
| 5b | `src/05b_mio_crosscheck.py` | Gurobi MIO vs. exact DP vs. IAI on subsamples (objective, bound, gap, runtime) |
| 6 | `src/06_final_evaluation.py` | all benchmarks (fixed, heuristics, predict-then-decide, cost-aware RF, trees, oracle); weighted results; figs 1, 3, 4 |
| 7 | `src/07_robustness.py` | cluster bootstrap, segments, feature ablation, sample-size/seed stability, strict vs. feasibility setup, censoring; figs 2, 5, 6 |
| - | `src/iai_opt.jl` | Julia driver for `IAI.OptimalTreePolicyMinimizer` |

Optional solvers:
* **Gurobi** (`pip install gurobipy` + license) for stage 5b and one test.
* **Interpretable AI** (academic license at `~/iai.lic`; system image for Julia 1.11.9 at
  `~/Library/Application Support/InterpretableAI/sysimage/v3.2.2/sys.dylib`, overridable with `IAI_SYSIMAGE` /
  `IAI_JULIA`). Without it, OPT-IAI rows are skipped; the exact DP needs only numpy.

---

## Homework 6 refinements (stages 8-9)

Findings: **[HW6_FINDINGS.md](HW6_FINDINGS.md)**. These extensions read the finalized outputs and never write them;
they need `data/processed/features.parquet` and `data/processed/final/policy_actions_*.parquet`.

```bash
./run_hw6.sh            # tests/test_hw6.py + stages 8 and 9 (~5 min)
```

| stage | script | what it does |
|---|---|---|
| lib | `src/hw6_lib.py` | forest leaf tables and weighted expected costs, cluster-to-action mapping, OCT bridge, threshold trees |
| 8 | `src/08_hw6_weighted_prescriptions.py` | P95 forest: point prediction vs. safety margin vs. forest-weighted prescription (Ch. 13) |
| 9 | `src/09_hw6_clustering.py` | k-means clusters → cheapest tier per cluster; IAI OCT interpretation of the clusters (Ch. 20) |
| - | `src/iai_oct.jl` | Julia driver for `IAI.OptimalTreeClassifier` |

Outputs: `results/hw6/`, `figures/hw6/`, `data/processed/hw6/`.

---

## β sensitivity and OPT complexity (stages 10-11)

Findings: **[SENSITIVITY_FINDINGS.md](SENSITIVITY_FINDINGS.md)**. These read the finalized and HW6 outputs and never
write them.

```bash
./run_sensitivity.sh              # tests + E1-E6 (~25 min; ~75 min the first time, E3 change-point paths dominate)
./run_sensitivity.sh --reuse-e3   # reuse cached E3 / E6 paths
./run_sensitivity.sh --depth4     # + exact depth-4 OPT (E7): 30-min probe, then three runs with a 75-min budget
```

| script | what it does |
|---|---|
| `src/sensitivity_lib.py` | fixed-policy cost lines, break-evens, lower envelopes, exact parametric change-point search |
| `src/10_beta_sensitivity.py` | E1 fixed-policy sensitivity; E2 re-optimized 12-point β grid; E3 exact OPT change points; E4 wrong-β regret |
| `src/11_opt_complexity.py` | E5 cost vs leaves; E6 exact leaf-penalty (λ) path; E7 exact depth-4 OPT runs and summary |

`src/final_lib.solve_opt_exact` gained two backward-compatible options, a `leaf_penalty` (default 0) and exact
`depth=4` with a `time_limit`. Defaults reproduce the earlier results bit for bit (see `tests/test_sensitivity.py`).
