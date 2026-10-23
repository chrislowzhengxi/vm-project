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
