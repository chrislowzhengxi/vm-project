# Report source audit

Every major number in `main.tex` traced to the saved output it comes from. Paths are relative to the repository root.
No experiment was rerun. The two report figures were built from saved CSVs by `report/scripts/make_report_figures.py`.

## Data and split (Section 2)

| Claim in report | Value | Source |
|---|---|---|
| VMs in the trace; VMs created during the trace | 2,695,548; 2,468,232 | `FINAL_FINDINGS.md` §3 (from `results/01_dataset_stats`) |
| Train / validation / test rows | 1,703,020 / 275,143 / 426,846 | `results/sensitivity/s10_runlog.json`; `FINAL_FINDINGS.md` §3 |
| Subscriptions per split (before the label rule) | 3,015 / 1,462 / 1,726 | `results/02_split_and_history_coverage.csv` |
| Label rule drops 2.5% of training VMs and 57% of training vCPU-hours | — | `FINAL_FINDINGS.md` §3 |
| Test share with ≥1 / ≥1,000 completed subscription VMs | 99.9% / 91.0% | `results/02_split_and_history_coverage.csv`; `results/final_segments_b5.md` |
| Test mean P95 | 46.4% | `results/02_split_and_history_coverage.csv` |
| Right-censored test VMs; share living under 1 h | 4.7%; 68.7% (18.9 + 49.8) | `results/final_segments_b5.md` |
| Feature counts: 26 = 7 + 13 + 6 | — | `src/common.py` (FEATS_A / FEATS_B / FEATS_C) |

## Methods (Section 3)

| Claim | Source |
|---|---|
| 8 OPT features, ≤32 bins, depth 1–3, minimum leaf 0.1–5% | `src/final_setup.py`; `results/final_feature_selection.csv`; `data/processed/final/binner.json` |
| 46,673 patterns; about 5 s per tree; 67 s for the grid | `FINAL_FINDINGS.md` §4; `results/final_05_log.txt` |
| Gurobi cross-check (86 / 212 s optimal at depth 2; 600 s limit with 7.5–13.1% gaps at depth 3); IAI matched 3 of 4 | `results/final_mio_crosscheck.md` |
| Depth 4: 28–36 min, 1.6 GB, certified | `results/sensitivity/s11_e7_depth4.md`; `results/sensitivity/depth4/run_*.json` |
| Best fixed tier is 50 / 100 / 100% | `results/final_validation_choices.csv`; `results/final_main_table.csv` |

## Main results (Table 2, Section 4)

| Claim | Source |
|---|---|
| All β = 5 columns of Table 2 | `results/extended_evaluation/ext_main_table_b5.csv` (reproduces `results/final_main_table_b5.md`, which the stage-12 consistency checks assert) |
| β = 2 and β = 10 savings | `results/final_cost_by_beta.md`; `results/extended_evaluation/ext_main_table_b{2,10}.csv` |
| OPT 20.1% saving, interval [14.1%, 26.9%] | `FINAL_FINDINGS.md` §1 (from `results/final_bootstrap.csv`, OPT vs Best fixed) |
| OPT − H1, OPT − cost-aware RF, OPT − greedy (all three β) | `results/final_bootstrap.csv` |
| OPT keeps 89–93% of the RF's savings; RF 1.6–1.9% cheaper | ratios from `results/final_cost_by_beta.md` |
| vCPU and vCPU-hour weighting lowers savings by 1–5 points | `results/final_weighted_reduction.md` |
| Ablation: 6.0 / 5.4 / 7.2 → 20.3 / 20.2 / 20.9 → 20.1 / 20.0 / 21.6; ≤0.9 at β = 10 | `results/final_ablation.md` |
| R² of 0.525 / 0.494 / 0.491; RMSE 25.0 pp; 16.2% under-predicted by more than 25 pp | `results/extended_evaluation/ext_prediction_quality.csv`; `results/final_prediction_quality.csv` |
| Margin rules differ by at most 2.2 points | `results/extended_evaluation/ext_prediction_vs_decision.csv` |
| Training-cost gain over greedy of 0.4–0.5%; greedy cheaper on the 200k sample; exact tree stable | `FINAL_FINDINGS.md` §6; `results/final_stability.md`; `results/final_tree_grid.csv` |
| Targeting: 26.6 pp less reservation; 6.5 pp penalty; random assignment gives 8.0 pp shortfall and −13.4% saving; 93.3% reservation at equal shortfall | `results/extended_evaluation/ext_savings_mechanism_b5.csv` |
| Tree content (Figure 1) | `results/final_models/OPT-exact_full_b5.txt`; `figures/final_fig4_opt_tree.png` |

## Refinements from the book (Table 3, Section 5)

| Claim | Source |
|---|---|
| k-means and OCT costs, savings, E[r], P(Y>r) and shortfall | `results/hw6/hw6_cluster_results_b{2,5,10}.md` |
| K grid, selection of K = 8 without hour, silhouette 0.39 vs 0.22 | `results/hw6/hw6_cluster_kmeans_grid.md` |
| Skewness 4.1 → 1.7 (vCPU) and 2.7 → −0.5 (VM count) | `results/hw6/hw6_cluster_feature_skew.md` |
| OCT and CART agreement 61.6 / 86.1 / 88.9% and 74.5 / 90.8% | `results/hw6/hw6_cluster_oct_grid.md` |
| Cluster profiles | `results/hw6/hw6_cluster_profiles.md` |
| Bootstrap: k-means − OPT, OCT − k-means | `results/hw6/hw6_cluster_bootstrap.md` |
| Cluster under-provisioning 4.5% vs OPT 3.4% at β = 10 | `results/hw6/hw6_cluster_results_b10.md` |
| Rules A / B / C (main forest, msl = 50) | `results/hw6/hw6_forest_results_b{2,5,10}.md`, rows "[main forest, msl=50]" |
| C − B, C − A, C − OPT, C − cost-aware RF intervals | `results/hw6/hw6_forest_bootstrap.md` |
| C vs B risk at β = 5 and β = 10 (intervals excluding 0 for shortfall and severe shortfall) | `results/extended_evaluation/ext_bootstrap.csv` |
| Calibration within 3.4 pp; the 40–50% bin (24%, P10–P90 of 3–84%, 41% vs 16% exceeded, 75% mean reservation) | `results/extended_evaluation/ext_rf_calibration.md` |
| Weighted shortfall calibration 0.284 / 0.285, … | `results/hw6/hw6_forest_calibration.md` |

## Sensitivity and robustness (Figure 2, Table 4, Section 6)

| Claim | Source |
|---|---|
| Savings across 12 β values | `results/sensitivity/s10_e2_saving_by_beta.csv` (Figure 2a) |
| Switch points 4.46 (validation) and 5.12 (test); 19.6% vs hindsight | `results/sensitivity/s10_e1_best_fixed_switches.md`; `s10_e1_headline_robustness.md` |
| Regret below 1% in [4, 8]; 2–3% at 3 and 10; 15–17% at 1.5 and 20 | `results/sensitivity/s10_e4_regret_vs_tuned.md` |
| Figure 2b frontier | `results/extended_evaluation/ext_risk_frontier_grid.csv`; `ext_savings_mechanism.csv` |
| Table 4: cost, E[r], P(Y>r) | `results/extended_evaluation/ext_main_table_b5.csv` |
| Table 4: Δ and intervals | `results/extended_evaluation/ext_bootstrap.csv` (equal to `results/sensitivity/s11_e5_four_vs_eight_leaves`, `s11_e6_selection`, `s11_e7_selected_vs_existing`); greedy and RF rows from `results/final_bootstrap.csv` (sign flipped) |
| Table 4: equal-subscription columns | `results/extended_evaluation/ext_weighting_comparison_b5.csv` |
| 4-leaf keeps 98.6% (94–100% across β) | `results/sensitivity/s11_e5_four_vs_eight_leaves.md` |
| Depth 4 gains 0.5–1.0%; at β = 10, exceedance 3.4 → 4.6% and reservation −1.5 pp | `results/extended_evaluation/ext_main_table_b10.csv`; `ext_bootstrap.csv` |
| Oracle exceedance of 9.9% and 24%; severity 13–14 vs 2.5 pp; P95 of 36.7 vs 28.6 pp | `results/extended_evaluation/ext_shortfall_severity_b{2,5}.csv` |
| 80% of exceedances have Y > 75% | `results/extended_evaluation/ext_segments.csv` (79.5% for OPT) |
| 0.7% two tiers below; 19% two or more tiers above | `results/extended_evaluation/ext_tier_errors_b5.md` |
| RF vs OPT +2.3 pts exceedance at β = 2 | `results/extended_evaluation/ext_bootstrap.csv` |
| Customer heterogeneity: median 10.4%; 94 subscriptions, 78% of VMs, 85% of saving; top-10 subscriptions 10.4%; 11.9% vs 7.6–7.7% higher cost; 5.6% losing >10% | `results/extended_evaluation/ext_customer_heterogeneity.csv` |
| Equal-subscription differences of 0.006–0.013 with intervals excluding 0 | `results/extended_evaluation/ext_bootstrap_equal_subscription.csv` |
| Validation equal-subscription exceedance 15.9% vs 12.5–13.9% | `results/extended_evaluation/ext_weighting_comparison_validation.csv` |
| Smaller minimum leaf would give 0.796 | `FINAL_FINDINGS.md` §9 (from `results/final_tree_grid.csv`, value 0.7957) |

## Discrepancies found and how they were resolved

1. **Refinement 2 rules A and B at β = 5.** `HW6_FINDINGS.md` and the task brief quote 0.856 / 0.801. Those come
   from the validation-selected forest for each rule (`min_samples_leaf` = 500 for A and B at β = 5). The report
   compares all three rules on *one identical* forest (msl = 50), as the refinement intends, so it uses 0.858 / 0.802.
   Both versions appear in `hw6_forest_results_b5.md`. C is the same in both (0.786).
2. **Forest-weighted saving at β = 2.** This is 13.25%, exactly 13.2498%
   (`hw6_forest_bootstrap.md`, "% of B" = −13.2498). The report rounds it to 13.2.
   `EXTENDED_EVALUATION_FINDINGS.md` §2 and `SENSITIVITY_FINDINGS.md` print 13.3 (rounding 13.250 at 3 decimals).
   This is a rounding difference only.
3. **C − B at β = 5.** `HW6_FINDINGS.md` gives −0.0157 for the validation-selected forests and −0.0167 for the
   identical forest. The report uses −0.0167, consistent with item 1.
4. **Cluster saving at β = 5.** 19.655% (k-means) and 19.508% (OCT): the "19.7%" in the brief is confirmed.

No other conflicts were found. The stage-12 run asserts that its recomputed metrics equal the stage-6, HW6 and
stage-11 tables (84 checks).

## Added in revision 2

| Claim | Source |
|---|---|
| Figure 1 (tree): splits and tiers | `results/final_models/OPT-exact_full_b5.json`. Leaf test shares and mean P95 come from applying that saved tree to `data/processed/final/Xb_opt.npy` (prediction only), in `report/scripts/make_report_figures.py`. Output: 19.9 / 5.0 / 32.9 / 6.0 / 7.5 / 8.3 / 16.3 / 4.3%, identical to `figures/final_fig4_opt_tree.png` and `FINAL_FINDINGS.md` §8. |
| Figure 2 (ablation) | `results/final_ablation.csv` |
| Figure 3a shaded bands (validation switch points 1.60 / 2.29 / 4.46) | `results/sensitivity/s10_e1_best_fixed_switches.csv` (asserted in the figure script) |
| RF restricted to the 8 tree features: 0.787 vs OPT 0.799 at β = 5 | `results/final_audit_rf_same_features.md` (0.7865) |
| "8–12% exceed, 1.1–1.7 pp expected shortfall, 2.2–3.5% severe" | Table 2 rows, excluding rounded-up prediction and the oracle (`ext_main_table_b5.csv`) |
| Conclusion: about 9% exceed, by 14 pp on average, 2–3% by more than 20 pp | OPT row of `ext_main_table_b5.csv` (9.25%, 14.09 pp, 2.56%) |
| "A customer with 40,000 VMs counts 10,000 times one with four" | Arithmetic illustration. The largest test subscription has 49,423 VMs (`ext_customer_per_subscription_b5.csv`). |
| Broad ordering stable across 12 β; cost-aware RF significantly cheaper than OPT at all 12 | `results/sensitivity/s10_e2_saving_by_beta.csv`; `s10_e2_bootstrap.csv` (CI excludes 0 at 12/12) |

The wording "ranking identical at all 12 β" was removed. In the saving table, RF + margin is slightly above OPT at
β = 1.5 and β = 3, so only the broad ordering is claimed.

## Added in revision 3 (10-page version)

| Claim | Source |
|---|---|
| Table 2, solver cross-check: times, statuses, gaps, and IAI and greedy relative to the optimum | `results/final_mio_crosscheck.md`. Gaps are the `gap` column; "vs. opt." is objective ÷ exact DP objective − 1, e.g. 0.798658 / 0.794759 = +0.49%. The 5,000-VM depth-3 Gurobi gap is the cold-start run (7.9%; the warm start gives 7.5%). |
| IAI aborted on the full training set | `FINAL_FINDINGS.md` §4 (IAI limitations) |
| "Remaining 0.2 points come from the 18 extra features" | `results/final_audit_rf_same_features.md`: 21.59 − 21.35 = 0.24 points at β = 5 |
| Ridge matches the RF with margins at β = 5 | `results/final_main_table_b5.md` (Ridge + margin 19.776%, RF + margin 19.768%) |
| OPT 17.6% with vCPU-hour weights | `results/final_weighted_reduction.md` |
| 23–32% of subscriptions unchanged | `ext_customer_heterogeneity.csv`, column "% subs equal" (22.8–31.8) |
| Figure 4 (customers): per-subscription distribution and % worse off by size | `ext_customer_per_subscription_b5.csv`; `ext_customer_heterogeneity.csv`. The 4-leaf tree has 0.0% for the ≥1,000 stratum (verified). |
| Deployment history: 20.1% and 15.3% | `results/final_ablation.csv` |
