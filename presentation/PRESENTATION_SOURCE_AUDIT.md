# Presentation source audit

Every number on the slides is checked against `ML_Final_Report.pdf` (the source of truth) and the saved result files
behind it. Report section numbers refer to the final PDF.

| Slide | Claim | Report location | Saved result |
|---|---|---|---|
| 2 | Cost C(r, Y; β) = r + β max(0, Y − r); tiers 25/50/75/100%; β = 5 main scenario | §2 Decision problem | `src/final_lib.py` |
| 2 | Example costs at Y = 0.40, β = 5: 1.00 / 0.50 / 0.75 / 1.00 | arithmetic from Eq. (1) | — (illustrative VM, labeled as such) |
| 3 | 2,468,232 VMs created during the trace (2.47M); 30 days; one region; 26 creation-time features | §2 Data and features | `FINAL_FINDINGS.md` §3 |
| 3 | Split: train days 0–20 (1,703,020), validation 21–24 (275,143), test 25–29 (426,846) | Table 1 | `results/sensitivity/s10_runlog.json` |
| 3 | History uses only finished VMs; labels must be complete by the cutoff | §2 | `src/02_features.py`, `src/final_setup.py` |
| 4 | Tree splits, thresholds, tiers and leaf shares (19.9 / 5.0 / 32.9 / 6.0 / 7.5 / 8.3 / 16.3 / 4.3% of test VMs) | Figure 1 | `results/final_models/OPT-exact_full_b5.json` (leaf shares recomputed in `report/scripts/make_report_figures.py`) |
| 4 | 8 features × 31 thresholds; certified training optimum within the class; about 5 s per depth-3 tree | §3, §3.1 | `FINAL_FINDINGS.md` §4 |
| 5 | Savings at β = 5: RF round-up 14.2, H1 19.1, RF + margin 19.8, greedy 20.0, OPT 20.1, cost-aware RF 21.6; oracle 41.1 | Table 3 | `results/extended_evaluation/ext_main_table_b5.csv` |
| 5 | OPT − greedy at β = 5 not significant (−0.0013 [−0.0054, 0.0027]) | §4 Exact OPT versus greedy | `results/final_bootstrap.csv` |
| 5 | Best fixed tier at β = 5 is 100% | Table 3 caption, §3 | `results/final_validation_choices.csv` |
| 6 | OPT ablation 6.0 → 20.3 → 20.1; greedy 5.4 / 20.2 / 20.0; cost-aware RF 7.2 / 20.9 / 21.6; β = 10: 0.9 → 15.1 | §4, Figure 2 | `results/final_ablation.csv` |
| 6 | 99.9% of test VMs from subscriptions with an earlier completed VM | §2 Chronological split | `results/02_split_and_history_coverage.csv` |
| 6 | Method adds at most about 1.5 points once history is included | §4 | `results/final_ablation.csv` (21.6 − 20.1) |
| 7 | Same forest: A 0.858 / 14.2%, B 0.802 / 19.8%, C 0.786 / 21.4% | Table 4, §5.2 | `results/hw6/hw6_forest_results_b5.md` (main forest, msl = 50) |
| 7 | VMs predicted 40–50% (24% of test VMs) span all four tiers; 41% vs 16% exceeded | §5.2 | `results/extended_evaluation/ext_rf_calibration.csv` |
| 8 | 4 leaves: 19.8% saving, 7.7% exceedance; 8 leaves: 20.1%, 9.3%; cost-aware RF: 21.6%, 9.1% | Table 5, Table 3 | `results/extended_evaluation/ext_main_table_b5.csv` |
| 8 | 4 leaves keep 98.6% of the 8-leaf savings; difference not significant | §6 Tree complexity | `results/sensitivity/s11_e5_four_vs_eight_leaves.csv` |
| 8 | The 4-leaf tree splits only on customer history (mean past P95, share of past VMs ≤ 75%) | §6 ("two decisions based on customer history") | `results/sensitivity/s11_e5_opt_depth2_b5.txt` |
| 8 | Median subscription saves 10.4% under OPT vs 20.1% overall | §6 Customer heterogeneity | `results/extended_evaluation/ext_customer_heterogeneity.csv` |
| 9 | +0.1 pt over greedy; cost-aware RF 1.5 pts better | Table 3 | as slide 5 |
| B1 | 46,673 patterns; depth 4 takes 28–36 min; certificate scope | §3.1 | `FINAL_FINDINGS.md` §4; `results/sensitivity/s11_e7_depth4.md` |
| B2 | Solver cross-check table | Table 2 | `results/final_mio_crosscheck.csv` |
| B3 | β grid: OPT 5.2% (β = 1.5), 20.1% (5), 13.3% (20); < 1% regret for β in [4, 8] | §6 Economic sensitivity, Figure 3 | `results/sensitivity/s10_e2_saving_by_beta.csv`, `s10_e4_regret_vs_tuned.csv` |
| B4 | 94 subscriptions with ≥ 1,000 test VMs deliver 85% of the saving; 11.9% of subscriptions pay more | §6 Customer heterogeneity, Figure 4 | `results/extended_evaluation/ext_customer_heterogeneity.csv` |
| B4 | Clustering: 19.7% at β = 5 vs 20.1% OPT (not significant); 13.0% vs 15.3% at β = 10 | §5.1, Table 4 | `results/hw6/hw6_cluster_results_b{5,10}.md`, `hw6_cluster_bootstrap.md` |

The backup figures (slides 12 and 13) are the report's own Figure 3 and Figure 4, rasterized at 400 dpi from
`report/figures/*.pdf`.

## Items noticed in `ML_Final_Report.pdf` (flagged, not changed)

1. **Broken table reference.** §6 "Tree complexity and interpretability" reads "(Table ??)". It should be Table 5.
   Re-run LaTeX twice, or check the `\label` and `\ref`.
2. **Significance vs H1 is β-specific.** §4 says "OPT performs significantly better than the customer rule H1". That
   holds at β = 5 (−0.0095 [−0.0163, −0.0041]) and marginally at β = 10, but not at β = 2 (−0.0021 [−0.0092, 0.0047])
   in `results/final_bootstrap.csv`. The deck makes no claim about OPT vs H1 significance.
3. **Typos.**
   * §3.1 Verification: "The validate our solver".
   * §3: "This proivdes"; "chosen by on their permutation importance"; "evaluated don the validation data".
   * §4: "o assess" (missing "T").
   * §4: "The β = 2 and β = 10 trees use the similar features".
   * §5.1 / §5.2 headings.
4. **No numeric discrepancies.** Every number used in the deck matches the saved outputs.
