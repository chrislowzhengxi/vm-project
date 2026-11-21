# Homework 6 refinements: findings

*Extensions to the finalized pipeline (`FINAL_FINDINGS.md`), implemented in `src/08_hw6_weighted_prescriptions.py`,
`src/09_hw6_clustering.py` and `src/hw6_lib.py`, reproduced by `./run_hw6.sh` (about 5 min).*

**Shared protocol.** Strict chronological split: 1,703,020 train / 275,143 validation / 426,846 test VMs. Leakage-safe
creation-time features. All tuning on validation; all numbers below on the full test period.

**Original results untouched.** All 237 finalized stage 1–7 files (results, figures, models, processed features) are
byte-identical before and after this work (SHA-256 check). New outputs live in `results/hw6/`, `figures/hw6/` and
`data/processed/hw6/`.

## 1. Bottom line

| question | answer |
|---|---|
| Does a forest-weighted (distribution-aware) prescription beat point prediction from the *same* forest? | **Yes, at every β.** It is 1.0% / 2.0% / 1.4% cheaper than point prediction + validation-tuned margin at β = 2 / 5 / 10. The β = 2 and 5 CIs clearly exclude 0; β = 10 is marginal. |
| Does it beat the existing cost-aware RF? | **No, it ties.** Differences are within ±0.35%, not significant at β = 5 and 10, and marginal at β = 2. |
| Does it beat exact OPT? | **Yes, by 1.4–1.9%** (CIs exclude 0), but it is a 100-tree black box, like the cost-aware RF. |
| Do k-means clusters give useful reservation groups? | **Partly.** 8 clusters with one tier each save 19.7% at β = 5, statistically tied with OPT (20.1%). At β = 10 they are significantly worse than OPT (13.0% vs 15.3%). |
| Can an OCT interpret the clusters without losing performance? | **Yes.** A depth-3 OCT reproduces 86% of cluster labels and a depth-4 OCT 89%. Replacing k-means by the OCT changes cost by at most 0.3%, never significantly. |
| Include in the 8-page report? | **Refinement 2 in the main text, Refinement 1 as a short paragraph or appendix** (§6). |

## 2. Refinement 2: distribution-aware prescriptions (Chapter 13)

**Method.** One P95 regression forest, three decision rules on the *same* fitted forest:
* **A, point prediction:** the predicted P95, rounded up to the next tier.
* **B, point prediction + safety margin:** the same prediction plus a margin δ tuned on validation for each β.
* **C, forest-weighted prescription:** â(x) = argmin_a Σ_i w_i(x)·[a + β·max(0, Y_i − a)], where
  w_i(x) = (1/T) Σ_t 1{X_i ∈ L_t(x)}/|L_t(x)|.

**Implementation:**
* **Weights are over all strict training rows,** not each tree's bootstrap sample. |L_t(x)| counts every training VM
  routed to that leaf, so every leaf has at least 50 rows and the weights are well defined.
* **No dense weight matrix.** Because the cost is linear in the shortfall, each tree stores per-leaf
  [count, Σ max(0, Y − 0.25), Σ max(0, Y − 0.5), Σ max(0, Y − 0.75)]. A query averages its 100 leaf rows, which is
  exact for every β at once.
* **Verified two ways.** Tests confirm the weights sum to 1 and that explicit weighted costs equal the leaf
  aggregation. On real data, the two illustrative VMs' explicit 1.7M-row weight vectors reproduce the aggregated
  costs exactly.
* **Forest settings** are those of the main pipeline (100 trees, `max_features` 0.5, 300k-row bootstrap samples,
  seed 0). Only `min_samples_leaf` ∈ {50, 500, 2000} is tuned, on validation. Rule C selects 50 at every β, which is
  exactly the main pipeline's forest: rules A and B on that forest reproduce the stage-6 "RF predict" policies bit for
  bit (asserted in the code).
* **Cost:** fitting takes about 35 s, and leaf tables plus validation and test prescriptions about 12 s.

### Test results [`results/hw6/hw6_forest_results_b{2,5,10}.md`]

| policy | cost β=2 | cost β=5 | cost β=10 | % saving β=2 | % saving β=5 | % saving β=10 |
|---|---:|---:|---:|---:|---:|---:|
| best fixed (existing) | 0.790 | 1.000 | 1.000 | 0 | 0 | 0 |
| A: point prediction → tier | 0.697 | 0.856 | **1.115** | 11.7 | 14.4 | **−11.5** |
| B: point prediction + margin | 0.692 | 0.801 | 0.847 | 12.4 | 19.9 | 15.3 |
| **C: forest-weighted prescription** | **0.685** | **0.786** | **0.835** | **13.2** | **21.4** | **16.5** |
| cost-aware RF (existing) | 0.687 | 0.784 | 0.833 | 12.9 | 21.6 | 16.7 |
| OPT-exact (existing) | 0.699 | 0.799 | 0.847 | 11.5 | 20.1 | 15.3 |
| oracle (existing) | 0.571 | 0.589 | 0.595 | 27.7 | 41.1 | 40.5 |

A and B are shown at their validation-selected forests. On the main forest (`min_samples_leaf` = 50) they are
0.858 / 0.802 at β = 5; all three rows are in the tables.

**Rule C's operating point at β = 5:**
* reserves 0.725 of capacity on average;
* 9.2% of VMs under-provisioned, mean shortfall 0.012;
* actions 17 / 12 / 34 / 36% at 25 / 50 / 75 / 100%;
* closes 52% of the gap between the best fixed rule and the oracle.

**vCPU-hour-weighted, β = 5** [`hw6_forest_vcpu_hour_b5`] (% saving vs the weighted best fixed):

| C | cost-aware RF | greedy tree | OPT | B | A |
|---:|---:|---:|---:|---:|---:|
| 19.7 | 19.3 | 17.9 | 17.6 | 17.1 | 13.2 |

### Paired subscription-cluster bootstrap [`hw6_forest_bootstrap`] (cost difference, 95% CI; negative = C cheaper)

| comparison | β = 2 | β = 5 | β = 10 |
|---|---|---|---|
| C − A (same forest family) | −0.0123 [−0.0173, −0.0085] | −0.0706 [−0.1006, −0.0384] | −0.2799 [−0.3554, −0.2003] |
| **C − B** | **−0.0070 [−0.0111, −0.0042]** | **−0.0157 [−0.0219, −0.0102]** | −0.0120 [−0.0246, −0.0015] *(marginal)* |
| C − B, identical forest (msl = 50) | −0.0070 [−0.0111, −0.0042] | −0.0167 [−0.0230, −0.0117] | −0.0120 [−0.0246, −0.0015] |
| C − cost-aware RF | −0.0024 [−0.0048, −0.0004] *(marginal)* | +0.0016 [−0.0030, 0.0069] | +0.0023 [−0.0058, 0.0101] |
| C − OPT-exact | −0.0136 [−0.0188, −0.0082] | −0.0134 [−0.0204, −0.0073] | −0.0118 [−0.0230, −0.0031] |

The intervals have no multiple-comparison adjustment and hold the trained forest fixed. Treat results whose upper
limit is within about 0.002 of zero as marginal.

### Why it works

The margin in rule B is one global shift; rule C adapts to the *shape* of each VM's conditional demand.
* **Rules C and A disagree on 21 / 53 / 70% of test VMs** at β = 2 / 5 / 10, and C and B on 12 / 15 / 23%
  [`hw6_forest_decision_disagreement`].
* **The weighted shortfall estimates are well calibrated on average.** Expected vs. realized test shortfall:

  | tier | expected (weighted) | realized (test) |
  |---|---:|---:|
  | 25% | 0.284 | 0.285 |
  | 50% | 0.145 | 0.145 |
  | 75% | 0.049 | 0.049 |

  [`hw6_forest_calibration`]

**Illustration** (`figures/hw6/hw6_fig_forest_example.png`; the two test VMs with the largest expected-cost gap
between A and C, so illustrative rather than typical):
* **A VM predicted at 75.0%.** Its weighted neighbours are bimodal, with 72% of the weight above 75% P95. A reserves
  75%; C reserves 100% (expected cost 1.00 vs 1.49). Realized P95 was 97%.
* **A VM predicted at 25.5%.** A rounds this up to 50%. Only 10% of the neighbour weight is above 50%, so C reserves
  25% (expected cost 0.58 vs 0.61). Realized P95 was 14%.

### Relation to the cost-aware RF

The two forests estimate the same quantity, E[max(0, Y − a) | x]. They differ in two ways:
* **Split criterion:** the cost-aware RF splits on the shortfall vector; the P95 forest splits on P95.
* **Leaf averages:** the cost-aware RF averages its bootstrap rows; rule C averages all training rows.

They end up statistically tied. The interesting result is the controlled one: *with the forest held fixed*, replacing
the point estimate by the leaf-weighted distribution is what closes the gap between predict-then-decide (B) and
direct cost learning.

## 3. Refinement 1: interpretable clustering (Chapter 20)

### Steps A and B: k-means → cluster-specific reservations

* **Features:** vCPU and memory bucket (log2), subscription mean and median past P95, subscription share of past VMs
  above 75% and above 50%, number of completed subscription VMs (log1p), deployment mean past P95, and optionally
  creation hour (sin/cos).
* **Transforms:** log transforms cut the skewness of vCPU from 4.1 to 1.7, memory from 1.2 to 0.0, and VM count from
  2.7 to −0.5 [`hw6_cluster_feature_skew`].
* **Fitting:** the scaler and k-means (5 restarts, seed 0) are fit on strict training rows only, taking 1–4 s per fit.
* **Selection** [`hw6_cluster_kmeans_grid`]: K ∈ {3, 5, 8, 10}, with or without hour. Validation cost at β = 5 selects
  **K = 8 without hour**. That variant also has the best silhouette (0.39 vs 0.22 with hour), since hour adds noise,
  not cost signal.
* **Actions:** each cluster takes the action with the lowest *training* cost for each β; the clusters themselves do
  not depend on β.

**The 8 clusters** (`figures/hw6/hw6_fig_cluster_profiles.png`, [`hw6_cluster_profiles`]). They order cleanly by the
customer's historical P95 level and upper-tail frequency; size separates otherwise similar groups. At β = 5:

| cluster | share of VMs | profile | realized mean P95 | reservation |
|---|---:|---|---:|---|
| 3 | 15% | near-idle customers, 4 vCPU / 32 GB | 4% | 25% |
| 1 | 15% | low-utilization customers, 2 vCPU | 28% | 50% |
| 2 and 6 | 8% and 22% | mid-level customers | 44–49% | 75% |
| 4, 0, 5, 7 | 7%, 16%, 10%, 7% | large VMs or hot customers | 63–81% | 100% |

### Step C: OCT interpretation

The OCTs are IAI `OptimalTreeClassifier` (local search, no certificate), fit on a reproducible 200k-row training
sample with minimum leaf size 1,000. A greedy CART classifier of the same depth on the same sample is the reference
[`hw6_cluster_oct_grid`].

| tree | leaves | test agreement with k-means | test cost β=2 | β=5 | β=10 | fit time |
|---|---:|---:|---:|---:|---:|---:|
| OCT depth 2 | 4 | 61.6% | 0.707 | 0.815 | 0.891 | 6 s |
| **OCT depth 3** (compact; `hw6_fig_cluster_oct_depth3.png`) | 8 | **86.1%** | 0.706 | 0.808 | 0.872 | 17 s |
| **OCT depth 4** (validation-selected; `hw6_fig_cluster_oct.png`) | 15 | 88.9% | 0.701 | 0.805 | 0.869 | 25 s |
| CART classifier depth 3 | 8 | 74.5% | 0.707 | 0.810 | 0.890 | <1 s |
| CART classifier depth 4 | 14 | 90.8% | 0.705 | 0.803 | 0.860 | <1 s |

**Interpretation:**
* **At depth 3, optimal splitting matters for interpretation.** The OCT reproduces 86% of the cluster labels vs 74%
  for greedy CART.
* **At depth 4 CART is slightly ahead out of sample** (91% vs 89%), even though the OCT fits the training sample better
  (95.5% vs 91.8%).
* **Agreement barely moves cost.** All depth-3 and depth-4 trees cost within about 0.006 of k-means at β = 5,
  because the misclassified VMs mostly land in clusters with the same or an adjacent reservation.

### Test results and significance

| policy | cost β=2 | cost β=5 | cost β=10 | % saving β=2 | % saving β=5 | % saving β=10 |
|---|---:|---:|---:|---:|---:|---:|
| k-means (K = 8) → action | 0.703 | 0.803 | 0.870 | 11.0 | 19.7 | 13.0 |
| OCT depth 4 → cluster → action | 0.701 | 0.805 | 0.869 | 11.3 | 19.5 | 13.1 |
| greedy policy tree (existing) | 0.700 | 0.800 | 0.852 | 11.4 | 20.0 | 14.8 |
| OPT-exact (existing) | 0.699 | 0.799 | 0.847 | 11.5 | 20.1 | 15.3 |
| cost-aware RF (existing) | 0.687 | 0.784 | 0.833 | 12.9 | 21.6 | 16.7 |

Bootstrap differences (95% CI) [`hw6_cluster_bootstrap`]:

| comparison | β = 2 | β = 5 | β = 10 |
|---|---|---|---|
| k-means − OPT | +0.0045 [−0.0033, 0.0129] | +0.0044 [−0.0047, 0.0140] | **+0.0228 [0.0103, 0.0365]** |
| OCT − OPT | +0.0020 [−0.0048, 0.0096] | +0.0059 [−0.0007, 0.0127] | **+0.0225 [0.0071, 0.0392]** |
| OCT − k-means | −0.0025 [−0.0065, 0.0013] | +0.0015 [−0.0067, 0.0105] | −0.0002 [−0.0130, 0.0134] |
| OCT − cost-aware RF | **+0.0132 [0.0085, 0.0193]** | **+0.0209 [0.0126, 0.0304]** | **+0.0367 [0.0216, 0.0529]** |

Every clustering policy vs best fixed is significant (CIs far from 0).

**vCPU-hour weighting, β = 5:** k-means 16.7% and OCT 17.1%, vs OPT 17.6% and cost-aware RF 19.3%.

**Why clustering trails OPT at high β.** k-means groups VMs by feature similarity, not by cost. At β = 10 the costly
mistakes are concentrated in customers with a heavy upper tail, which OPT isolates with splits on the share of past
VMs above 75%. The k-means boundaries do not line up with those thresholds, so the cluster policy under-provisions
more (4.5% vs 3.4% of VMs).

## 4. Comparison with the existing methods

| β = 5, test cost | value | interpretable? | trained on the allocation cost? |
|---|---:|---|---|
| cost-aware RF (existing) | 0.784 | no (100 trees) | yes |
| **forest-weighted prescription (new)** | 0.786 | no (100 trees, about 4,200 leaves each) | yes (through leaf weights) |
| OPT-exact (existing) | 0.799 | yes (8 leaves) | yes |
| greedy policy tree (existing) | 0.800 | yes (5 leaves) | yes |
| k-means K = 8 → action (new) | 0.803 | moderately (8 centroids in 8 standardized features) | no (clusters), yes (actions) |
| OCT depth 4 → cluster → action (new) | 0.805 | yes (15 leaves, 8 clusters) | no |
| point prediction + margin (existing) | 0.801–0.802 | no | one margin parameter |

Neither refinement beats the existing cost-aware RF, so the main conclusion stands: about 21–22% saving at β = 5 is
the ceiling for these features. The forest-weighted rule *matches* that ceiling, and the cluster pipeline roughly
matches OPT except at β = 10.

## 5. Is the extra complexity justified?

* **Refinement 2, yes as an explanation; no as a new best policy.** It costs nothing extra (the same forest as the
  existing RF benchmark plus about 12 s of leaf statistics). It cleanly shows that the gap between predict-then-decide
  and direct cost learning comes from discarding the conditional distribution. As an operational policy it is
  equivalent to the existing cost-aware RF.
* **Refinement 1, not as a replacement for OPT.** It needs a scaler, k-means with 8 centroids and a 15-leaf OCT. OPT
  reaches the same cost at β ≤ 5, and a lower one at β = 10, with a single 8-leaf tree trained directly on cost. Its
  value is descriptive: the natural workload groups in this data already align closely with the reservation tiers,
  which is why unsupervised segmentation captures about 98% of OPT's β = 5 savings.

## 6. Recommendation for the 8-page report

1. **Include Refinement 2 in the main results**, as one paragraph plus:
   * **Rows:** add "forest-weighted prescription" to the benchmark table (or the β table).
   * **Figure:** `figures/hw6/hw6_fig_forest_rules.png`, a three-rule comparison on the same forest. Alternatively the
     example figure `hw6_fig_forest_example.png`, which is effective on a slide.
   * **Key sentence:** *"Holding the forest fixed, replacing the point prediction by its leaf-weighted demand
     distribution lowers cost by 2.0% at β = 5 (95% CI [1.3%, 2.7%]) and matches the cost-aware Random Forest."*
2. **Mention Refinement 1 briefly** in the discussion or appendix:
   * **Finding:** unsupervised workload groups recover OPT-level savings at β ≤ 5 but not at β = 10.
   * **OCT:** it interprets the groups with 86–89% agreement at no measurable cost.
   * **Optional figure:** `hw6_fig_cluster_oct_depth3.png` in the appendix.
3. **Presentation:** the forest example figure is the most intuitive single slide from this work.

## 7. Limitations and computational notes

* **Forest-weight convention.** Weights are over all training rows, not each tree's bootstrap sample; the in-bag
  variant was not run. Only `min_samples_leaf` was tuned (3 values); the other forest settings are fixed at the main
  pipeline's values.
* **OCT optimality.** OCTs come from IAI local search on a 200k-row sample, so there is no optimality certificate. An
  exact MIO OCT was not attempted: the stage-5b cross-check showed Gurobi cannot close depth-3 tree gaps in 10 minutes
  even at 1–5k rows, so 200k is out of reach.
* **Clustering selection.** The K grid was small ({3, 5, 8, 10}), selection used β = 5 only, and k-means used 5
  restarts. Clusters are fixed across β and only the action map changes.
* **Bootstrap.** CIs are unadjusted and condition on the trained models. The weighted results have no CIs.
* **Illustrative examples** were chosen for the largest disagreement, so they are not representative.
* **Modeled cost.** All main-project caveats (modeled rather than real cost, exogenous demand, short-lived VMs) apply
  unchanged.
* **Runtime** on an Apple M4 (10 cores): stage 8 about 2 min (3 forests), stage 9 about 1.5 min (8 k-means fits,
  3 OCT fits). Intermediate data: 21 MB in `data/processed/hw6/`.

## 8. Files

* **Code:** `src/hw6_lib.py`, `src/08_hw6_weighted_prescriptions.py`, `src/09_hw6_clustering.py`, `src/iai_oct.jl`,
  `tests/test_hw6.py` (10 tests), `run_hw6.sh`.
* **Results:** `results/hw6/` (forest grid, selection, per-β tables, bootstrap, calibration, disagreement, example,
  vCPU-hour; k-means grid, profiles, OCT grid, per-β tables, bootstrap, vCPU-hour, feature skewness, `hw6_oct_tree.json`).
* **Figures** (`figures/hw6/`):
  * `hw6_fig_forest_rules.png`
  * `hw6_fig_forest_example.png`
  * `hw6_fig_cluster_profiles.png`
  * `hw6_fig_cluster_oct.png` (depth 4, validation-selected)
  * `hw6_fig_cluster_oct_depth3.png` (compact alternative)
