# Final findings: Optimizing Azure VM CPU reservations with Optimal Policy Trees

*MIT 15.095 final project (Chris Low, Emma Li). Source of truth for the 8-page report and the December 8 presentation.
Every number below comes from a file in `results/` (named in brackets) produced by `./run_final.sh`.*

## 1. Answer to the research question

> *Can historical Azure VM workload information help us reserve less CPU capacity while controlling
> under-provisioning, and can an Optimal Policy Tree learn an interpretable allocation policy that competes with
> prediction-based and black-box methods?*

1. **Yes, historical information substantially reduces modeled allocation cost.** On the held-out last five days
   (426,846 VMs), a depth-3 Optimal Policy Tree cuts modeled cost relative to the best fixed reservation by
   **11.5% (β = 2), 20.1% (β = 5) and 15.3% (β = 10)**. Subscription-clustered 95% CIs exclude zero, e.g.
   [14.1%, 26.9%] at β = 5. The tree closes 38–49% of the gap between the best fixed rule and a perfect-hindsight
   oracle. At β = 5 it reserves 73% of VM capacity on average (vs. 100%), leaves 9.3% of VMs short, and the mean
   shortfall is 1.3% of a VM's capacity.
2. **OPT is competitive but not the best.** The black-box cost-aware Random Forest is better by a small but
   statistically clear margin: 1.6–1.9% lower cost at every β. OPT retains 89–93% of the Random Forest's savings
   using 8 leaves and 4–6 interpretable variables. Most of the gap is model flexibility, not extra information: an RF
   restricted to the trees' 8 features still costs 0.786 at β = 5 vs. OPT's 0.799
   [`final_audit_rf_same_features`].
3. **Global optimization helps in-sample, but only marginally out of sample.**
   * **In sample:** for every shared configuration (same data, depth, minimum leaf, β), exact OPT has the lowest
     training cost, as it must. Greedy is 0–0.7% higher (median 0.3%), and IAI and CART-MSE are never lower. For
     the validation-selected trees the gap to greedy is 0.4–0.5%.
   * **Out of sample:** the improvement over greedy is 0.2% at β = 2 and β = 5 (not significant) and 0.7% at β = 10
     (significant, CI [0.2%, 1.3%]).
   * **Against the feasibility study's MSE-based CART policy tree,** OPT is 0.8% (β = 5) and 2.4% (β = 10) better
     (both significant). That tree used a fixed, untuned configuration and all 26 features. Against the *tuned*
     CART-MSE on the same 8 features, OPT's advantage is significant only at β = 10. Most of the gain over CART is
     already captured by the greedy tree trained on the allocation-cost objective.
4. **The method matters less than the information.**
   * **Two transparent customer-history rules recover most of the value:** a one-parameter rule (customer's mean past
     P95 + margin) saves 19.1% at β = 5, and a zero-parameter critical-ratio rule saves 18.1%.
   * **Removing subscription history** collapses every method's savings to 0–7%.
   * **Naive "predict, then round up to a tier"** is worse than the fixed rule at β = 10. Decisions must account for
     the asymmetric cost.

A one-sentence summary for the presentation: *a three-question, eight-leaf decision tree learned by exact
optimization reserves about a quarter less CPU per VM on average than always reserving 100% and cuts modeled cost
by 20%. It comes
within 2 points of a 100-tree black box, and its advantage over greedy splitting is real in training but mostly
within noise on the test period.*

## 2. What was already present vs. new in this phase

| | already present (feasibility study, stages 1-4) | new in the final project (stages 5-7) |
|---|---|---|
| data | VM table, strictly historical features, chronological split | **strict label availability**: train/val rows must have completed before their cutoff |
| policy trees | multi-output MSE CART on cost vectors (not OPT) | **exact globally optimal policy trees** (DP), **IAI OPT**, **objective-aligned greedy tree**, **Gurobi MIO** cross-check |
| baselines | fixed, predict-then-decide, cost-aware RF, oracle | **customer-history heuristics H1/H2**; all baselines refit under the strict setup |
| evaluation | per-VM costs, simple bootstrap vs best fixed | consolidated metrics, vCPU and vCPU-hour weighting, **paired subscription-cluster bootstrap**, segments, ablation, stability, censoring and setup sensitivity |
| software | scripts 01-04 | `final_lib.py`, `final_setup.py`, scripts 05-07, `iai_opt.jl`, **23 automated tests**, `run_final.sh` |

The feasibility outputs (`REPORT.md`, `results/0*`, `figures/fig*`) are untouched. The only edit to existing code is
an optional `train_mask` argument added to `common.impute`; its default behaviour is identical.

## 3. Data, modeling population and leakage safeguards

* **Dataset.** Azure Public Dataset V2 VM table: 2,695,548 VMs over 30 days. The model population is the 2,468,232
  VMs created during the trace. The 5-minute telemetry was not needed.
* **Target.** Y = `p95maxcpu`/100 ∈ [0, 1], the fraction of the VM's allocated vCPUs. Actions
  a ∈ {0.25, 0.5, 0.75, 1}. Modeled cost: C(a; Y, β) = a + β·max(0, Y − a).
* **Strict chronological setup (primary).**

  | split | days | rows | label rule |
  |---|---|---:|---|
  | train | 0–20 | **1,703,020** (of 1,747,514) | VM deleted before day 21 |
  | validation | 21–24 | **275,143** (of 293,872) | VM deleted before day 25 |
  | test | 25–29 | **426,846** | all VMs; censored VMs and day 29 removed only in sensitivity checks |

  The 2.5% of training VMs dropped account for **57% of training vCPU-hours** (they are long-lived). This is the
  realistic cost of waiting for complete labels.
* **Features.** Every history statistic uses only VMs of the same subscription or deployment *deleted strictly
  before* the new VM's creation. Counts use VMs *created* strictly before it.
* **Never used as predictors:** `p95maxcpu`, `avgcpu`, `maxcpu`, deletion time, lifetime, `vmcategory` (assigned
  only after at least 5 days of observed utilization), and the deployment table's `deploymentsize` (counts future VMs).
  Lifetime is used only as a post-hoc evaluation weight and segment label.
* **Automated checks** (`tests/test_final.py`):
  * no forbidden column in any feature set;
  * brute-force re-computation of history features against raw timestamps;
  * label availability and disjointness of the masks;
  * units (percent vs. fraction).
* **Imputation and tuning.** All imputation values come from the strict training rows. All tuning (tree depth and
  leaf size, margins, best fixed tier, regression-tree size) uses the strict validation rows only.
* **Strict vs. feasibility setup.** Per-VM savings are unchanged (OPT 20.10% vs. 20.16% at β = 5). vCPU-hour-weighted
  savings are mostly 0.1–2.0 points lower under the strict setup, e.g. greedy at β = 10: 11.0% vs. 13.1%. One case
  goes the other way: OPT at β = 10 is 13.3% vs. 12.3% [`final_setup_comparison`].

## 4. Optimal Policy Trees: method and solvers

**Feature set.** The trees use 8 features, chosen as the top-8 by validation-set permutation importance of the
cost-aware RF at β = 5 [`final_feature_selection`]; no test data was involved.
* subscription mean past P95
* subscription median past P95
* share of past VMs with P95 ≤ 75%
* vCPU bucket
* share of past VMs with P95 ≤ 50%
* memory bucket
* GB per vCPU
* deployment mean past P95

**Candidate splits.** Up to 31 training-quantile thresholds per feature (≤ 32 bins). Every tree method uses the same
candidates.

**Tree methods compared:**

| method | objective | split search | optimality |
|---|---|---|---|
| CART-MSE (26 feat.) | squared error of the 4-dim cost vector | greedy | none (this is the feasibility study's tree) |
| CART-MSE | same, on the 8 binned features | greedy | none |
| Greedy | **modeled allocation cost** | greedy, one-step lookahead | none |
| **OPT-exact** | **modeled allocation cost** | **exhaustive dynamic programming over all trees of depth ≤ 3** | **certified global optimum over the candidate split set (gap 0)** |
| OPT-IAI | modeled allocation cost | Interpretable AI `OptimalTreePolicyMinimizer` v3.2.2 (local search; Amram, Dunn & Zhuo 2022) | no certificate |

**What "exact" means.** For a given β, depth limit D ≤ 3 and minimum leaf size m, the returned tree minimizes total
*training* cost over every tree that satisfies all of the following:
* depth ≤ D;
* axis-aligned splits x_j ≤ θ, using one of the 8 features and one of that feature's candidate thresholds;
* every leaf contains at least m training VMs;
* any of the four actions at each leaf.

Shallower trees are included. This is enumeration, so no relaxation or bound is involved. It is checked against
brute-force enumeration at depths 2 and 3 and against Gurobi at depth 2 (`tests/test_final.py`). The tree reported
for each β is optimal for its validation-selected (D, m). This is the same exhaustive-search idea as the R package
`policytree` (Sverdrup et al. 2020; Zhou, Athey & Wager 2023), made fast by sufficient statistics and pattern
aggregation. "OPT-IAI" is the Amram, Dunn & Zhuo algorithm; "OPT-exact" solves the same policy-tree optimization
problem exactly over the stated class.

**How OPT-exact works** (`final_lib.solve_opt_exact`). The cost of any leaf for any β is a linear function of four
sufficient statistics: the count and the shortfall sums at 25%, 50% and 75%. Training rows collapse to 46,673 unique
binned patterns. For each pair of features (a, b), a joint histogram over (a, b, c) with 2-D prefix sums gives the
cost of every grandchild region for every pair of thresholds at once. Enumerating all of them is therefore exact and
takes about 67 s for the full 15-configuration depth-3 grid on 1.70M rows (about 5 s for a single tree). Every leaf
can take any of the four actions. A minimum leaf size is enforced. Depth (1–3) and minimum leaf size
(0.1–5% of the training rows) are selected on validation separately for each β.

**Certification and solver diagnostics** [`final_mio_crosscheck`]. On 1k- and 5k-row subsamples (8 bins), an
independent Gurobi 13.0.3 MIO formulation was solved:

| problem | Gurobi MIO | exact DP | IAI | greedy |
|---|---|---|---|---|
| depth 2, n = 1,000 | OPTIMAL, gap 0, 86 s; objective = DP to 1e-16 | 0.003 s | = optimum | +0.07% |
| depth 2, n = 5,000 | OPTIMAL, gap 0, 212 s; objective = DP | 0.004 s | = optimum | +0.07% |
| depth 3, n = 1,000 | TIME_LIMIT 600 s, gap 13.1% | 0.07 s | +0.09% above optimum | +0.49% |
| depth 3, n = 5,000 | TIME_LIMIT 600 s, gap 7.5–7.9% | 0.07 s | = optimum | +0.50% |

At depth 3 the MIO never finds a better solution than the DP, even when warm-started with it, but it cannot close its
bound in 10 minutes. The DP certificate is enumeration, not an MIO bound.

**IAI limitations.** IAI 3.2.2 aborted twice with an internal local-search assertion ("no descent: before X after X")
on the full 1.70M-row training set, once with `cp = 0` and once with `cp = 1e-6`, after about 2.5 minutes each.
OPT-IAI is therefore trained on a uniformly random, reproducible 200,000-row sample of the strict training set
(seed 0; not stratified by time or customer). Exact OPT, greedy and CART-MSE are also reported on that same sample
for an equal-sample comparison. IAI fits take 5–22 s per tree. Two IAI settings differ from the other trees:
* **`cp = 1e-6` instead of 0.** IAI aborts with `cp = 0`. The penalty only removes splits that do not lower cost.
* **IAI's grid starts at depth 2.** The validation-selected IAI trees are all depth 3, so this does not affect the
  comparison.

## 5. Main benchmark (test period, strict setup)

### β = 5 (main scenario) [`final_main_table_b5`]

| policy | cost | % vs best fixed | % of gap to oracle closed | avg reserved | % VMs short | mean shortfall | mean unused | actions 25/50/75/100 (%) | complexity |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Always 100% = best fixed (val) | 1.000 | 0.0 | 0 | 1.000 | 0.0 | 0.000 | 0.536 | 0/0/0/100 | – |
| H2 customer critical-ratio rule | 0.819 | 18.1 | 44.0 | 0.793 | 3.5 | 0.005 | 0.334 | 16/9/16/58 | 0 parameters |
| H1 customer mean P95 + margin | 0.809 | 19.1 | 46.6 | 0.725 | 11.7 | 0.017 | 0.277 | 15/11/44/30 | 1 parameter |
| RF predict → tier (naive) | 0.858 | 14.2 | 34.7 | 0.590 | 29.9 | 0.053 | 0.180 | 19/35/36/10 | 100 trees |
| Ridge predict + margin | 0.802 | 19.8 | 48.1 | 0.730 | 10.3 | 0.014 | 0.280 | 15/13/39/34 | linear |
| RF predict + margin | 0.802 | 19.8 | 48.1 | 0.732 | 10.0 | 0.014 | 0.282 | 14/11/41/33 | 100 trees |
| CART-MSE (26 feat.), feasibility tree | 0.806 | 19.4 | 47.3 | 0.721 | 12.0 | 0.017 | 0.273 | 16/10/45/30 | depth 3, 8 leaves |
| CART-MSE (8 binned feat.) | 0.804 | 19.6 | 47.7 | 0.717 | 12.1 | 0.017 | 0.270 | 16/10/45/29 | depth 3, 8 leaves |
| Greedy policy tree | 0.800 | 20.0 | 48.6 | 0.743 | 8.1 | 0.011 | 0.290 | 16/11/32/41 | depth 3, 5 leaves |
| **OPT-exact** | **0.799** | **20.1** | **48.9** | 0.734 | 9.3 | 0.013 | 0.283 | 16/12/34/38 | **depth 3, 8 leaves** |
| OPT-IAI (200k sample) | 0.796 | 20.4 | 49.7 | 0.730 | 9.2 | 0.013 | 0.279 | 16/15/30/39 | depth 3, 8 leaves |
| Cost-aware RF | **0.784** | **21.6** | **52.5** | 0.725 | 9.1 | 0.012 | 0.273 | 17/14/32/37 | 100 trees |
| Oracle (hindsight lower bound) | 0.589 | 41.1 | 100 | 0.577 | 9.9 | 0.002 | 0.115 | 40/17/17/27 | not achievable |

Equal 200k sample: greedy 0.7955, OPT-exact 0.7990, OPT-IAI 0.7957, CART-MSE 0.8041.
Depth-4 extension (selected on validation): greedy 0.794, IAI 0.797 [`final_tree_selected`].

### Across β [`final_cost_by_beta`]

| policy | cost β=2 | cost β=5 | cost β=10 | % saving β=2 | % saving β=5 | % saving β=10 |
|---|---:|---:|---:|---:|---:|---:|
| best fixed (val) | 0.790 (50%) | 1.000 (100%) | 1.000 (100%) | 0 | 0 | 0 |
| H1 customer mean + margin | 0.701 | 0.809 | 0.857 | 11.3 | 19.1 | 14.3 |
| H2 critical-ratio rule | 0.716 | 0.819 | 0.851 | 9.4 | 18.1 | 14.9 |
| RF predict → tier (naive) | 0.697 | 0.858 | **1.125** | 11.7 | 14.2 | **−12.5** |
| RF predict + margin | 0.692 | 0.802 | 0.847 | 12.4 | 19.8 | 15.3 |
| CART-MSE (26 feat.) | 0.698 | 0.806 | 0.867 | 11.6 | 19.4 | 13.3 |
| Greedy | 0.700 | 0.800 | 0.852 | 11.4 | 20.0 | 14.8 |
| **OPT-exact** | 0.699 | 0.799 | 0.847 | 11.5 | 20.1 | 15.3 |
| OPT-IAI (200k) | 0.699 | 0.796 | 0.846 | 11.5 | 20.4 | 15.4 |
| **Cost-aware RF** | **0.687** | **0.784** | **0.833** | **12.9** | **21.6** | **16.7** |
| oracle | 0.571 | 0.589 | 0.595 | 27.7 | 41.1 | 40.5 |

## 6. Does global optimization beat greedy construction?

**Results ladder** (full training data, validation-selected configs; test cost, lower is better):

| β | CART-MSE (26 feat.) | CART-MSE (8 feat.) | Greedy, cost objective | OPT-exact |
|---|---:|---:|---:|---:|
| 2 | 0.6982 | 0.6983 | 0.6997 | 0.6985 |
| 5 | 0.8058 | 0.8041 | 0.8004 | 0.7990 |
| 10 | 0.8673 | 0.8541 | 0.8524 | 0.8468 |

Paired subscription-cluster bootstrap of OPT minus each comparator (1,000 reps; negative = OPT better)
[`final_bootstrap`]:

| comparison | β = 2 | β = 5 | β = 10 |
|---|---|---|---|
| OPT − greedy (full data) | −0.0012 [−0.0058, 0.0040] | −0.0013 [−0.0054, 0.0027] | **−0.0057 [−0.0110, −0.0014]** |
| OPT − CART-MSE (8 feat.) | +0.0002 [−0.0061, 0.0062] | −0.0051 [−0.0110, 0.0003] | **−0.0073 [−0.0160, −0.0002]** |
| OPT − CART-MSE (26 feat.) | +0.0003 [−0.0058, 0.0061] | **−0.0067 [−0.0126, −0.0017]** | **−0.0206 [−0.0354, −0.0091]** |
| OPT − greedy (equal 200k sample) | −0.0014 [−0.0063, 0.0039] | +0.0035 [−0.0025, 0.0107] | −0.0026 [−0.0071, 0.0013] |
| OPT-exact − OPT-IAI (200k) | −0.0002 [−0.0005, 0.0000] | +0.0034 [−0.0004, 0.0100] | +0.0013 [−0.0035, 0.0078] |
| **OPT − cost-aware RF** | **+0.0112 [0.0049, 0.0174]** | **+0.0149 [0.0069, 0.0238]** | **+0.0141 [0.0023, 0.0292]** |
| OPT − H1 customer heuristic | −0.0021 [−0.0092, 0.0047] | **−0.0095 [−0.0163, −0.0041]** | **−0.0100 [−0.0213, −0.0006]** |

**How to read these intervals.** They are percentile intervals from 1,000 paired resamples of the 1,726 test
subscriptions. They reflect test-sample variability with each trained policy held fixed, not variability from
retraining; the retraining variability is shown separately by the stability study. There is no
multiple-comparison correction across the 21 comparisons. Intervals whose upper limit lies within about 0.002 of
zero should be called *marginal*:
* OPT vs. greedy, vs. CART-MSE (8 features) and vs. H1, all at β = 10;
* OPT vs. CART-MSE (26 features) at β = 5.

The robust differences are:
* OPT vs. the cost-aware RF, at every β;
* OPT vs. CART-MSE (26 features) at β = 10;
* OPT vs. H1 at β = 5;
* every learned policy vs. the best fixed rule.

**Interpretation:**
* **Training objective vs. MSE.** Switching from MSE on cost vectors to the allocation-cost objective helps when the
  cost is asymmetric. At β = 2 all four trees are within 0.2% of each other. CART-MSE (8 features) → greedy improves
  test cost by 0.5% at β = 5 and 0.2% at β = 10. CART-MSE (26) → greedy improves it by 0.7% at β = 5 and 1.7% at
  β = 10, but that comparison also changes features and tuning, since the 26-feature tree was not tuned.
* **Global vs. greedy.** Exact optimization lowers *training* cost by 0.4–0.5% at every β (selected trees). *Test*
  gains are small: marginally significant only at β = 10 (0.7%), and not significant on the equal 200k sample,
  where greedy even scores slightly lower at β = 5.
* **Why the gain is small.** The cost landscape is dominated by one strong variable family (the customer's past P95
  distribution), so greedy rarely gets trapped. The greedy tree at β = 5 is a 5-leaf pruned version of the same
  rule.
* **Exact vs. IAI.** The exact DP and IAI produce nearly identical trees at β = 5; they differ in two thresholds
  (§8). IAI matched the certified optimum in 3 of 4 cross-check problems.
* **Stability.** From 200k training rows upward, the exact tree is identical across 5 random training samples and
  equal to the full-data tree (100% test-action agreement). IAI (fixed `random_seed` = 1) gives 5 different trees on
  the same 5 random 200k samples (test cost 0.7966–0.8017) [`final_stability`]. In `final_fig5`, "seeds" means the
  sampling seeds.

## 7. Robustness

**Weighting** [`final_weighted_reduction`]. % saving vs. the best fixed rule chosen under the same weighting:

| policy | vCPU-wtd β=2 | β=5 | β=10 | vCPU-hour-wtd β=2 | β=5 | β=10 |
|---|---:|---:|---:|---:|---:|---:|
| H1 customer mean + margin | 11.6 | 18.6 | 15.1 | 9.2 | 15.7 | 9.8 |
| Greedy | 12.0 | 20.2 | 16.0 | 9.0 | 17.9 | 11.0 |
| OPT-exact | 12.7 | 20.2 | 16.7 | 10.1 | 17.6 | 13.3 |
| OPT-IAI (200k) | 12.7 | 20.7 | 16.3 | 10.0 | 18.1 | 13.7 |
| Cost-aware RF | 13.9 | 21.6 | 18.1 | 11.9 | 19.3 | 14.2 |

Savings shrink by 1–5 points when long-lived VMs get more weight. The main conclusions hold under both weightings:
* The cost-aware RF is best in every column.
* The tree policies stay within about 2 points of it.
* Every method keeps large savings over the fixed rule.

Under vCPU-hour weighting OPT is ahead of greedy at β = 2 and β = 10 and slightly behind at β = 5. At β = 2 the H1
heuristic (9.2) ties greedy (9.0). No CIs were computed for weighted differences. The weighted best fixed tier
(re-chosen on validation) equals the unweighted one: 50% / 100% / 100%.

**Segments, β = 5** [`final_segments_b5`, `final_fig6`]:

| segment | share of test VMs | OPT saving | RF saving | share of OPT's total saving |
|---|---:|---:|---:|---:|
| lifetime ≤ 5 min | 18.9% | 29.5% | 32.2% | 27.8% |
| lifetime 5 min – 1 h | 49.8% | 23.5% | 24.5% | 58.3% |
| lifetime 1 h – 1 day | 26.5% | 6.9% | 8.7% | 9.1% |
| lifetime > 1 day | 4.7% | 20.5% | 21.2% | 4.8% |
| 2 vCPU bucket | 57.7% | 11.0% | 12.6% | 31.7% |
| 4 vCPU bucket | 29.7% | 40.3% | 41.8% | 59.6% |
| >24 vCPU bucket | 1.1% | 1.7% | 5.0% | 0.1% |
| ≥ 1,000 completed subscription VMs | 91.0% | 20.4% | 21.9% | 92.6% |
| 1–49 completed subscription VMs | 1.1% | 22.6% | 22.3% | 1.3% |
| no completed subscription history | 0.14% (577 VMs) | 10.5% | 18.0% | 0.1% |
| right-censored at trace end | 4.7% | 17.1% | 18.5% | 4.0% |

Savings are positive in every segment but concentrated in short-lived and 4-vCPU VMs. VMs living one hour to one day
benefit least (about 7%). Cold start is too small to evaluate reliably; OPT handles it worse than greedy or RF there
(fewer than 600 VMs).

**Feature ablation** (% saving vs best fixed) [`final_ablation`]:

| feature set | OPT β=2 / 5 / 10 | greedy β=2 / 5 / 10 | cost-aware RF β=2 / 5 / 10 |
|---|---|---|---|
| A: size + creation time | 4.5 / 6.0 / 0.9 | 3.4 / 5.4 / 0.0 | 5.1 / 7.2 / 1.2 |
| B: A + subscription history | 11.5 / 20.3 / 15.1 | 11.3 / 20.2 / 15.0 | 12.6 / 20.9 / 16.4 |
| C: B + deployment history (main) | 11.5 / 20.1 / 15.3 | 11.4 / 20.0 / 14.8 | 12.9 / 21.6 / 16.7 |

Subscription history is essential for *decisions*, not just predictions. Deployment history adds nothing for the
trees and about 0.4–0.7 points for the RF.

**Censoring and final day** [`final_censoring_sensitivity`]. Excluding right-censored test VMs and the partial day 29
raises OPT's β = 5 saving from 20.1% to 20.8% (RF: from 21.6% to 22.4%). Conclusions are unchanged.

**Additional feature engineering was not pursued.** The gap that matters (OPT vs. RF) is about 1.5 points. Deeper
trees close at most about 0.5 points of it, and the ablation shows the binding constraint is the information in
customer history, not missing transformations of it.

## 8. What the learned policy does (OPT-exact, β = 5, `figures/final_fig4_opt_tree.png`)

Shares and mean realized P95 below are on the test period.

```
How often did this customer's past VMs exceed 75% CPU?  (share of past VMs with P95 <= 75% <= 0.858)
├─ often (>14% of past VMs ran hot; 64% of test VMs)
│   ├─ customer median past P95 <= 62%
│   │   ├─ vCPU <= 2 ........................ reserve 75%   (19.9% of VMs, mean P95 44%)
│   │   └─ vCPU > 2 ......................... reserve 100%  ( 5.0%, mean P95 52%)
│   └─ customer median past P95 > 62%
│       ├─ <= 61.9% of past VMs under 75% ... reserve 100%  (32.9%, mean P95 73%)
│       └─ otherwise ........................ reserve 75%   ( 6.0%, mean P95 59%)
└─ rarely (36% of test VMs)
    ├─ <= 84.9% of past VMs under 50%
    │   ├─ deployment mean past P95 <= 38.6% . reserve 50%  ( 7.5%, mean P95 29%)
    │   └─ otherwise ........................ reserve 75%   ( 8.3%, mean P95 46%)
    └─ > 84.9% of past VMs under 50%
        ├─ customer mean past P95 <= 21.1% .. reserve 25%   (16.3%, mean P95 2%)
        └─ otherwise ........................ reserve 50%   ( 4.3%, mean P95 31%)
```

**How the rule behaves:**
* The root question is about the *upper tail* of the customer's history, not its average.
* Near-idle customers (16% of VMs, realized P95 ≈ 2%) get 25%.
* Customers with frequent hot VMs keep 100% unless their VMs are small.
* VM size matters only inside the hot-customer branch.
* The β = 2 and β = 10 trees ([`results/final_models/OPT-exact_full_b{2,10}.txt`]) use the same variable families
  (the customer's upper-tail shares and P95 level, vCPU size, deployment mean P95). They shift toward smaller
  (β = 2) or larger (β = 10) reservations.

## 9. Limitations: what we cannot claim

* **Modeled cost is not real money.** Costs are modeled allocation costs, not Azure billing, server consolidation, or
  performance. We observe each VM's demand under its original allocation; capping CPU could change the workload, so
  realized P95 is treated as exogenous demand. The oracle is a hindsight lower bound under this model, not an
  achievable policy.
* **β is a sensitivity parameter, not an estimate.** For β ≤ 1 the problem is trivial (always 25%).
* **Certified optimality is narrow.** It holds for the *training* objective, over depth ≤ 3 trees on 8 features with
  ≤ 32 quantile thresholds per feature. It is not out-of-sample optimality and not optimality over all thresholds or
  features.
* **OPT-IAI used a 200k sample** because IAI 3.2.2 failed on the full training set.
* **P95 for short-lived VMs comes from few readings.** 68% of VMs live under an hour. Results weighted by vCPU-hours
  are 1–5 points lower.
* **The strict label rule removes long-lived training VMs.** A production system would have partial telemetry for
  still-running VMs, which this dataset does not provide.
* **One region, 30 days in 2019, a five-day test window.** Savings are concentrated in large customers (91% of test
  VMs come from subscriptions with ≥ 1,000 completed VMs). The cluster bootstrap accounts for this, which is why the
  absolute-saving CIs are wide (about ±6 points at β = 5) while paired differences are tight.
* **Tuning and feature selection are done once.** The 8 tree features were chosen by validation importance at
  β = 5 and reused for β = 2 and 10. The strict validation set also excludes VMs still running at day 25, so tuning
  under-represents long-lived VMs.
* **Validation selection is noisy.** It does not always pick the test-best configuration, e.g. a smaller minimum
  leaf would have given OPT 0.7957 at β = 5. We report the validation-selected results.

## 10. Recommended figures and tables

**8-page report:**
1. `figures/final_fig4_opt_tree.png`: the learned OPT (methods/results centerpiece).
2. `figures/final_fig1_benchmark_b5.png` with the β = 5 table (§5), trimmed to about 10 rows.
3. `figures/final_fig2_beta_ci.png`: savings across β with 95% cluster CIs.
4. Table: the OPT vs greedy ladder and bootstrap (§6), the key course-method result.
5. Table: MIO vs DP vs IAI cross-check (§4), for solver diagnostics.
6. `figures/final_fig3_frontier.png`: capacity vs under-provisioning (all learned methods dominate the fixed rules).
7. Appendix: `figures/final_fig5_opt_scaling.png` (stability and runtime), `figures/final_fig6_segments.png`, the
   ablation table, the weighting table, and the strict vs feasibility setup table.

**December 8 presentation (about 6 result slides):**
1. Problem and cost function.
2. The OPT tree (fig 4).
3. Benchmark bar chart (fig 1).
4. Frontier (fig 3).
5. OPT vs greedy vs RF across β with CIs (fig 2 + the one-line ladder).
6. Limitations and takeaways.

## 11. Reproducibility

`./run_final.sh` runs the tests and stages 5, 5b, 6 and 7 in about 40 minutes. Stage 5b's MIO time limits
account for about 30 minutes of that.

* **Environment** [`results/final_environment.md`]: Apple M4 (10 cores, 16 GB), Python 3.12.8, numpy 2.4.4,
  pandas 3.0.3, scikit-learn 1.9.1, Gurobi 13.0.3 (academic), IAI system image 3.2.2 on Julia 1.11.9 (academic).
* **Seeds:** numpy/sklearn 0, IAI `random_seed` 1, Gurobi `Seed` 0. IAI fits are cached by input hash.
* **Large intermediates** live in `data/processed/final/` (git-ignored). Selected trees are saved as JSON and text in
  `results/final_models/`.
