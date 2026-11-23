# Extended evaluation beyond modeled cost: findings

*MIT 15.095 final project (Chris Low, Emma Li). Stage 12, produced by `./run_extended_evaluation.sh`. Every
number comes from a file in `results/extended_evaluation/` (named in brackets).*

**Scope and ground rules**

* **Unchanged objective:** C(r, Y; β) = r + β·max(0, Y − r). Here Y is the realized P95 CPU and r the reserved tier,
  both as fractions of the VM.
* **No retraining, no selection.** No policy was retrained and nothing was selected. All metrics come from saved
  test actions, saved forest predictions and saved trees.
  * The 4-leaf, leaf-penalized and certified depth-4 OPT trees were applied to the saved binned features.
  * Ridge and the stage-6 CART regressor were refit only to recover their test predictions (8.6 s).
* **84 consistency checks**, all passing. Every overlapping number reproduces the finalized tables exactly:
  * stage-6 cost, reservation, under-provisioning and shortfall;
  * HW6 forest costs;
  * the E5/E6/E7 tree costs;
  * stage-6 R² and tier actions;
  * every previously published cost bootstrap CI.
* **Test data are retrospective only.** Wherever a choice could be informed by a result, the validation version is
  reported (§6.4).

**Units used throughout**

* **pp** = percentage points of VM CPU capacity.
* **Exceedance** = P(Y > r).
* **Severe shortfall** = P(Y − r > 20 pp).
* **E[sf]** = E[(Y − r)+].

---

## 1. Executive summary

At β = 5 the learned policies cut mean reservation from 100% to 72–75% of VM capacity. In exchange:

* **7.7–10% of test VMs exceed their reservation.**
* **The overshoot is 13–14.5 pp on average** when they do.
* **2.0–2.8% of VMs overshoot by more than 20 pp.**

The modeled savings come from **targeting, not from reserving less across the board**:

* **Untargeted counterfactual.** The same tier mix assigned at random would cost about 13% *more* than always-100%.
* **Equal-shortfall counterfactual.** A randomized fixed-tier rule with OPT's expected shortfall must reserve 93% of
  capacity on average. OPT reserves 73%.

**Do the expanded metrics strengthen the main conclusions?** Yes.

* **The cost ranking survives on risk.** It carries over to reservation and shortfall risk.
* **The RF-family policies are not cheaper by taking more risk.** At β = 5 the cost-aware RF and the forest-weighted
  prescription have:
  * lower mean reservation than depth-3 OPT;
  * lower expected shortfall;
  * lower exceedance;
  * lower severe-shortfall frequency.

  These are point estimates; the risk differences are not individually significant.
* **At matched risk, the trees match the margin rule.** At equal expected shortfall, depth-3 OPT, greedy and RF +
  margin need the same reservation (within 0.35 pp). The RF-family needs 1.0–1.8 pp less [`ext_matched_risk`].

**Does the cost objective hide trade-offs?** Four:

* **(a) Frequency is not what β controls.** β penalizes shortfall *magnitude*, not how often it occurs. Even the
  hindsight oracle exceeds on 9.9% of VMs at β = 5. The difference lies in severity: the oracle's exceedances average
  2.5 pp, the policies' 13–14 pp.
* **(b) At β = 2 lower cost buys more exceedance.** The cost-aware RF is cheaper than OPT but exceeds on 2.3 points
  more VMs (significant, unadjusted).
* **(c) Exact depth-4 at β = 10 saves by reserving less.** It reserves 1.5 pp less than depth 3, and exceedance rises
  from 3.4% to 4.6% (significant).
* **(d) The 4-leaf tree is more conservative at β = 5.** It reserves 1.2 pp more and has 1.5 points lower exceedance
  than the 8-leaf tree (both significant) at the same cost.

**Does any lower-cost policy have materially worse shortfall risk?**

* **Not under the VM-weighted totals the objective optimizes.**
* **Yes for depth-3 OPT at the customer level:**
  * **It spreads risk less evenly across subscriptions** than greedy, the 4-leaf tree or the cost-aware RF.
  * **Treating every subscription equally** (1,036 subscriptions with ≥ 5 test VMs), OPT has the highest exceedance of
    the trees: 13.1% vs 11.2–11.4% for greedy and 4-leaf; the RF has 10.3%.
  * **The differences are significant** (unadjusted) and appear on validation too.
  * **About 1 in 5 subscriptions** have more than 25% of their VMs exceeded under OPT, vs 1 in 8 under the cost-aware RF.

**Should the policies we emphasize change?** Only modestly:

* **Keep depth-3 OPT as the interpretable headline.**
* **Present the 4-leaf exact tree as an equally good, simpler alternative.** It has the same cost and lower exceedance,
  and is no worse at the customer level.
* **Report risk columns next to cost.**
* **Do not describe OPT as the "safer" policy.** The forest-based rules are at least as good on every risk metric at
  β = 5 and β = 10.

**Caveats we had overlooked:**

* **The aggregate saving is carried by large customers.**
  * Subscriptions with ≥ 1,000 test VMs (94 of 1,726) hold 78% of VMs and deliver 85% of OPT's saving.
  * The 10 largest subscriptions save only 10%.
* **Some subscriptions are worse off.** 12% of subscriptions (≥ 5 VMs) have higher modeled cost under OPT than
  always-100%; 5.6% lose more than 10%.
* **Exceedances concentrate in hot VMs.** 80% of OPT's exceedances are VMs whose realized P95 is above 75%.
* **The > 24 vCPU segment barely benefits:** 1.7% saving under OPT.
* **Over-reservation remains common.** About 19% of VMs are reserved two or more tiers above what they needed.
* **Every policy has a few extreme misses.** Each has VMs reserved at 25% that ran near 100% (maximum shortfall
  72–74 pp).

---

## 2. Main comparison table

### β = 5 (main scenario) [`ext_main_table_b5`]

Columns:

* **Saving** is vs best fixed (validation-selected: always 100%).
* **Captured** = share of the best-fixed → oracle gap closed.
* **Shortfall | exceeded** = E[Y − r | Y > r].

| policy (leaves) | cost | saving % | E[r] % | P(Y>r) % | E[sf] pp | shortfall \| exceeded pp | P(Y−r>20pp) % | captured % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Best fixed (always 100%) | 1.000 | 0.0 | 100.0 | 0.00 | 0.00 | – | 0.00 | 0 |
| H1 customer mean + margin | 0.809 | 19.1 | 72.5 | 11.65 | 1.67 | 14.4 | 3.50 | 46.6 |
| RF point → tier | 0.858 | 14.2 | 59.0 | 29.93 | 5.35 | 17.9 | 13.12 | 34.7 |
| RF point + margin | 0.802 | 19.8 | 73.2 | 9.96 | 1.41 | 14.1 | 2.77 | 48.1 |
| Greedy depth 3 (5) | 0.800 | 20.0 | 74.3 | 8.12 | 1.14 | 14.1 | 2.25 | 48.6 |
| **OPT depth 3 (8)** | **0.799** | **20.1** | **73.4** | **9.25** | **1.30** | **14.1** | **2.56** | **48.9** |
| OPT 4-leaf, depth 2 (4) | 0.802 | 19.8 | 74.6 | 7.72 | 1.12 | 14.5 | 2.21 | 48.2 |
| OPT leaf-penalized (5) | 0.796 | 20.4 | 73.5 | 8.58 | 1.22 | 14.2 | 2.44 | 49.6 |
| OPT depth 4, certified (16) | 0.795 | 20.5 | 73.8 | 8.43 | 1.15 | 13.6 | 2.03 | 49.9 |
| Cost-aware RF | 0.784 | 21.6 | 72.5 | 9.12 | 1.18 | 13.0 | 2.19 | 52.5 |
| Forest-weighted prescription | 0.786 | 21.4 | 72.5 | 9.17 | 1.22 | 13.3 | 2.19 | 52.2 |
| Oracle (hindsight, infeasible) | 0.589 | 41.1 | 57.7 | 9.94 | 0.25 | 2.5 | 0.00 | 100 |

### β = 2 and β = 10 [`ext_main_table_b2`, `ext_main_table_b10`]

Column groups: saving % / E[r] % / P(Y>r) % / E[sf] pp / P(Y−r>20pp) %.

| policy | β = 2 | β = 10 |
|:--|:--|:--|
| Best fixed | always 50%: 0.0 / 50.0 / 47.9 / 14.48 / 33.2 | always 100%: 0.0 / 100 / 0 / 0 / 0 |
| RF point → tier | 11.7 / 59.0 / 29.9 / 5.35 / 13.1 | **−12.5** / 59.0 / 29.9 / 5.35 / 13.1 |
| RF point + margin | 12.4 / 56.2 / 34.1 / 6.51 / 16.2 | 15.3 / 76.8 / 5.84 / 0.79 / 1.48 |
| Greedy depth 3 | 11.4 / 56.0 / 33.6 / 7.01 / 16.3 | 14.8 / 81.0 / 3.17 / 0.43 / 0.77 |
| OPT depth 3 | 11.5 / 56.5 / 32.7 / 6.68 / 15.2 | 15.3 / 79.7 / 3.42 / 0.49 / 0.92 |
| OPT 4-leaf | 11.4 / 56.0 / 33.0 / 6.96 / 15.7 | 15.4 / 79.5 / 3.84 / 0.51 / 1.04 |
| OPT depth 4 (certified) | 12.4 / 55.5 / 34.1 / 6.81 / 15.8 | 15.9 / 78.3 / 4.60 / 0.58 / 0.97 |
| Cost-aware RF | 12.9 / 54.6 / 35.0 / 7.06 / 16.5 | 16.7 / 80.1 / 2.72 / 0.31 / 0.48 |
| Forest-weighted | 13.3 / 54.4 / 34.9 / 7.06 / 16.8 | 16.5 / 80.1 / 2.79 / 0.34 / 0.61 |
| Oracle | 27.7 / 54.1 / 24.1 / 1.48 / 0.0 | 40.5 / 58.9 / 5.02 / 0.06 / 0.0 |

At β = 2 and β = 10 the leaf-penalized rule selects an 8-leaf tree, so it is not a "simpler" variant there. Its rows
are in the CSV files.

---

## 3. Previously established vs. new

**Already reported** (stages 6–7, HW6, E1–E7). The figures above reproduce these exactly:

* **Cost:**
  * test cost;
  * % saving;
  * % of fixed→oracle gap closed;
  * cost under vCPU and vCPU-hour weighting.
* **Reservation and shortfall (single values):**
  * average reservation;
  * % under-provisioned (= exceedance);
  * mean shortfall;
  * mean unused capacity;
  * tier shares.
* **Prediction quality:** Ridge/CART/RF R² only.
* **Segment savings** for OPT, greedy and RF, by:
  * lifetime;
  * vCPU;
  * history;
  * cold start;
  * censoring.
* **Uncertainty and frontier:**
  * cost-difference bootstraps;
  * the capacity-vs-under-provisioning sweep (`final_fig3_frontier.png`).
* **HW6 forest diagnostics:**
  * mean calibration of the forest-weighted shortfalls;
  * A/B/C disagreement rates.

**New here:**

* **Shortfall severity:**
  * conditional shortfall;
  * severity quantiles;
  * > 5/10/20 pp frequencies.
* **Cost decomposition.**
* **"Targeting vs. reserving less" counterfactuals.**
* **Risk comparisons:**
  * Pareto dominance in risk dimensions;
  * matched-risk frontiers.
* **Customer-level analysis:**
  * customer-level distributions;
  * equal-subscription weighting, with a validation check.
* **Segment risk metrics,** including deployment history and realized-P95 diagnostics.
* **Prediction diagnostics:**
  * RMSE/MAE/bias/under-prediction;
  * RF calibration by bin;
  * shortfall-prediction calibration;
  * tier-error matrix.
* **Regret and disagreement:**
  * regret as % of oracle cost;
  * disagreement magnitude and concentration.
* **Bootstraps of risk metrics.**
* **Risk profiles of the 4/5-leaf and depth-4 OPT variants.**

---

## 4. Operational performance

### 4.1 Reservation, exceedance and severity (β = 5) [`ext_shortfall_severity_b5`, `ext_cost_decomposition_b5`]

* **Reservation.** All good policies reserve 72.5–74.6% of capacity on average.
  * Mean unused capacity E[(r − Y)+] falls from 0.54 (always 100%) to 0.27–0.29. The oracle's is 0.115.
  * Shortfall penalties are 7–9% of total cost for these policies, 31% for the naive RF tier and 2% for the oracle.
* **Exceedance frequency** is 7.7–10.0% for the good policies and 29.9% for the naive RF tier.
* **Severity, given exceedance:**

  | measure | cost-aware RF | OPT | greedy | 4-leaf |
  |:--|--:|--:|--:|--:|
  | median | 11.4 pp | 12.7 pp | 12.4 pp | 13.0 pp |
  | P90 | 22.8 pp | 23.2 pp | 23.4 pp | 23.6 pp |
  | P95 | **28.6 pp** | 36.7 pp | 38.3 pp | 41.7 pp |
  | P99 | 46.5 pp | 48.6 pp | 48.8 pp | 48.8 pp |

  The cost-aware RF's tail is the lightest. The P95 is the one place where the policies differ visibly.
* **Large overshoots:**
  * **By more than 10 pp:** 4.6–5.4% of VMs for the trees, 5.0% for the RF.
  * **By more than 20 pp:** 2.0–2.6% (H1: 3.5%).
* **Maximum shortfall** is 72–74 pp for every learned policy. This is an extreme single-VM statistic: a VM reserved at
  25% that ran at about 99%. The oracle's maximum is 5 pp, because it only under-reserves when the overshoot is
  cheaper than the next tier.

### 4.2 Reserving less vs. targeting [`ext_savings_mechanism_b{2,5,10}`]

Three ways to look at OPT's saving at β = 5 (pp of VM capacity unless noted):

| view | reservation | expected shortfall | result |
|:--|--:|--:|:--|
| OPT as deployed | 26.6 pp lower than always-100% | 1.30 pp (penalty: 6.5 pp of cost) | 20.1% saving |
| Same tier mix, assigned at random | same as OPT | 8.0 pp | −13.4% (costs more than always-100%) |
| Best randomized fixed-tier rule, same expected shortfall | must reserve 93.3% | 1.30 pp | 20 pp more reservation than OPT |

* **OPT avoids 84% of the untargeted shortfall.** The other good policies avoid 82–86%.
* **The pattern holds at β = 10:** 88–95% avoided.
* **At β = 2 it is reversed.** The policies reserve *more* than the 50% baseline (+4–7 pp) and save by cutting
  shortfall: 6.7–7.1 pp vs 14.5 pp.

**Conclusion:** essentially all of the saving comes from identifying *which* VMs can safely run on a lower tier. Without
that targeting the same reservation mix would be a net loss.

### 4.3 Reservation mistakes relative to the needed tier (β = 5) [`ext_tier_errors_b5`]

The "needed tier" is the smallest tier ≥ Y.

| policy | ≥ 2 tiers below | 1 below | exact | 1 above | ≥ 2 above |
|:--|--:|--:|--:|--:|--:|
| RF point → tier | 4.7 | 25.2 | 46.3 | 17.6 | 6.1 |
| RF point + margin | 0.4 | 9.5 | 48.3 | 24.1 | 17.6 |
| Forest-weighted | 0.6 | 8.5 | 53.9 | 19.4 | 17.5 |
| Cost-aware RF | 0.6 | 8.5 | 54.6 | 18.9 | 17.4 |
| OPT depth 3 | 0.7 | 8.6 | 52.5 | 19.4 | 18.9 |
| Oracle | 0 | 9.9 | 90.1 | 0 | 0 |

**Under-reservation by two or more tiers is rare** for every non-naive policy: 0.4–0.7%, about 1,700–3,000 VMs.

**About 18% of VMs are still reserved two or more tiers above need.** That is the main remaining efficiency headroom, and
it cannot be captured without better information about individual VMs.

---

## 5. Cost vs. risk

### 5.1 Magnitude vs frequency

β multiplies the *expected magnitude* E[(Y − r)+]. It does not price the *frequency* P(Y > r).

* **Even the oracle exceeds often.** The cost-optimal hindsight choice deliberately under-reserves when the overshoot is
  small, so it exceeds on 24% (β = 2), 9.9% (β = 5) and 5.0% (β = 10) of VMs.
* **Frequency alone is therefore not a failure measure.** An exceedance rate of about 9% at β = 5 is consistent with
  cost-optimal behaviour.
* **What separates the learned policies from the oracle is severity:** 13–14 pp vs 2.5 pp per exceedance.
* **Weighting frequency would need a different objective.** If an operator cares about frequency itself (e.g. per-VM
  SLO counts), the objective would have to change. We did not introduce one.

### 5.2 Pareto dominance (point estimates) [`ext_pareto`, `ext_pareto_summary`]

Dimensions compared: mean reservation, expected shortfall, exceedance and severe-shortfall frequency.

* **β = 5.**
  * Depth-3 OPT is dominated on all four dimensions jointly by **both** the cost-aware RF and the forest-weighted
    prescription. So are RF + margin and H1.
  * Greedy, the 4-leaf, leaf-penalized and depth-4 trees are *not* dominated. They reserve more and exceed less.
* **β = 10.** Greedy and the leaf-penalized tree are dominated by the RF-family. OPT is not: it reserves 0.4 pp less
  than the RF but exceeds more.
* **β = 2.** No policy is dominated on all four. The RF-family reserves less but exceeds more than the trees, a genuine
  trade-off.
* **Point-estimate dominance is not significance.** At β = 5 the RF-vs-OPT risk differences all have CIs that include
  0 (§9).

### 5.3 Matched-risk comparison [`ext_matched_risk`, `ext_risk_frontier_grid`, `ext_fig1_risk_frontier.png`]

**Method.**

* **Configurations.** Each method's existing validation-selected configurations come from the stage-10 β grid (12
  values, 1.5–20).
* **Frontier.** Randomizing between two configurations achieves the straight line between them in (risk, mean
  reservation). The lower convex hull is therefore an attainable frontier.
* **Comparison.** For each method we report the frontier reservation at depth-3 OPT's own risk level, together with the
  nearest single configuration.
* **Status of the frontier.** This is retrospective description, not model selection.

**At OPT's β = 5 risk levels:**

| method | at E[sf] = 1.30 pp | at P(Y>r) = 9.25% | at P(Y−r>20pp) = 2.56% |
|:--|--:|--:|--:|
| OPT depth 3 | 73.4 | 73.4 | 73.4 |
| Greedy depth 3 | 73.6 | 73.2 | 73.7 |
| RF point + margin | 73.7 | 73.7 | 73.7 |
| H1 customer mean + margin | 74.7 | 74.5 | 75.2 |
| Cost-aware RF | **71.9** | **72.4** | **71.6** |
| Forest-weighted | **72.0** | **72.4** | **71.5** |
| Fixed tiers (randomized) | 93.3 | 89.0 | 95.3 |

* **The RF-family's advantage holds at matched risk.** It needs 1.0–1.8 pp less reservation than OPT at OPT's own
  shortfall, exceedance or severe-shortfall level.
* **The two RF methods have close single-configuration matches:** their own β = 5 configuration, within 10% of OPT's
  risk for expected shortfall and exceedance.
* **OPT, greedy and RF + margin are indistinguishable at matched risk** (within 0.35 pp).
* **At β = 10, RF + margin and H1 have no configuration near OPT's risk** (nearest within 60–70%). Their matched values
  rely on mixing distant configurations, so we make no matched-risk claim for them there.
* **At β = 5 the fixed-tier frontier needs 15–22 pp more reservation at every risk measure.** At β = 2 it needs
  5–16 pp more.

---

## 6. Customer-level heterogeneity (subscriptions; not demographic fairness)

**Population**

* **Test subscriptions:** 1,726, with a median of 8 test VMs.
* **Primary sample:** 1,036 subscriptions with ≥ 5 test VMs, holding 99.7% of VMs.
* **Concentration:** the 94 subscriptions with ≥ 1,000 test VMs hold 78.1% of test VMs.

Baseline: the validation-selected best fixed rule for each β.

### 6.1 Distribution of per-subscription savings, β = 5 [`ext_customer_heterogeneity_b5`, `ext_fig2_customer_savings.png`]

Primary sample (≥ 5 test VMs):

| policy | VM-wtd saving | equal-sub mean | median [IQR] | % subs lower / equal / higher cost | % subs losing > 10% | median sub exceedance | % subs with > 25% exceeded |
|:--|--:|--:|:--|:--|--:|--:|--:|
| OPT depth 3 | 20.1 | 17.6 | 10.4 [0, 32.0] | 60.3 / 27.8 / 11.9 | 5.6 | 7.3% | 19.8 |
| Greedy depth 3 | 20.0 | 18.3 | 11.6 [0, 31.0] | 60.5 / 31.8 / 7.7 | 3.9 | – | 15.9 |
| OPT 4-leaf | 19.8 | 18.2 | 11.9 [0, 27.9] | 60.6 / 31.8 / 7.6 | 4.1 | – | 15.6 |
| Cost-aware RF | 21.6 | 18.9 | 12.3 [0, 32.2] | 69.6 / 22.8 / 7.6 | 2.8 | 5.1% | 12.5 |
| Forest-weighted | 21.4 | 19.0 | 12.6 [0, 32.1] | 68.6 / 22.9 / 8.5 | 3.7 | 6.2% | 13.9 |
| RF point + margin | 19.8 | 16.3 | 11.1 [0, 25.8] | 64.2 / 26.1 / 9.7 | 3.8 | – | 14.3 |

How to read this table:

* **Half the subscriptions save little.** The median subscription saves 10–13%, half of the VM-weighted figure.
* **"Equal" subscriptions are untouched.** 23–32% of subscriptions are reserved 100% throughout, so their cost equals
  the baseline.
* **Losses under OPT are moderate but sometimes large.**
  * The 12% of subscriptions with higher cost under OPT hold 3.7% of VMs.
  * Their median loss is 9.4%.
  * The worst subscription's modeled cost roughly doubles.
* **β = 10.** 7.2% of subscriptions lose more than 10% under OPT, vs 2.8% under the cost-aware RF.

### 6.2 Do large customers drive the benefit? (β = 5, OPT; RF in parentheses)

| subscription size (test VMs) | subs | % of VMs | VM-wtd saving | % subs with higher cost | share of total saving |
|:--|--:|--:|--:|--:|--:|
| 1–4 | 690 | 0.3 | 20.3 (19.6) | 12.9 (8.3) | 0.3 |
| 5–19 | 384 | 0.8 | 19.3 (19.7) | 14.1 (9.9) | 0.8 |
| 20–99 | 302 | 3.5 | 17.9 (19.9) | 12.6 (7.3) | 3.1 |
| 100–999 | 256 | 17.2 | 12.7 (14.6) | 11.3 (5.9) | 10.9 |
| ≥ 1000 | 94 | 78.1 | 21.8 (23.2) | 2.1 (4.3) | 84.8 |
| top 10 | 10 | 29.8 | 10.4 (12.1) | – | 15.4 |
| top 50 | 50 | 64.7 | 23.8 (25.1) | – | 76.5 |

* **Large customers deliver the aggregate saving, slightly more than in proportion** (85% of saving from 78% of VMs).
* **That comes from the 11th–50th largest customers.** The ten very largest save only about 10%.
* **Smaller customers' average savings are similar** (18–20%). The weak stratum is 100–999 VMs (12.7%).
* **Small customers are made worse off more often.** For the depth-3 tree the rate is 11–14% vs 2% in the ≥ 1,000
  group. For the RF-family the gap is smaller.

### 6.3 VM-weighted vs equal-subscription-weighted, β = 5 (test) [`ext_weighting_comparison_b5`, `ext_bootstrap_equal_subscription`]

Weighting each of the 1,036 subscriptions equally:

* **Savings fall from 20–22% to 16–19%.**
* **Depth-3 OPT drops to the lowest saving among the trees:**

  | policy | saving % (equal-subscription) |
  |:--|--:|
  | OPT depth 3 | 17.6 |
  | depth 4 | 18.1 |
  | 4-leaf | 18.2 |
  | greedy | 18.3 |
  | leaf-penalized | 18.5 |
  | forest-weighted | 19.0 |
  | cost-aware RF | 18.9 |

* **OPT also has the highest exceedance among the trees:**

  | policy | exceedance % (equal-subscription) |
  |:--|--:|
  | OPT depth 3 | 13.1 |
  | greedy | 11.4 |
  | 4-leaf | 11.2 |
  | cost-aware RF | 10.3 |

Paired subscription bootstrap of the equal-weighted differences (1,000 reps, unadjusted, exploratory):

| A − B (equal-subscription weighting, β = 5) | cost [95% CI] | P(Y>r) pts [CI] | E[sf] pp [CI] |
|:--|:--|:--|:--|
| Cost-aware RF − OPT | −0.013 [−0.020, −0.007] | −2.8 [−3.6, −2.1] | −0.69 [−0.84, −0.55] |
| Greedy − OPT | −0.007 [−0.011, −0.004] | −1.7 [−2.3, −1.1] | −0.26 [−0.36, −0.17] |
| 4-leaf − OPT | −0.006 [−0.011, −0.001] | −1.9 [−2.6, −1.2] | −0.22 [−0.36, −0.10] |
| Forest-weighted − RF + margin | −0.028 [−0.033, −0.022] | +1.3 [+0.6, +1.9] | +0.17 [+0.04, +0.29] |

### 6.4 The same comparison on validation [`ext_weighting_comparison_validation`]

The pattern is not a test-set artifact. On the strict validation period, under equal-subscription weighting at β = 5:

| policy | exceedance % | saving % |
|:--|--:|--:|
| OPT depth 3 | 15.9 | 14.1 |
| 4-leaf | 13.4 | 14.2 |
| greedy | 13.9 | 14.6 |
| cost-aware RF | 12.5 | 16.5 |

Hypothesis for the cause, untested: the exact tree minimizes *VM-weighted* training cost. That cost is dominated by the
largest subscriptions, so its extra leaves tune decisions for them at the expense of mid-sized subscriptions.

---

## 7. Workload segments (β = 5) [`ext_segments_saving_b5`, `ext_segments_risk_b5`, full detail in `ext_segments.csv`]

Within each segment, savings are measured against the same best fixed rule (always 100%).

**Creation-time segments (usable as policy inputs):**

* **vCPU size.**
  * 4 vCPU VMs (30% of VMs; mean P95 31%) save about 40%.
  * 2 vCPU VMs (58%; mean P95 53%) save 11–13%.
  * **> 24 vCPU VMs** (n = 4,849; mean P95 80%):

    | policy | saving | exceedance |
    |:--|--:|--:|
    | OPT | 1.7% | 4.6% |
    | cost-aware RF | 5.0% | 0.6% |
    | RF + margin | −1.1% | 12.4% |

    Large VMs are hot and offer little room.
* **Subscription history.**
  * **≥ 1,000 completed VMs (91% of VMs):** all methods save about 20–22%, with exceedance about 9%.
  * **1–49 completed VMs (n = 4,871):** OPT matches the RF's saving (22.6 vs 22.3%) but with higher exceedance (12.4%
    vs 8.0%) and twice the expected shortfall (2.5 vs 1.2 pp). This mirrors §6.
  * **No completed history (n = 577):** OPT is conservative, reserving 85% and saving 10.5% vs 18–20% for others.
    This is a small segment.
* **Deployment history.** With no completed deployment history (9.9% of VMs), savings drop to 15.8% (OPT) and 17.5%
  (RF), from 20.6% and 22.0%.

**Retrospective diagnostics only** (these use outcomes and are never model inputs):

* **Lifetime.** Short VMs (≤ 5 min) save 28–32%. VMs living 1 h–1 day are hot (mean P95 66%), save only 6–9% and have
  13–14% exceedance.
* **Realized P95.** This view is necessarily unflattering to any non-oracle policy.
  * **VMs with Y > 75%** (29.6% of VMs) all lose 10–14% vs always-100%. 23–30% of them are under-reserved.
  * **They account for about 80% of all exceedances** (OPT 79.5%, RF 75.2%).
  * **VMs that needed only 25%** (37% of VMs) are still reserved at 52–54% on average; the oracle reserves 25%.

---

## 8. Predictive performance and why it is insufficient

### 8.1 Accuracy [`ext_prediction_quality`, `ext_prediction_vs_decision`]

Test set; Ridge and CART were refit deterministically and verified against the stage-6 actions and R².

| model | R² | RMSE pp | MAE pp | bias pp | % under-predicted | P(Y − ŷ > 10pp) | P(Y − ŷ > 25pp) | P99 of Y − ŷ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Ridge | 0.494 | 25.8 | 19.5 | +1.9 | 47.9 | 32.4 | 12.5 | 52.1 |
| CART (32 leaves) | 0.491 | 25.9 | 19.4 | +0.5 | 49.6 | 33.9 | 15.6 | 54.3 |
| RF | 0.525 | 25.0 | 18.8 | −0.1 | 50.2 | 33.4 | 16.2 | 51.9 |

* **The typical error is one full tier.** The RF's RMSE (25 pp) equals the width of one reservation tier.
* **Large under-predictions are common.** One VM in six is under-predicted by more than one tier.

### 8.2 Calibration [`ext_rf_calibration`, `ext_fig4_rf_calibration.png`]

* **Mean calibration is good.** RF bin means are within ±3.4 pp of realized means, with mild shrinkage: about 3 pp
  over-prediction below 30% and under-prediction above 50%.
* **Within a bin the spread is huge.**
  * The [40, 50) bin holds 24% of test VMs. Its realized P95 has a 10th–90th percentile range of 3–84%, and spans all four
    tiers (32 / 28 / 25 / 16%).
  * "Predict, then round up" reserves 50% for that bin and exceeds on 41% of it.
  * The forest-weighted rule uses the same forest's leaf distribution. It reserves 75% on average there and exceeds
    on 16%.
* **The distribution-aware rules use calibrated shortfall predictions.**
  * Predicted shortfalls have per-VM R² of 0.42–0.48. Means are calibrated within 0.25 pp.
  * Decile calibration of E[(Y − 0.75)+] is close near the β = 5 decision threshold (5 pp).
  * The top decile is under-predicted by about 1.2 pp [`ext_shortfall_prediction_calibration`].

### 8.3 Prediction quality vs decision quality

Test % saving at β = 2 / 5 / 10:

| predictions | decision rule | saving % (β = 2 / 5 / 10) |
|:--|:--|:--|
| Ridge (R² 0.494) | point → tier | 10.6 / 14.7 / **−9.9** |
| Ridge | + val-tuned margin | 11.7 / 19.8 / 14.5 |
| CART (0.491) | point → tier | 10.8 / 11.5 / **−18.6** |
| CART | + margin | 12.1 / 18.8 / 13.1 |
| RF (0.525) | point → tier | 11.7 / 14.2 / **−12.5** |
| RF | + margin | 12.4 / 19.8 / 15.3 |
| **same RF forest** | forest-weighted prescription | 13.3 / 21.4 / 16.5 |
| separate forest | cost-aware RF (shortfall targets) | 12.9 / 21.6 / 16.7 |
| none (decides directly) | OPT depth 3 | 11.5 / 20.1 / 15.3 |

* **Accuracy differences matter little, and not in proportion to R².** With a tuned margin, the three regressors are
  within 0–2.2 pp of each other. Ridge equals the RF at β = 5 despite a lower R². Ridge beats CART at the same R².
* **The decision rule applied to the *same* forest matters far more.** Savings rise from 14.2% to 21.4% at β = 5, and
  from −12.5% to 16.5% at β = 10. This is the HW6 A/B/C comparison, where all three rules share one forest.
* **Naive rounding is worse than doing nothing at β = 10** for every regressor.

We do not compute R² between Y and the chosen action. A reservation is a decision, not a prediction.

---

## 9. Oracle regret, disagreement and statistical evaluation

### 9.1 Regret [`ext_oracle_regret`]

The oracle is an infeasible hindsight benchmark.

* **Regret as % of oracle cost** at β = 5: OPT 35.7%, RF 33.1%, greedy 35.9%, depth-4 35.0%.
* **Share of attainable saving captured:**

  | policy | β = 2 | β = 5 | β = 10 |
  |:--|--:|--:|--:|
  | OPT | 42 | 49 | 38 |
  | cost-aware RF | 47 | 53 | 41 |
  | forest-weighted | 48 | 52 | 41 |

### 9.2 Disagreement [`ext_disagreement_matrix`, `ext_disagreement_pairs_b{2,5,10}`]

**Share of test VMs with different actions, β = 5:**

| pair | % of VMs |
|:--|--:|
| OPT vs greedy | 9.5 |
| OPT vs depth 4 | 9.4 |
| OPT vs 4-leaf | 12.2 |
| OPT vs forest-weighted | 14.6 |
| OPT vs cost-aware RF | 16.9 |
| cost-aware RF vs forest-weighted | 8.7 |

**Where disagreements occur:**

* **Disagreements are almost always one tier apart** (mean |Δr| 25–27 pp).
* **The cost difference lives entirely in the disagreement set,** and it is mostly offsetting.
  * Cost-aware RF vs OPT: a net −0.015 per VM out of a gross 0.060.
  * The RF is cheaper on 60% of the disagreeing VMs.
* **Not concentrated at the VM level.** The top 10% of disagreeing VMs carry 25–28% of the gross difference.
* **Concentrated by subscription.** The 10 subscriptions with the largest net differences carry 41–61% of the gross
  subscription-level difference.

### 9.3 Paired subscription-cluster bootstrap [`ext_bootstrap`]

1,000 reps, seed 0, percentile CIs.

* **Unadjusted for multiplicity:** 13 comparisons × 5 metrics.
* **Conditional on the trained models.**
* **Cost rows reproduce the earlier CIs exactly;** the risk rows are new.

All values are A − B. Bold marks a CI that excludes 0.

| A − B | β | cost | E[r] pp | P(Y>r) pts | E[sf] pp | P(sev) pts |
|:--|--:|:--|:--|:--|:--|:--|
| Cost-aware RF − OPT | 2 | **−0.011** | **−1.9** | **+2.3** | +0.38 | +1.3 |
| | 5 | **−0.015** | −0.9 | −0.1 | −0.12 | −0.37 |
| | 10 | **−0.014** | +0.4 | −0.7 | **−0.18** | **−0.44** |
| Forest-weighted − RF + margin | 2 | **−0.007** | **−1.8** | +0.8 | **+0.55** | +0.6 |
| | 5 | **−0.017** | −0.7 | −0.8 | **−0.19** | **−0.58** |
| | 10 | **−0.012** | **+3.2** | **−3.1** | **−0.44** | **−0.86** |
| 4-leaf − OPT (8 leaves) | 2 | +0.001 | −0.5 | +0.4 | +0.28 | +0.5 |
| | 5 | +0.003 | **+1.2** | **−1.5** | −0.18 | −0.35 |
| | 10 | −0.001 | −0.2 | +0.4 | +0.02 | +0.12 |
| 5-leaf penalized − OPT | 5 | −0.003 | +0.1 | −0.7 | −0.09 | −0.12 |
| Depth-4 − depth-3 OPT | 2 | **−0.007** | −1.0 | +1.4 | +0.13 | +0.6 |
| | 5 | −0.004 | +0.4 | −0.8 | −0.16 | **−0.53** |
| | 10 | **−0.006** | −1.5 | **+1.2** | +0.09 | +0.06 |

* **At β = 5, the cost-aware RF's advantage over OPT is not bought with risk.** Every risk point estimate favours the RF,
  though no single risk CI excludes 0.
* **The forest-weighted vs margin comparison is mostly a risk improvement** at β = 5 and β = 10. At β = 10 the margin
  rule under-reserves.
* **4/5-leaf vs 8-leaf.** Cost is not distinguishable; at β = 5 the 4-leaf tree is more conservative.
* **Depth 4.** The cost gain at β = 10 comes with more exceedance; at β = 5 the tail is lighter.
* **Non-significant differences are not evidence of equivalence.**

---

## 10. Implications for the report

The report is not modified here.

1. **Add risk next to cost.** Add P(Y > r), E[(Y − r)+], E[Y − r | Y > r] and P(Y − r > 20 pp) to the main benchmark
   table. One sentence then makes the "magnitude vs frequency" point: the oracle also exceeds on about 10% of VMs at
   β = 5.
2. **Use the targeting counterfactual as the strongest single argument that the savings are real:**
   *a random assignment of OPT's own tier mix would cost 13% more than always-100%.*
3. **Keep depth-3 OPT as the interpretable headline,** but:
   * **Show the 4-leaf exact tree alongside it.** It has the same cost (CI includes 0) and significantly lower
     exceedance at β = 5. It also does better at the customer level, on test and on validation.
   * **Do not claim OPT is lower-risk than the forest rules.** At β = 5 the forest rules are at least as good on every
     risk metric.
4. **One short paragraph on customer heterogeneity.**
   * The median subscription saves about 10–13%.
   * About 8–12% of subscriptions are made worse off.
   * ≥ 1,000-VM customers drive about 85% of the saving.
   * Equal-subscription weighting reverses the ordering of OPT and greedy.
5. **Candidate figures:**
   * `ext_fig1_risk_frontier.png` (panel a or b) — the clearest new picture;
   * `ext_fig4_rf_calibration.png` — for the "prediction is not enough" argument.

   `ext_fig2` and `ext_fig3` are appendix material.

---

## 11. Limitations

* **Y is the lifetime P95 CPU utilization.**
  * A positive Y − r means the P95 exceeded the reservation. It is not necessarily throttling or an SLO violation.
  * Shortfall severity is in modeled units, not latency or revenue.
* **Censoring.** Right-censored test VMs (alive at trace end) carry partial P95s; stage 7 showed the headline is
  insensitive to excluding them.
* **Weighting.** All risk metrics are per VM.
  * Cost under vCPU and vCPU-hour weighting exists from stage 6.
  * Risk under those weightings was not recomputed here.
* **The equal-subscription analysis is exploratory.**
  * Subscription size is confounded with workload type.
  * The bootstrap there treats the 1,036 subscriptions as the sampling units.
* **Matched-risk frontiers are retrospective test descriptions.** They are built from randomized mixtures of existing
  configurations and are not a deployable selection procedure.
* **Bootstrap CIs:**
  * unadjusted for about 77 intervals;
  * conditional on the fitted models and the chosen hyperparameters;
  * not adjusted for the β grid.
* **Depth-4 OPT** actions are derived by applying the saved certified trees; no optimization was rerun.
* **The maximum shortfall is a single-VM statistic.** Tail quantiles above P99 rest on a few hundred VMs.

---

## 12. Files and reproducibility

**Run**

* Command: `./run_extended_evaluation.sh` (add `--skip-tests` to skip the suite; `N_BOOT` and `MIN_SUB_VMS` env vars).
* Cost: about 25 s of evaluation, peak RSS 3.4 GB, seed 0, 1,000 bootstrap reps.
* Tests: the full suite passes (63 tests: 48 earlier + 15 new in `tests/test_extended_evaluation.py`).

**Code**

* `src/12_extended_evaluation.py`.
* `src/evaluation_metrics_lib.py`: pure metric helpers, unit-tested for:
  * zero exceedances;
  * extreme and threshold shortfalls;
  * weighting = duplication;
  * equal-subscription weighting;
  * oracle regret;
  * the permutation counterfactual;
  * mixture frontiers;
  * Pareto dominance;
  * disagreement accounting;
  * bootstrap reproduction.

**Results** (`results/extended_evaluation/`)

| topic | files |
|:--|:--|
| main and severity | `ext_main_table_b{2,5,10}`, `ext_operational_metrics.csv`, `ext_shortfall_severity_b*`, `ext_cost_decomposition_b*` |
| mechanism | `ext_savings_mechanism*` |
| tier errors | `ext_tier_errors*` |
| risk frontier | `ext_pareto*`, `ext_risk_frontier_grid.csv`, `ext_matched_risk` |
| customers | `ext_customer_heterogeneity*`, `ext_customer_per_subscription_b5.csv`, `ext_weighting_comparison*` (incl. `_validation`), `ext_bootstrap_equal_subscription` |
| segments | `ext_segments*` |
| prediction | `ext_prediction_quality`, `ext_prediction_vs_decision`, `ext_rf_calibration`, `ext_shortfall_prediction_calibration` |
| regret and disagreement | `ext_oracle_regret`, `ext_disagreement_matrix`, `ext_disagreement_pairs*` |
| statistics | `ext_bootstrap` |
| run record | `ext_runlog.json` (versions, runtime, the 84 checks), `ext_log.txt` |

**Figures** (`figures/extended_evaluation/`)

* `ext_fig1_risk_frontier.png`
* `ext_fig2_customer_savings.png`
* `ext_fig3_shortfall_severity.png`
* `ext_fig4_rf_calibration.png`

**Integrity.** SHA-256 hashes of all 412 pre-existing results, figures, processed data, models, findings documents,
source files and tests were recorded before this stage. They were byte-identical afterwards. The only later change to a
pre-existing file is the stage-12 paragraph appended to `README.md`.
