# β sensitivity and OPT complexity: findings

*Extensions E1–E7 to the finalized project (`FINAL_FINDINGS.md`, `HW6_FINDINGS.md`).*

* **Code:** `src/sensitivity_lib.py`, `src/10_beta_sensitivity.py`, `src/11_opt_complexity.py`, plus a
  backward-compatible `leaf_penalty` / depth-4 extension of `src/final_lib.py`.
* **Run:** `./run_sensitivity.sh` (add `--depth4` for E7).
* **Outputs:** `results/sensitivity/`, `figures/sensitivity/`.
* **Protocol:** the strict chronological split (1,703,020 train / 275,143 validation / 426,846 test VMs). Every
  selection uses validation; **test numbers are retrospective evaluation only**.
* **Originals untouched:** all 297 finalized files (stage 1–7 results and figures, HW6 outputs, saved models, processed
  data, the three findings documents) are byte-identical before and after this work (SHA-256).
* **Tests:** 48 pass (33 existing + 15 new).
* **Runtime** (Apple M4, single-threaded solver):
  * E1 under 1 s;
  * E2 about 7 min (including one forest refit, 48 s);
  * E3 28 min and 20 min for the two exact paths, run in parallel;
  * E4 under 1 s;
  * E5 seconds; E6 about 4.5 min;
  * E7 28 min (probe) plus 36 min (three runs in parallel).

  Peak memory is ≤ 2 GB per process.

**β is an economic input, not a hyperparameter.** Costs at different β are never compared to choose β. Within each β,
policies are compared by % saving vs the best fixed tier chosen on validation, by regret, or by paired differences.

## 1. Executive summary

1. **The headline β = 5 savings sit near the most favourable point of the β curve.** OPT's test saving vs the best
   fixed rule is 5.2% at β = 1.5, 11.5% at 2, 15.9% at 3, 17.5% at 4, **20.1% at 5**, 18.9% at 6, 15.3% at 10 and
   13.3% at 20.
   * **Why β = 5 is the peak:** it lies just above the validation β (4.46) at which the best fixed tier jumps from 75%
     to 100%, where a uniform rule is most wasteful.
   * **Test-hindsight baseline:** on the test period that switch happens at β = 5.12, so a hindsight best fixed rule
     would reserve 75% at β = 5. Against that stronger, retrospective baseline the β = 5 savings are 0.45 points lower
     (OPT 19.6%, RF 21.1%). β = 2 and β = 10 are unaffected.
2. **The method ranking does not depend on β.** At every one of 12 β values from 1.5 to 20:

   | rank | methods |
   |---|---|
   | 1 | forest-weighted prescription ≈ cost-aware RF |
   | 2 | exact OPT ≈ greedy policy tree |
   | 3 | CART-MSE ≈ prediction + margin |
   | 4 | customer heuristics |
   | 5 | naive predict-then-round (worse than the fixed rule for β ≥ 8) |

   * **OPT vs RF:** OPT keeps 88–94% of the RF's savings, and the RF is cheaper at all 12 β (CIs exclude 0).
   * **OPT vs greedy:** differences are −0.67% to +0.12%. The CI excludes 0 only at β = 8, 10 and 15, all in OPT's
     favour and all ≤ 0.7%.
3. **The β = 5 policy is robust to a moderately wrong β.** If the true β is anywhere in [4, 8], the tree built for β = 5
   costs < 1% more on test than a tree re-optimized for the true β. Regret rises to about 2–3% at β = 3 or 10, 9% at
   β = 2 or 15, and 15–17% at β = 1.5 or 20. The cost-aware RF's β = 5 decisions behave almost identically.
4. **The optimal tree changes shape often; its behaviour changes smoothly.**
   * **Change points:** exact enumeration over β ∈ [1, 20] finds 120 change points with min leaf 1%, and 82 with 5%,
     mostly at β < 2.5. The root split changes 49 and 36 times.
   * **Training (guaranteed):** reserved capacity rises and shortfall falls monotonically along the path, and V(β) is
     concave, as theory requires.
   * **Validation and test:** the same trees show only tiny reversals (≤ 0.7 percentage points of capacity), so
     reserved capacity rises and shortfall falls with β there too, for all practical purposes.
   * **Upshot:** individual tree structure is not identifiable across β, but the policy's operating characteristics are.
5. **Simpler trees are almost free.**
   * **4 leaves:** a 4-leaf exact OPT (depth 2) keeps 94–100% of the 8-leaf tree's savings at every β. Its test cost is
     not significantly different at β = 2, 5 or 10 (largest difference +0.0028 at β = 5).
   * **Leaf penalty:** the exact λ-penalty hull (1–8 leaves) shows 5–8 leaves are interchangeable at β = 5. With a
     simplicity tie-break, validation selects a **5-leaf** tree whose test cost (0.7961) is no worse than the original
     8-leaf OPT (0.7990).
6. **Exact depth-4 OPT is feasible and certified, but buys little.**
   * **Feasibility:** each solve takes 28–36 minutes on 1.70M training VMs with the original 32-bin candidate set, and
     every run proved optimality.
   * **Gain:** 0.5–1.0% lower test cost than depth-3 OPT, with about twice the leaves (15–16). The gain is significant
     (marginally) at β = 2 and 10, but not at β = 5.
   * **Comparisons:** at β = 5 it ties depth-4 greedy and IAI. At β = 2 and 10 it is statistically indistinguishable
     from the cost-aware RF.

**What changes in the project's claims.** The ranking, the OPT-vs-greedy and OPT-vs-RF conclusions, and the
interpretability story all survive. Depth-4 exact OPT narrows but does not close the gap to the RF at β = 5. Two
presentation changes are needed:
* **Headline savings:** quote the β = 5 saving as "about 20% vs the validation-chosen fixed rule (19.6% vs the
  test-hindsight fixed tier)".
* **Robustness context:** add one sentence saying savings are 15–20% for β ∈ [3, 10] and that β = 5 is near the peak.

## 2. Previously established vs new

| topic | before this study | added here |
|---|---|---|
| β values | 2, 5, 10, with policies re-optimized at each (`final_cost_by_beta`) | 12-point validated grid; fixed-policy curves for any β; exact switch points; wrong-β regret |
| best fixed rule | chosen on validation at 2 / 5 / 10 | exact switch β on train / val / test, and the β = 5 hindsight caveat |
| OPT structure across β | three saved trees (roots on the same feature) | every exact change point on [1, 20] at two leaf sizes, with monotonicity checks |
| complexity | depth 1–3 × min-leaf grid (depth was the only binding control) | 4- vs 8-leaf comparison at 12 β; exact λ-penalty hull with 1–8 leaves; exact depth 4 (§8) |
| solver | exact depth ≤ 3 | backward-compatible `leaf_penalty` and exact depth-4 search, both brute-force tested |

## 3. E1: fixed-policy sensitivity (algebra on saved actions; no training)

For a fixed action vector the cost is linear: E[C] = E[r] + β·E[(Y − r)+]
[`s10_e1_policy_lines.csv` covers all 69 saved policy × β columns, on validation and test].

**Best fixed tier switch points** [`s10_e1_best_fixed_switches`]:

| split | 25% → 50% | 50% → 75% | 75% → 100% |
|---|---:|---:|---:|
| train | 1.74 | 2.49 | 4.75 |
| **validation (selection set)** | **1.60** | **2.29** | **4.46** |
| test (retrospective) | 1.78 | 2.61 | 5.12 |

β = 5 sits between the validation (4.46) and test (5.12) switch. Hence the headline note
[`s10_e1_headline_robustness`]:

| policy at β = 5 | saving vs best fixed (val: 100%) | saving vs test-hindsight best fixed (75%) |
|---|---:|---:|
| OPT-exact | 20.10% | 19.64% |
| greedy tree | 19.96% | 19.51% |
| cost-aware RF | 21.59% | 21.15% |
| H1 heuristic | 19.14% | 18.68% |

**Break-evens between the β = 5 versions of policies, if the true β differs** [`s10_e1_break_evens`]:
* **OPT vs greedy:** they cross at β = 5.25 (validation) and 5.84 (test). Below that OPT is cheaper, above it greedy
  is, because greedy reserves more. The two trees sit at nearly the same point of the trade-off.
* **OPT vs the RF:** no crossing for β ≥ 1; the RF's β = 5 decisions are cheaper at every β.
* **OPT vs always-100%:** OPT stays cheaper until β ≈ 17 (validation) or 20 (test).
* **OPT vs always-75%:** OPT is cheaper at every β ≥ 1.

**Leaf-level switch points of the saved β = 5 OPT** (training statistics; [`s10_e1_leaf_switch_points`]). Each leaf's
best tier only rises with β, since slopes decrease with the tier:
* **The 25% leaf** (15.5% of training VMs; near-idle customers) keeps 25% for all β ∈ [1, 20].
* **The largest 100% leaf** (36% of VMs) would already pick 100% from β = 2.2.
* **The 75% leaves** switch to 100% at β = 7.6, 9.3 and 17.8.
* **The 50% leaves** move to 75% at β = 8.1, or stay at 50% up to 20.

So small β changes leave most leaf decisions unchanged.

## 4. E2: re-optimized sensitivity on a validated β grid

**What is re-solved at each β:** OPT-exact, greedy and CART-MSE trees (depth 1–3 × five min-leaf sizes, selected on
validation per β), plus re-tuned margins.

**What is re-decided without retraining:** the cost-aware RF (saved predictions) and the forest-weighted prescription
(one deterministic forest refit, identical to HW6).

**Consistency checks** (all passed):
* at β = 2 / 5 / 10 the re-solved trees reproduce stage 5's test costs exactly;
* the RF, margin and forest decisions reproduce stage 6 and HW6 bit for bit;
* an integrity check recomputes every selected policy's cost from its stored actions.

**% saving vs best fixed (validation-chosen tier), test period** [`s10_e2_saving_by_beta`]:

| policy | 1.5 | 2 | 3 | 4 | **5** | 6 | 7 | 8 | 10 | 12 | 15 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best fixed tier (val) | 25% | 50% | 75% | 75% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| H1 customer mean + margin | 4.6 | 11.3 | 15.3 | 16.9 | 19.1 | 17.5 | 16.8 | 16.0 | 14.3 | 12.7 | 8.8 | 8.3 |
| RF point → tier (naive) | 1.0 | 11.7 | 16.3 | 15.0 | 14.2 | 8.9 | 3.6 | −1.8 | −12.5 | −23.2 | −39.2 | −65.9 |
| RF point + margin | 5.3 | 12.4 | 16.3 | 16.7 | 19.8 | 18.6 | 17.6 | 16.6 | 15.3 | 13.7 | 11.4 | 9.0 |
| CART-MSE | 4.7 | 11.6 | 15.5 | 16.8 | 19.6 | 16.9 | 16.4 | 15.8 | 14.6 | 14.0 | 12.4 | 11.6 |
| greedy tree | 4.9 | 11.4 | 16.0 | 17.3 | 20.0 | 19.0 | 17.8 | 16.3 | 14.8 | 14.5 | 13.9 | 13.2 |
| **OPT-exact** | **5.2** | **11.5** | **15.9** | **17.5** | **20.1** | **18.9** | **17.8** | **16.8** | **15.3** | **14.8** | **14.1** | **13.3** |
| forest-weighted prescription | 6.1 | 13.3 | 18.0 | 19.0 | 21.4 | 20.1 | 19.2 | 18.2 | 16.5 | 15.4 | 14.2 | 13.6 |
| cost-aware RF | 5.9 | 12.9 | 17.7 | 18.9 | 21.6 | 20.2 | 19.1 | 18.1 | 16.7 | 16.0 | 15.3 | 14.4 |
| oracle (hindsight bound) | 17.2 | 27.7 | 35.2 | 38.0 | 41.1 | 40.9 | 40.8 | 40.6 | 40.5 | 40.4 | 40.3 | 40.2 |

**How OPT's operating point moves with β** [`s10_e2_opt_by_beta`; tier shares in `sens_fig2`]:
* **Average reserved capacity** rises from 0.42 to 0.85.
* **Share of VMs under-provisioned** falls from 53% to 0.7%.
* **Share of VMs at 100%** rises from 2% to 70%. The 25% tier stays at about 14–17% for β ≥ 4 (20% at β = 3);
  these are the near-idle customers.
* **Selected trees:** always depth 3 (7–8 leaves), but with different roots across β (§5).
* **Cost-aware RF:** its tier shares follow almost the same path. The oracle's do not: it keeps about 38–46% of VMs at
  25% for every β, showing how much per-VM information a creation-time policy cannot recover.

**Paired subscription-cluster bootstrap at each β** [`s10_e2_bootstrap`; 1,000 reps; unadjusted, conditional on the
trained models]:
* **OPT − cost-aware RF:** +0.7% to +2.1% of the RF's cost; CI excludes 0 at **12/12** β.
* **OPT − forest-weighted:** +0.1% to +2.5%; CI excludes 0 at 9/12 β.
* **OPT − greedy:** −0.67% to +0.12%; CI excludes 0 at 3/12 β (8, 10, 15, all favouring OPT, magnitudes ≤ 0.7%,
  upper limits within 0.0016 of zero). This is consistent with "no material difference".
* **OPT − best fixed:** CI excludes 0 at 12/12 β.

## 5. E3: exact change points of the optimal depth-3 tree in β

**Method.** V(β) = min over trees of (A + β·B) is concave and piecewise linear. A parametric search with the exact solver
recovers *every* breakpoint on β ∈ [1, 20] [`s10_e3_change_points`, `s10_e3_monotonicity`, `sens_fig3`]:

| min leaf | distinct optimal trees | change points | root changes | exact solves | coverage | runtime |
|---|---:|---:|---:|---:|---|---|
| 1% (17,030 VMs) | 121 | 120 | 49 | 241 | **complete** (no unresolved interval) | 28 min, 0.9 GB |
| 5% (85,151 VMs) | 83 | 82 | 36 | 165 | **complete** | 20 min, 0.9 GB |

**Where the change points fall:**
* **β < 2.5:** 57–66% of them, where the policy moves from "mostly 25%" to "mostly 75/100%".
* **β ∈ [4, 8]:** 10–14 of them.
* **β > 8:** 10–15 of them.
* **The saved β = 5 tree** (min leaf 5%) is optimal on β ∈ [4.60, 5.27]; with min leaf 1%, the tree optimal on
  [4.78, 5.21] has the same root (share of past VMs with P95 ≤ 75% ≤ 0.858).

**Monotonicity:**
* **Training (guaranteed):** reserved capacity is non-decreasing and shortfall non-increasing at every change point,
  and V(β) is concave (verified).
* **Validation and test (not guaranteed):** the same trees show 3–7 small reversals, all ≤ 0.65 percentage points of
  capacity.

**Out-of-sample vs in-sample change points.** Where two adjacent trees cost the same on test differs from the training
change point by a median of 0.23–0.32 in β. For 6–8% of adjacent pairs there is no test crossover in range at all.
In-sample change points are therefore a good but imperfect guide to out-of-sample policy differences.

**Interpretation.** Many structurally different depth-3 trees are near-optimal at any β, so the identity of the root
split is *not* a stable finding. The variable families are stable (the customer's upper-tail share, P95 level, VM
size, deployment mean), and so is the operating point.

## 6. E4: cost of assuming the wrong β

Each regret below compares the policy built for β = 5 with one re-optimized for the true β, as % extra cost
[`s10_e4_regret_vs_tuned`, `s10_e4_regret_same_class`, `sens_fig1`].

| true β | 1.5 | 2 | 3 | 4 | 6 | 7 | 8 | 10 | 12 | 15 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OPT, vs re-tuned OPT, **test** | 17.3 | 8.8 | 2.6 | 0.8 | 0.1 | 0.4 | 0.8 | 2.1 | 4.5 | 8.2 | 14.8 |
| OPT, vs re-tuned OPT, validation | 15.7 | 7.9 | 3.3 | 1.1 | 0.2 | 1.0 | 1.6 | 3.2 | 5.4 | 9.2 | 16.1 |
| OPT, vs same-class optimum, train (≥ 0 by construction) | 14.4 | 6.9 | 1.7 | 0.3 | 0.2 | 0.8 | 1.5 | 3.2 | 5.5 | 9.1 | 15.6 |
| cost-aware RF decision rule, test | 16.5 | 8.9 | 3.0 | 0.7 | −0.3 | −0.2 | 0.1 | 1.3 | 3.2 | 6.6 | 12.4 |
| best fixed tier (100% at β = 5), test | 47.6 | 26.7 | 11.5 | 5.8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Interpretation:**
* **Mis-specifying β by a factor of up to about 1.6 (true β in [4, 8]) costs < 1%.** Off by 2×, it costs about 2–5%
  (true β = 10: 2.1%; true β = 2.5: roughly 5%, interpolated between 8.8% at β = 2 and 2.6% at β = 3). Off by 3–4×,
  it costs about 8–17%.
* **Over-estimating β is cheaper than under-estimating it by the same factor.** A conservative (higher) β is the safer
  assumption.
* **Slightly negative test regret** for the RF rule at β = 6–7 is noise: the β = 5 rule happened to do marginally
  better out of sample.

## 7. E5–E6: complexity, leaves and an explicit leaf penalty

**E5: 4 leaves vs 8 leaves** (exact OPT, depth 2 vs depth 3, each validation-tuned) [`s11_e5_four_vs_eight_leaves`]:
* **Share of depth-3 savings captured by the 4-leaf tree:** 95.5 / 98.7 / 99.6 / 99.0 / **98.6** / 97.8 / 99.4 / 93.7 /
  100.4 / 97.2 / 98.5 / 97.1% at β = 1.5 … 20.
* **Paired test difference (4-leaf − 8-leaf):**

  | β | difference | 95% CI | significant? |
  |---|---:|---|---|
  | 2 | +0.0011 | [−0.0013, 0.0040] | no |
  | 5 | +0.0028 | [−0.0017, 0.0079] | no |
  | 10 | −0.0006 | [−0.0053, 0.0040] | no |

* **2 leaves (depth 1)** are clearly worse (13% saving at β = 5).

**E6: exact λ-path of mean cost + λ·#leaves** (depth ≤ 3, min leaf 0.1%) [`s11_e6_lambda_hull`, `s11_e6_selection`,
`sens_fig4`].

*Correctness.* The penalty adds a constant to every leaf, so the objective stays a sum over leaves and the exhaustive
recursion stays exact. Brute force agrees on small instances (`tests/test_sensitivity.py`). The path is exact: 13–15
solves per β, complete.

*Hull.* Optimal trees exist on the lower convex hull for every leaf count 1–8 at β = 2 and 5 (β = 10 skips 6 leaves):

| β = 5 leaves | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| training cost (unpenalized) | 0.8013 | 0.8018 | 0.8029 | 0.8045 | 0.8066 | 0.8254 | 0.8631 | 1.000 |
| validation cost | 0.8402 | 0.8410 | 0.8417 | 0.8406 | 0.8430 | 0.8557 | 0.8898 | 1.000 |
| test cost (retrospective) | 0.7957 | 0.7958 | 0.7959 | 0.7961 | 0.8019 | 0.8119 | 0.8676 | 1.000 |

*Selection.*
* **Minimum validation cost** picks the 8-leaf tree at every β.
* **"Simplest within 0.1% of the minimum validation cost"** picks 8 leaves at β = 2 and 10, but **5 leaves at β = 5**
  (λ ∈ [0.0016, 0.0021]). Its test cost is 0.7961 vs 0.7990 for the original OPT: −0.0029, CI [−0.0101, 0.0027], not
  significant.
* **The penalized training objective is never presented as allocation cost.** The tables report the unpenalized cost
  separately.

**Does an explicit penalty improve the selected policy?** Not in cost. It adds what depth alone could not: an exact
cost-versus-leaves frontier at every leaf count. That frontier shows the original 8-leaf model can be replaced by a
5-leaf model at β = 5 with no measurable loss, and by a 4-leaf model at about +0.003–0.006 test cost. Leaf counts 3–7 do
not beat 8 out of sample; 5–8 are equivalent.

## 8. E7: exact depth-4 OPT

**Method.** Every depth ≤ 4 tree is either a depth ≤ 3 tree or a root split with two depth ≤ 3 children. The cost is
separable across children, so solving both children exactly for every feasible root split is exact. Root splits whose
children cannot hold a minimum-size leaf are skipped, which is also exact. Root splits are evaluated best-first, so a
run stopped by its budget would still hold a valid (but uncertified) incumbent. Brute-force tests confirm exactness on
small instances. Same 8 features and 32-bin candidate thresholds as the original OPT.

**Stage A probe** (β = 5, min leaf 5%, 30-minute budget): **completed and certified OPTIMAL in 28.3 min** (154/154 root
splits, peak 1.6 GB). The original configuration was feasible, so **Stage C (16 bins) was not needed and not run.**

**Stage B** (three single-threaded processes in parallel, min leaf 1%, the depth-3 choice at β = 2 / 10, also used at
β = 5 for selection): each completed and certified OPTIMAL in about 36 min (164/164 root splits, ≤ 0.8 GB)
[`s11_e7_depth4`, `s11_e7_bootstrap`, `s11_e7_selected_vs_existing`, `sens_fig5_depth4`].

| β | min leaf | depth-3 train / val / test | **depth-4 train / val / test** | depth-4 leaves | depth 4 − depth 3 (test, 95% CI) |
|---|---|---|---|---:|---|
| 2 | 1% | 0.7114 / 0.7407 / 0.6985 | **0.7081 / 0.7350 / 0.6914** | 16 | −0.0071 [−0.0124, −0.0016] |
| 5 | 1% | 0.8013 / 0.8402 / 0.7957 | **0.7974 / 0.8326 / 0.7950** | 16 | −0.0007 [−0.0076, 0.0081] |
| 5 | 5% | 0.8016 / 0.8396 / 0.7990 | **0.7996 / 0.8377 / 0.7947** | 10 | −0.0044 [−0.0108, −0.0001] |
| 10 | 1% | 0.8408 / 0.8782 / 0.8468 | **0.8352 / 0.8729 / 0.8409** | 15 | −0.0059 [−0.0125, −0.0007] |

**Validation-selected depth-4 vs the existing policies** (test; paired cluster bootstrap):

| β | vs original depth-3 OPT | vs greedy depth 4 | vs IAI depth 4 (200k) | vs cost-aware RF |
|---|---|---|---|---|
| 2 | −0.0071 [−0.0124, −0.0016] (−1.0%) | −0.0057 [−0.0104, −0.0011] | −0.0017 [−0.0047, 0.0013] | +0.0040 [−0.0004, 0.0082] |
| 5 | −0.0040 [−0.0088, 0.0002] (−0.5%) | +0.0006 [−0.0057, 0.0090] | −0.0024 [−0.0116, 0.0064] | +0.0109 [0.0040, 0.0195] |
| 10 | −0.0059 [−0.0125, −0.0007] (−0.7%) | −0.0085 [−0.0157, −0.0025] | −0.0010 [−0.0037, 0.0014] | +0.0082 [−0.0016, 0.0207] |

Validation picks min leaf 1% for depth 4 at β = 5 (validation cost 0.8326 vs 0.8377 at 5%).

**Interpretation:**
* **In and out of sample.** Depth 4 is strictly better in sample (as it must be) and on validation at every β. On test
  it is 0.5–1.0% cheaper than depth-3 OPT. That is marginally significant at β = 2 and 10 (upper limits −0.0016 and
  −0.0007), and not at β = 5.
* **Exact vs heuristic depth 4.** Exact depth-4 beats greedy depth-4 at β = 2 and 10, ties it at β = 5 (greedy is
  slightly lower on test, 0.7944, despite a worse validation cost), and is indistinguishable from IAI's depth-4 local
  search everywhere.
* **Against the RF.** The remaining gap is no longer significant at β = 2 and 10; at β = 5 the RF is still 1.4% cheaper.
* **Cost of interpretability.** Depth 4 has 15–16 leaves vs 8, so it trades readability for about 0.5–1% cost.

**Hypothesis classes.** Depth-4 exact and depth-4 greedy are compared within the same class (depth ≤ 4, same 32-bin
candidates; greedy used 0.1–1% leaves). Comparisons with depth-3 OPT are across classes and labelled as such.

## 9. Uncertainty, incidents and limitations

* **Bootstrap CIs** are percentile intervals over 1,000 paired resamples of 1,726 test subscriptions. They are
  *unadjusted* for the 48 comparisons in E2 (plus E5–E7) and condition on the trained models, so they capture test
  sampling noise, not retraining noise. Only the OPT-vs-RF and policy-vs-best-fixed gaps are robust across β. The
  OPT-vs-greedy exclusions are tiny and marginal.
* **Operational relevance.** Differences of ≤ 0.3% (OPT vs greedy, 4 vs 8 leaves, penalized vs original) are about
  0.002–0.003 VM-capacity units per VM. That is negligible next to the 15–20% gap to the fixed rule, or the 1.5–2% gap
  to the RF.
* **Calibrating β.** β cannot be learned from this trace. It would need operational evidence:
  * the SLO and latency impact of CPU throttling as a function of the shortfall;
  * how much P95 shortfall becomes unmet demand rather than queueing;
  * the marginal value of capacity released by smaller reservations (consolidation, deferred hardware);
  * customer-specific SLAs, since β plausibly differs by workload class.

  E4's asymmetry (over-estimating β is cheaper) is the practical guidance in the meantime.
* **Incident (detected and resolved).** In the first stage-10 run, the cached β = 12 test cost matrix changed partway
  through execution. Cross-checks flagged it: the same tree had different test costs depending on when it was
  recorded, while its training and validation costs were consistent. A faithful replay with checksums at each step did
  not reproduce it, so the root cause is undetermined and appears transient. Mitigation:
  * stage 10 now computes every cost directly from stored action vectors;
  * it asserts that each selected policy's recorded validation and test costs equal a recomputation from its actions;
  * all reported numbers come from the clean rerun, which passed these assertions.

  No finalized result is implicated. Those were reproduced bit for bit in several earlier reruns and in the audit's
  recomputation from saved actions.
* **Other limitations:**
  * The tree features were chosen once, at β = 5, and reused across β.
  * E3 and E6 run at fixed min-leaf sizes.
  * The E2 grid has 12 points, but E1 curves and E3 change points are exact on the continuum.
  * Modeled-cost caveats from the main project apply unchanged.

## 10. What deserves space in the 8-page report

1. **Essential (one figure, three sentences):** `figures/sensitivity/sens_fig1_fixed_vs_reoptimized.png`.
   * The ranking is stable across β.
   * Savings are 15–20% for β ∈ [3, 10], peaking near β = 5 because of the fixed-rule switch.
   * Assuming β = 5 costs < 1% if the true β is in [4, 8].

   Also add the one-line headline caveat (19.6% vs the test-hindsight fixed tier).
2. **Useful (one sentence + one table row):** "a 4-leaf exact OPT keeps about 99% of the 8-leaf savings (n.s.), and the
   exact leaf-penalty frontier shows 5–8 leaves are equivalent". Optionally `sens_fig4_complexity.png` in the appendix.
3. **Appendix:**
   * `sens_fig3_opt_change_points.png`: exact change points; structure unstable, behaviour smooth and monotone in
     training. This is a nice Chapter 8 point about the parametric structure of exact trees.
   * `sens_fig2_tier_shares.png`.
   * The E7 depth-4 result (§8) and `sens_fig5_depth4.png`: exact depth 4 is feasible at full scale (about 30 min,
     certified) and buys 0.5–1% at twice the leaves, a good Chapter 8 talking point.
4. **Not needed in the main text:** the leaf-level switch tables and the full break-even matrix (keep them in results).
