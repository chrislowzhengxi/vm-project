# Report revision summary

## Revision 4: figure design pass (presentation only)

No result, number or narrative claim changed. Only captions and two figure cross-references were updated, for the
new three-panel customer figure.

**Diagnosis of the previous figures**
* **Software:** all four were matplotlib 3.10 vector PDFs, with no raster content.
* **Inconsistent sizes.** They were saved with tight cropping, so their canvases differed: the tree was 5.16 in wide,
  the β figure 6.88 in. Each was then stretched to `\textwidth`, which scaled fonts inconsistently. The tree text was
  enlarged 1.26×, the β figure shrunk 0.94×.
* **Mismatched font.** Text used DejaVu Serif, not the document's Latin Modern.

**New common style** (`report/scripts/make_report_figures.py`; still matplotlib, vector PDF):
* **Fixed 6.5 in canvas** (= `\textwidth`), saved without tight cropping, so each figure is included at exactly 1:1.
* **Type sizes:** 8 pt labels, 7.5 pt ticks and legends, against 10 pt body text.
* **Font:** text typeset by LaTeX (`text.usetex`, Latin Modern), identical to the paper's font.
* **Palette:** one color, line style and marker per method across all figures, taken from the project's original
  figures. Exact OPT is blue solid, greedy light-blue dashed, cost-aware RF green, RF + margin orange dash-dot, RF
  rounded up red dotted, 4-leaf OPT violet dashed, fixed tiers grey.
* **Axes:** light horizontal grid, no top or right spines, one frameless legend per figure placed above the panels.

**Per figure**
* **Figure 1 (OPT tree), regenerated.**
  * Smaller node text (7.5 pt).
  * Leaves show only the tier; share of VMs and mean P95 sit beneath in grey.
  * More vertical spacing.
  * All thresholds and tiers come from the saved model JSON. Leaf statistics are recomputed from saved test features
    and asserted equal to `FINAL_FINDINGS.md` §8.
* **Figure 2 (feature ablation), regenerated.**
  * Grouped bars replaced by a three-panel line chart, one panel per β.
  * A shaded band marks the customer-history step.
  * Value labels were removed because they collided with markers and duplicated the text.
* **Figure 3 (β sensitivity and risk), regenerated** in the common style.
  * Shorter legend entries; the legend is no longer clipped.
  * Alternating shaded bands with tier labels replace the overlapping band annotations.
  * β = 5 points have dark outlines.
  * The random-assignment counterfactual is a red cross with an arrow and its own legend entry.
* **Figure 4 (customers), regenerated, now three panels.** Compared with the original `ext_fig2_customer_savings.png`,
  the earlier recreation had dropped the original's most informative middle panel. It is restored as (b): median
  per-subscription saving with IQR against the VM-weighted saving, by size. Panel (a) is the ECDF and panel (c) the
  share of subscriptions made worse off.
* **Originals not reused.** The originals (`final_fig4_opt_tree.png`, `sens_fig1_*.png`, `final_fig3_frontier.png`,
  `ext_fig2_*.png`) are raster PNGs designed for standalone viewing, with long titles. When scaled to the text width
  their text falls to about 4–5 pt, and none shows exactly the report's content. Their visual language (palette,
  markers, light grid) was adopted instead.

**Checks**
* `main.pdf` compiles to 10 pages, with no overfull boxes and no undefined references.
* All figure pages were inspected at reading resolution.
* The Overleaf zip was rebuilt and test-compiled.

## Revision 3: 10-page version

The page limit was relaxed to 10 pages. The 8-page version is preserved in `report/backup_v2_8page/`.

The current `main.pdf` has **10 pages including references**:
* 10pt font, 1-inch margins;
* no overfull boxes or undefined references;
* every page inspected visually.

**What changed:**
* **Restored sentences** that had been trimmed only for length:
  * the full list of history statistics;
  * why the label rule removes long-lived VMs;
  * the DP recursion's explanation;
  * the cost-aware RF's role;
  * the β = 2 / 10 trees;
  * the equal 200k-sample greedy result;
  * Ridge ≈ RF with margins;
  * vCPU-hour savings;
  * the full diagnostic-variant description;
  * operational-risk and customer detail;
  * fuller limitations (vCPU-hour, feature selection, the 0.796 example);
  * the 15–20% summary range.
* **Restored the Zhou et al. citation** and the dataset's access note. Bibliography is back at `\small` with normal spacing.
* **New Table 2:** a compact solver cross-check (exact DP vs Gurobi MIO vs IAI local search vs greedy) in §3.1.
* **New Figure 4:** customer-level outcomes, with the per-subscription savings distribution and the % of subscriptions worse off by size. Built from saved stage-12 outputs.
* **Float placement:** the tree figure and the refinements table are anchored after the text that introduces them, so no float precedes its section heading.
* **Section 5** is unchanged and still about one page; the other sections grew.
* **Removed one inferential clause** about hot VMs from 'moderate-history' customers, because it was not a measured result.

## Revision 2 (8-page version)

The previous draft is preserved in `report/backup_v1/`: `main.tex`, `references.bib`, `main.pdf`,
`overleaf_upload.zip`, `figures/` and the old figure script.

The final `main.pdf` has **8 pages including references**, compiled with pdfLaTeX and BibTeX. It has no overfull
boxes and no undefined references. `overleaf_upload.zip` contains everything needed and was test-compiled in an empty
directory, producing 8 pages.

## 1. Rebalancing

| Section | Before | After |
|---|---|---|
| 3 Methodology | One dense OPT paragraph | New §3.1 "Exact Optimal Policy Trees" with separate paragraphs: tree class, recursion, why it is tractable, what is certified, verification |
| 4 Main results | 6 paragraphs | 8 paragraphs (see below), plus the new Figure 2 |
| 5 Refinements from the book | About 1.5 pages, a 9-column table that repeated Table 2 rows | About 495 words of prose plus a 6-column, 5-row table. It runs from the last tenth of page 5 through page 6, under one page in total. The table sits after the section heading. |
| 6 Sensitivity, complexity and robustness | — | Retitled; the complexity paragraph now explains the interpretability trade-off |

**Section 4, new paragraphs.**
* "Customer history is the dominant source of value" (with a figure) explains *why*: past VMs are a direct sample of
  the same workload distribution.
* "Why exact OPT barely beats greedy" gives two reasons: the training gain is within period drift, and one dominant
  variable family makes greedy splitting safe. Stability is noted as the benefit of exactness.
* "How the forest improves on the tree" is new evidence: the RF restricted to the same 8 features still costs 0.787
  vs 0.799, so the gap is mostly flexibility.
* "Targeting, not uniform reduction" now states what the counterfactual rules out.
* "Beyond cost" summarizes what the risk columns show.
* The same-forest A → B → C comparison (0.858 → 0.802 → 0.786) is now stated in the main results and only explained
  in §5.

**Section 5 still contains, for each refinement, everything Homework 6 Q3 asks for:**
* what was implemented;
* comparators (fixed tier, OPT, cost-aware RF);
* metrics (cost, reservation, shortfall, plus OCT agreement);
* test results (19.7% vs 20.1% vs 21.6%; 0.858 / 0.802 / 0.786);
* interpretation (similarity vs cost at high β; mean prediction vs expected-cost decision).

**Removed from §5:**
* skewness statistics;
* the K and hour grid;
* intermediate OCT depths and CART agreement;
* the per-cluster profile list;
* leaf-aggregation implementation details;
* the second calibration statistic;
* repeated intervals and the reference rows (OPT and RF are now in the caption).

## 2. Figures

Margins are unchanged at 1 inch, with a 10pt font. Figures were rebuilt only from saved outputs by
`report/scripts/make_report_figures.py`; no model was refit.

| Figure | Change |
|---|---|
| 1. OPT tree | Redrawn from the saved model JSON at print size, with node text at about 7–8pt. Thresholds are in readable units (85.8%, not 0.858). Leaf annotations are simplified to tier, share and mean P95. Leaf shares are recomputed by applying the saved tree to the saved binned test features, and match the original figure exactly. |
| 2. Feature ablation (**new third figure**) | Grouped bars for exact OPT, greedy and cost-aware RF at β = 2, 5, 10 over the three nested feature sets. It shows the jump from 6.0% to 20.3% at β = 5 when customer history is added. |
| 3. β sensitivity and risk | Reduced to five representative methods, with distinct line styles and markers and one shared legend. The validation-selected fixed tier is shown as shaded bands labeled 25 / 50 / 75 / 100% (switch points read from `s10_e1_best_fixed_switches.csv`). The random-assignment counterfactual is a red cross with its own legend entry, separate from the policy frontiers. |

We considered other candidates and did not add them:
* **The existing `fig4_grouped_importance.png`** is from the feasibility stage and shows prediction importance, not
  decision value.
* **The customer savings distribution and the forest-weighted example** would have pushed the paper past 8 pages;
  Table 4 and §6 carry the customer result.

## 3. Corrections

| Item | Change |
|---|---|
| A. "statistically indistinguishable" (clustering vs OPT) | Now "the difference from OPT is not statistically significant". The Evaluation paragraph states that an interval containing zero shows no significant difference, not equivalence. The 4-leaf comparison says it "does not establish equivalence". |
| B. "ranking identical at all 12 β" | Now "the broad ordering is stable across the grid", with the specific verified claim that the cost-aware RF is significantly cheaper than OPT at all 12 values. (RF + margin edges above OPT at β = 1.5 and 3, so a strict ranking claim would be false.) |
| C. "mostly mild shortfalls" | Replaced with measured values: about 9% of VMs exceed, by 14 pp on average, and 2–3% by more than 20 pp. |
| D. Optimality wording | "Globally optimal" no longer appears. §3.1 "What is certified" and the Limitations restrict the certificate to training cost within Π_{D,m}. |
| E. Counterfactual limitation | Stated in Assumptions ("we never observe demand or performance under a reduced reservation") and in Limitations. |
| F. References | The Interpretable AI year is removed, since no verifiable year was found. The book entry now has its ISBN. Chapter numbering is checked against the print edition's part structure; see the review notes. The second `policytree` citation (Zhou et al.) was dropped because its claim was redundant. |
| G. Redundancy | Repeated numbers removed: Table 3 no longer repeats OPT/RF rows, and the conclusion no longer restates intermediate numbers. Long lists are condensed. No em dashes. |

## 4. Layout checks (all 8 pages inspected)

* No float appears before the section that introduces it. Table 3 is at the top of page 6, after the §5 heading on
  page 5.
* No orphaned headings, clipped equations or overflowing tables. The DP recursion and the forest-weight equation each
  sit on one page with their explanation.
* Page 3 ends with a few blank lines because Table 2 is placed at the top of page 4. This is acceptable, and better
  than splitting the table from §4.

## 5. Numbers

Every number new in this revision is listed in `REPORT_SOURCE_AUDIT.md` under "Added in revision 2". No experiment
result, script or model output was modified: SHA-256 hashes of all pre-existing project outputs were re-verified
after the revision.
