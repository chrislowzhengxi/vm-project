# Report review notes

## Compliance check

| Requirement | Status |
|---|---|
| 8 pages, including references | **Met.** The compiled `main.pdf` has 8 pages; the references end about three-quarters down page 8. Format: 10pt `article` (the LaTeX default size), 1-inch margins, letter paper. |
| Due 12/01/2026 11:59 PM on Gradescope (syllabus) | The date on the title block is December 1, 2026. Slides are due separately at noon. |
| Section titled exactly "Refinements from the book" | **Met** (Section 5). |
| Both HW6 Q2(b) refinements, each with implementation, test, metrics, results and interpretation | **Met.** §5.1 covers #7, interpretable clustering (Ch. 20, §§20.1–20.2), and §5.2 covers #10, distribution-aware prescription (Ch. 13, §§13.1–13.2). Both use the comparisons, data split and metrics promised in Q2(b)(iii): best fixed, OPT, cost-aware RF; cost, reserved capacity and shortfall; OCT agreement; several β. |
| Book chapters cited | Ch. 13, 14 and 20 of Bertsimas & Dunn. |
| Authors and title | "Optimizing Cloud Virtual Machine CPU Reservations via Optimal Policy Trees", Chris Low and Emma Li. |

## Items to confirm before submission

1. **Book chapter numbering (checked in revision 2).** The print edition (Dynamic Ideas 2019, ISBN
   978-1-7337885-0-2, now in the bib entry) puts missing-data imputation and interpretable clustering in Part V and
   matrix completion and interpretable optimization in Parts VI and VII. This is consistent with the syllabus's
   chapter numbers (Ch. 19, 24, 26), which the report follows. A chapter-by-chapter table of contents was not
   available online to confirm Ch. 13 and Ch. 20 exactly.
   * **Source of the numbers:** the chapter and section numbers (Ch. 13 §§13.1–13.2, Ch. 14, Ch. 20 §§20.1–20.2) follow
     the Locus edition used in Homework 6 and the syllabus table.
   * **Possible mismatch:** the bibliography entry is the 2019 Dynamic Ideas print edition. If the printed edition
     numbers chapters differently, either cite the Locus edition or adjust the numbers.
   * **Status:** we could not check the print edition.
2. **Interpretable AI entry (resolved in revision 2).** We could not find a verifiable publication year, so the
   year was removed. The entry now cites the software and documentation, version 3.2.2 (the version used, per
   `results/final_environment.md`), with its URL. BibTeX warns about the empty year; this is harmless.
3. **Azure dataset entry.** This is a `@misc` pointing to the GitHub repository with "Accessed November 2026". The
   dataset's own README asks users to cite Cortez et al. (SOSP 2017), which the report also cites.
4. **Verified references.** The following were checked against publisher or indexing records (volume, issue, pages,
   DOI) during writing:
   * Amram et al. 2022
   * Bertsimas & Kallus 2020
   * Bertsimas & Dunn 2017
   * Bertsimas, Orfanoudaki & Wiberg 2021
   * Cortez et al. 2017 (the DOI came from the author PDF)
   * Zhou, Athey & Wager 2023
   * Sverdrup et al. 2020
   * Elmachtoub & Grigas 2022
   * Meinshausen 2006

   Breiman 2001 and Lloyd 1982 are standard entries, with DOIs from memory. They are very likely correct but were
   not re-checked.
5. **Course lecture citations from the proposal.** The proposal's lecture citations ([4], [5]) were not carried over.
   Add them if the course expects lecture citations.
6. **HW6 numbers differ from `HW6_FINDINGS.md` for rules A and B** (0.858 / 0.802 here vs 0.856 / 0.801 there). The
   report compares the three rules on one identical forest; see `REPORT_SOURCE_AUDIT.md`, discrepancy 1. If a grader
   cross-checks against the HW6 submission, the explanation is that the earlier numbers used a separately tuned forest
   per rule.

## Content trade-offs (deliberate omissions to fit 8 pages)

* **Omitted results:**
  * the H2 critical-ratio heuristic;
  * the CART-MSE trees;
  * the OPT-IAI rows (mentioned only as a solver check);
  * segment tables (vCPU, lifetime);
  * the change-point analysis (E3);
  * the strict vs feasibility setup comparison;
  * the censoring sensitivity (+0.7 points when censored VMs are excluded);
  * the illustrative forest-weight example VMs.

  All of these are in the findings documents.
* **Equal-subscription results** appear in Table 4 and §6 and are labeled exploratory. They are the main *new* caveat
  about OPT and could draw questions. The validation check, which shows the same pattern, is cited in the text.

## Interpretation risks to watch in the presentation

* **Optimality.** "Certified optimal" applies to *training* cost within the stated class only. The report says this
  explicitly in §3 and §7; keep the same wording on slides.
* **The β = 5 headline.** It sits near the peak of the savings curve. The report states the 15–20% range for
  β ∈ [3, 10] and the 19.6% hindsight-baseline figure.
* **Significance of depth 4.** Its gains are significant only at β = 2 and β = 10, and at β = 10 they come with more
  exceedance.

## Build

* **Compile:** `latexmk -pdf main.tex` in `report/`, with pdfLaTeX and BibTeX (abbrvnat).
* **Overleaf:** upload `report/overleaf_upload.zip` via New Project → Upload Project. Set the compiler to pdfLaTeX.
* **Figures:** regenerate with `python3 report/scripts/make_report_figures.py`. It reads only saved results.
