"""Report figures (matplotlib 3.10, LaTeX text rendering), built only from saved results. No model is refit and no
result file is modified.

Design rules shared by every figure
  * Canvas is exactly the LaTeX text width (6.5 in) and is saved WITHOUT tight cropping, so main.tex includes each
    figure at 1:1 and every font renders at its designed size: 8 pt labels, 7.5 pt ticks and legends (body text is 10 pt).
  * Text is typeset by LaTeX in Latin Modern, the document's font.
  * One palette and one line style per method, used identically in all figures (taken from the project's original
    figures): exact OPT blue, greedy light blue dashed, cost-aware RF green, RF + margin orange, rounded-up RF red
    dotted, 4-leaf OPT violet, fixed tiers grey.
  * Light horizontal grid, no top/right spines, one frameless legend per figure.

fig_opt_tree.pdf   saved exact OPT (results/final_models/OPT-exact_full_b5.json); leaf test shares and mean P95 from
                   applying that saved tree to the saved binned test features (data/processed/final/Xb_opt.npy).
fig_ablation.pdf   results/final_ablation.csv
fig_beta_risk.pdf  results/sensitivity/s10_e2_saving_by_beta.csv, s10_e1_best_fixed_switches.csv,
                   results/extended_evaluation/ext_risk_frontier_grid.csv, ext_savings_mechanism.csv
fig_customers.pdf  results/extended_evaluation/ext_customer_per_subscription_b5.csv, ext_customer_heterogeneity.csv
Run from the repository root:  python3 report/scripts/make_report_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import final_lib as fl  # noqa: E402

OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RES = ROOT / "results"
W = 6.5                                   # LaTeX \textwidth in inches

INK, GREY, LIGHT, GRID = "#1d1d1b", "#5f5e5a", "#9a9994", "#e3e2de"
plt.rcParams.update({
    "text.usetex": True, "font.family": "serif",
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": GREY, "ytick.color": GREY, "text.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.3, "lines.markersize": 3.6, "legend.frameon": False, "legend.handlelength": 2.2,
    "axes.titlepad": 4, "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42,
    "figure.constrained_layout.use": True, "figure.constrained_layout.h_pad": 0.03,
    "figure.constrained_layout.w_pad": 0.05,
})
# one style per method, used in every figure
STY = {
    "opt": dict(color="#2a78d6", ls="-", marker="o", label="exact OPT (depth 3)"),
    "greedy": dict(color="#86b6ef", ls=(0, (4, 1.5)), marker="s", label="greedy policy tree"),
    "rf": dict(color="#1baf7a", ls="-", marker="D", label="cost-aware RF"),
    "margin": dict(color="#eb6834", ls=(0, (5, 1.5, 1, 1.5)), marker="^", label="RF prediction + margin"),
    "round": dict(color="#e34948", ls=(0, (1, 1.2)), marker="x", label="RF prediction, rounded up"),
    "opt4": dict(color="#6250d6", ls=(0, (4, 1.5)), marker="v", label="exact OPT (4 leaves)"),
}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")          # fixed canvas, no tight crop -> 1:1 in the document
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


# ================================================================== Figure 1: the saved exact OPT tree
js = json.loads((ROOT / "data/processed/final/binner.json").read_text())
tree_js = json.loads((RES / "final_models/OPT-exact_full_b5.json").read_text())
tree = fl.tree_from_json(tree_js, type("Bn", (), {"names": js["names"]})())
F = pd.read_parquet(ROOT / "data/processed/features.parquet", columns=["split", "y_p95"])
ite = np.flatnonzero((F.split == "test").to_numpy())
Xbt = np.load(ROOT / "data/processed/final/Xb_opt.npy")[ite]
qt = F.y_p95.to_numpy()[ite] / 100
assert ite.size == 426_846

LAB = {"sub_hist_frac_le75": r"share of past VMs\\with P95 $\le$ 75\% $\le$ {}?",
       "sub_hist_frac_le50": r"share of past VMs\\with P95 $\le$ 50\% $\le$ {}?",
       "sub_hist_median": r"median past P95\\$\le$ {}?",
       "sub_hist_mean": r"mean past P95\\$\le$ {}?",
       "dep_hist_mean": r"deployment mean\\past P95 $\le$ {}?",
       "cores": r"vCPUs $\le$ {}?"}


def thr_txt(feat, thr):
    return str(int(thr)) if feat == "cores" else (f"{100 * thr:.1f}\\%" if "frac" in feat else f"{thr:.1f}\\%")


def annotate(nj, node, idx, depth, pos):
    nj["_n"], nj["_q"], nj["_d"] = len(idx), qt[idx].mean(), depth
    if "action" in nj:
        nj["_x"] = pos[0]
        pos[0] += 1
        return
    go = Xbt[idx, node["feature"]] <= node["t"]
    annotate(nj["left"], node["left"], idx[go], depth + 1, pos)
    annotate(nj["right"], node["right"], idx[~go], depth + 1, pos)
    nj["_x"] = (nj["left"]["_x"] + nj["right"]["_x"]) / 2


annotate(tree_js, tree, np.arange(len(ite)), 0, [0])
TIERC = {0.25: "#d6e6fa", 0.5: "#9cc3f0", 0.75: "#4a8ad8", 1.0: "#173f73"}
YS = {0: 0.0, 1: -1.0, 2: -2.0, 3: -3.1}
fig = plt.figure(figsize=(W, 2.75), layout="none")
ax = fig.add_axes((0, 0, 1, 1))
ax.axis("off")
ax.grid(False)


def node_box(x, y, txt):
    ax.text(x, y, r"\shortstack{" + txt + "}", ha="center", va="center", fontsize=7.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.15", fc="#f4f3ef", ec="#a8a7a2", lw=0.5))


def draw(n):
    x, y = n["_x"], YS[n["_d"]]
    if "action" in n:
        a = n["action"]
        ax.text(x, y, f"\\textbf{{{int(100 * a)}\\%}}", ha="center", va="center", fontsize=8.5,
                color="white" if a >= 0.75 else INK,
                bbox=dict(boxstyle="round,pad=0.38,rounding_size=0.15", fc=TIERC[a], ec="none"))
        ax.text(x, y - 0.42, f"{100 * n['_n'] / len(ite):.1f}\\% of VMs\nP95 {100 * n['_q']:.0f}\\%", ha="center",
                va="top", fontsize=7, color=GREY, linespacing=1.25)
        return
    for ch, lab in ((n["left"], "yes"), (n["right"], "no")):
        cx, cy = ch["_x"], YS[ch["_d"]]
        y0, y1 = y - 0.22, cy + (0.2 if "action" in ch else 0.26)
        ax.plot([x, cx], [y0, y1], color=LIGHT, lw=0.7, zorder=0)
        ax.text(x + 0.45 * (cx - x), y0 + 0.45 * (y1 - y0), lab, fontsize=6.8, color=GREY, ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=0.6))
        draw(ch)
    node_box(x, y, LAB[n["feature"]].format(thr_txt(n["feature"], n["threshold_le"])))


draw(tree_js)
ax.set_xlim(-0.55, 7.55)
ax.set_ylim(-3.95, 0.35)
save(fig, "fig_opt_tree")
leaves = []


def collect(n):
    if "action" in n:
        leaves.append((n["action"], round(100 * n["_n"] / len(ite), 1), round(100 * n["_q"])))
    else:
        collect(n["left"]); collect(n["right"])


collect(tree_js)
assert leaves == [(0.75, 19.9, 44), (1.0, 5.0, 52), (1.0, 32.9, 73), (0.75, 6.0, 59), (0.5, 7.5, 29),
                  (0.75, 8.3, 46), (0.25, 16.3, 2), (0.5, 4.3, 31)], leaves     # matches FINAL_FINDINGS.md section 8

# ================================================================== Figure 2: feature ablation (line chart)
A = pd.read_csv(RES / "final_ablation.csv")
SETS = ["A: size + time (7)", "B: A + subscription history (top 8)", "C: B + deployment history (top 8, main)"]
XL = ["size\n\\& time", "+ customer\nhistory", "+ deploy-\nment hist."]
METH = [("OPT-exact", "opt"), ("Greedy", "greedy"), ("Cost-aware RF (all set features)", "rf")]
fig, axes = plt.subplots(1, 3, figsize=(W, 1.95), sharey=True)
for ax, b in zip(axes, ("2", "5", "10")):
    ax.axvspan(0.5, 1.5, color="#f4f3ef", zorder=0, lw=0)          # the customer-history step
    for m, k in METH:
        v = [A[(A.method == m) & (A.variant == s)][b].iloc[0] for s in SETS]
        st = STY[k]
        ax.plot(range(3), v, color=st["color"], ls=st["ls"], marker=st["marker"], label=st["label"], zorder=3,
                ms=4)
    ax.set_xticks(range(3), XL)
    ax.set_xlim(-0.35, 2.35)
    ax.set_title(f"$\\beta={b}$")
    ax.set_ylim(-1, 24)
axes[0].set_ylabel("\\% saving vs best fixed")
axes[0].set_yticks([0, 5, 10, 15, 20])
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, ["exact OPT", "greedy policy tree", "cost-aware RF"], loc="outside upper center", ncol=3)
save(fig, "fig_ablation")

# ================================================================== Figure 3: beta sensitivity and reservation vs shortfall
S = pd.read_csv(RES / "sensitivity/s10_e2_saving_by_beta.csv").set_index("policy")
SW = pd.read_csv(RES / "sensitivity/s10_e1_best_fixed_switches.csv")
sw = SW[SW.split.str.startswith("val") & ~SW["best fixed tier"].str.startswith("pairwise")].sort_values(
    "beta from")["beta to"].to_numpy()[:3]
assert np.allclose(np.round(sw, 2), [1.60, 2.29, 4.46])
betas = np.array([float(c) for c in S.columns])
SHOW = [("OPT-exact (val-selected)", "opt"), ("Greedy (val-selected)", "greedy"), ("Cost-aware RF", "rf"),
        ("RF point + margin", "margin"), ("RF point -> tier", "round")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.65), gridspec_kw={"width_ratios": [1.1, 1]})
edges = [1.35, *sw, 23]
for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
    if k % 2:
        a1.axvspan(lo, hi, color="#f1f0ec", zorder=0, lw=0)
    a1.text(np.sqrt(lo * hi), 25.4, f"{[25, 50, 75, 100][k]}\\%", ha="center", va="top", fontsize=6.8, color=GREY)
for key, k in SHOW:
    st = STY[k]
    a1.plot(betas, S.loc[key].to_numpy(float), color=st["color"], ls=st["ls"], marker=st["marker"], label=st["label"],
            zorder=3)
a1.axhline(0, color=GREY, lw=0.6)
a1.set_xscale("log")
a1.set_xlim(1.35, 23)
a1.set_xticks([1.5, 2, 3, 5, 10, 20], ["1.5", "2", "3", "5", "10", "20"])
a1.minorticks_off()
a1.set_ylim(-8, 26)
a1.set_xlabel("penalty $\\beta$ (log scale)")
a1.set_ylabel("\\% saving vs best fixed tier")
a1.set_title("(a) Saving across $\\beta$")

FR = pd.read_csv(RES / "extended_evaluation/ext_risk_frontier_grid.csv")
for nm, k in (("OPT depth 3", "opt"), ("Greedy depth 3", "greedy"), ("Cost-aware RF", "rf"),
              ("RF point + margin", "margin")):
    st = STY[k]
    g = FR[FR.method == nm].sort_values("beta")
    a2.plot(100 * g["E[sf]"], 100 * g["E[r]"], color=st["color"], ls=st["ls"], marker=st["marker"], ms=3, zorder=3)
    r5 = g[g.beta == 5].iloc[0]
    a2.plot(100 * r5["E[sf]"], 100 * r5["E[r]"], marker=st["marker"], ms=6, color=st["color"], mec=INK, mew=0.6,
            ls="none", zorder=4)
fx = FR[FR.method == "Fixed tier"].sort_values("tier")
a2.plot(100 * fx["E[sf]"], 100 * fx["E[r]"], color=GREY, ls=(0, (3, 2)), lw=0.9, marker="s", ms=3, mfc="white",
        zorder=2, label="fixed tiers and their mixtures")
for _, r in fx.iterrows():
    if r.tier >= 0.5:
        a2.annotate(f"{int(100 * r.tier)}\\%", (100 * r["E[sf]"], 100 * r["E[r]"]), xytext=(4, 2),
                    textcoords="offset points", fontsize=7, color=GREY)
M = pd.read_csv(RES / "extended_evaluation/ext_savings_mechanism.csv")
o5 = M[(M.beta == 5) & (M.policy == "OPT depth 3")].iloc[0]
opt5 = FR[(FR.method == "OPT depth 3") & (FR.beta == 5)].iloc[0]
xr, yr = o5["E[(Y-r)+] same tier mix, random assignment (pp)"], 100 * opt5["E[r]"]
a2.annotate("", xy=(100 * opt5["E[sf]"] + 0.35, yr), xytext=(xr - 0.35, yr),
            arrowprops=dict(arrowstyle="-|>", color="#e34948", lw=0.8, mutation_scale=7), zorder=4)
a2.plot(xr, yr, marker="X", ms=6.5, color="#e34948", mec="white", mew=0.4, ls="none", zorder=5,
        label="OPT's $\\beta=5$ tiers at random")
a2.set_xlim(-0.4, 15.5)
a2.set_ylim(38, 103)
a2.set_xlabel("expected shortfall $E[(Y-r)^+]$ (pp)")
a2.set_ylabel("mean reservation $E[r]$ (\\%)")
a2.set_title("(b) Reservation vs shortfall")
a2.legend(loc="upper right", fontsize=7, handlelength=2.0)
h, l = a1.get_legend_handles_labels()
fig.legend(h, ["exact OPT", "greedy tree", "cost-aware RF", "RF + margin", "RF rounded up"], loc="outside upper center", ncol=5, columnspacing=1.6, handlelength=2.4)
save(fig, "fig_beta_risk")

# ================================================================== Figure 4: customer-level outcomes (beta = 5)
P = pd.read_csv(RES / "extended_evaluation/ext_customer_per_subscription_b5.csv")
CU = pd.read_csv(RES / "extended_evaluation/ext_customer_heterogeneity.csv")
CP = [("OPT depth 3", "opt", "exact OPT (8 leaves)"), ("OPT 4-leaf (depth 2)", "opt4", "exact OPT (4 leaves)"),
      ("Cost-aware RF", "rf", "cost-aware RF")]
STR = ["5-19", "20-99", "100-999", ">= 1000"]
STRL = ["5--19", "20--99", "100--\n999", "$\\ge$1,000"]
fig, (b1, b2, b3) = plt.subplots(1, 3, figsize=(W, 2.3), gridspec_kw={"width_ratios": [1.25, 1, 1]})
for key, k, lab in CP:
    st = STY[k]
    v = np.sort(P[(P.policy == key) & (P["test VMs"] >= 5)]["% saving"].to_numpy())
    b1.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=st["color"], ls=st["ls"], lw=1.2, label=lab)
b1.axvline(0, color=GREY, lw=0.6)
b1.axvline(20.1, color=STY["opt"]["color"], lw=0.6, ls=(0, (1, 1.5)))
b1.text(21.5, 0.08, "OPT aggregate\n(VM-weighted)", fontsize=6.8, color=STY["opt"]["color"], va="bottom")
b1.set_xlim(-40, 80)
b1.set_ylim(0, 1.02)
b1.set_xlabel("\\% saving per subscription")
b1.set_ylabel("cumulative share of subscriptions")
b1.set_title("(a) Distribution")
b1.grid(axis="x")
xs = np.arange(len(STR))
off = {"opt": -0.22, "opt4": 0.0, "rf": 0.22}
for key, k, lab in CP:
    st = STY[k]
    d = CU[(CU.beta == 5) & (CU.policy == key)].set_index("subscriptions").loc[STR]
    x = xs + off[k]
    b2.errorbar(x, d["median % saving"], yerr=[d["median % saving"] - d.P25, d.P75 - d["median % saving"]], fmt="o",
                ms=3.6, color=st["color"], lw=0.9, capsize=1.8, capthick=0.8)
    b2.plot(x, d["VM-weighted % saving"], ls="none", marker="D", ms=3.6, mfc="white", mec=st["color"], mew=0.9)
    b3.bar(x, d["% subs higher cost"], width=0.21, color=st["color"])
b2.plot([], [], "ko", ms=3.6, label="median [IQR]")
b2.plot([], [], ls="none", marker="D", ms=3.6, mfc="white", mec="k", label="VM-weighted")
b2.legend(loc="upper right", fontsize=6.8, handletextpad=0.3, borderaxespad=0.2)
b2.set_ylim(-3, 50)
b2.set_ylabel("\\% saving")
b2.set_title("(b) Saving by subscription size")
b3.set_ylabel("\\% of subscriptions")
b3.set_title("(c) Costlier than always-100\\%")
for ax in (b2, b3):
    ax.set_xticks(xs, STRL)
    ax.set_xlabel("test VMs per subscription")
    ax.set_xlim(-0.55, 3.55)
h, l = b1.get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=3)
save(fig, "fig_customers")
print("figures ok; leaves verified:", leaves)
