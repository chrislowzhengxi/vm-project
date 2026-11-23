"""Report-specific figures, built only from saved results (no model is refit, no result file is modified).

fig_opt_tree.pdf   : the saved exact OPT (results/final_models/OPT-exact_full_b5.json) redrawn at print size.
                     Leaf shares and mean realized P95 are computed by applying the SAVED tree to the saved binned
                     test features (data/processed/final/Xb_opt.npy); nothing is refit.
fig_ablation.pdf   : results/final_ablation.csv (nested feature sets, % saving vs best fixed, test).
fig_beta_risk.pdf  : (a) results/sensitivity/s10_e2_saving_by_beta.csv and s10_e1_best_fixed_switches.csv
                     (b) results/extended_evaluation/ext_risk_frontier_grid.csv and ext_savings_mechanism.csv
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

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
BLUE, LBLUE, AQUA, ORANGE, RED, GREY, INK = "#2a78d6", "#7fb0ea", "#1baf7a", "#eb6834", "#d03b3b", "#6d6c68", "#1d1d1b"

# ============================================================== Figure 1: the saved exact OPT tree
js = json.loads((ROOT / "data/processed/final/binner.json").read_text())
names = js["names"]
tree_js = json.loads((RES / "final_models/OPT-exact_full_b5.json").read_text())
tree = fl.tree_from_json(tree_js, type("Bn", (), {"names": names})())
F = pd.read_parquet(ROOT / "data/processed/features.parquet", columns=["split", "y_p95"])
ite = np.flatnonzero((F.split == "test").to_numpy())
Xbt = np.load(ROOT / "data/processed/final/Xb_opt.npy")[ite]
qt = F.y_p95.to_numpy()[ite] / 100
assert ite.size == 426_846

LAB = {"sub_hist_frac_le75": ("share of past VMs", "at P95$\\leq$75%"),
       "sub_hist_frac_le50": ("share of past VMs", "at P95$\\leq$50%"),
       "sub_hist_median": ("median", "past P95"),
       "sub_hist_mean": ("mean", "past P95"),
       "dep_hist_mean": ("deployment mean", "past P95"),
       "cores": ("vCPUs", "")}


def fmt_thr(feat, thr):
    if feat.startswith("sub_hist_frac"):
        return f"{100 * thr:.1f}%"
    if feat == "cores":
        return f"{int(thr)}"
    return f"{thr:.1f}%"


def annotate(node_js, node, idx, depth, pos):
    node_js["_n"], node_js["_q"], node_js["_d"] = len(idx), qt[idx].mean(), depth
    if "action" in node_js:
        node_js["_x"] = pos[0]
        pos[0] += 1
        return
    go = Xbt[idx, node["feature"]] <= node["t"]
    annotate(node_js["left"], node["left"], idx[go], depth + 1, pos)
    annotate(node_js["right"], node["right"], idx[~go], depth + 1, pos)
    node_js["_x"] = (node_js["left"]["_x"] + node_js["right"]["_x"]) / 2


annotate(tree_js, tree, np.arange(len(ite)), 0, [0])
TIERC = {0.25: "#cfe1f8", 0.5: "#8fbbef", 0.75: "#3a7fd5", 1.0: "#123f78"}
fig, ax = plt.subplots(figsize=(6.6, 2.6))
ax.axis("off")
YS = {0: 0.0, 1: -0.95, 2: -1.9, 3: -2.9}


def draw(n):
    x, y = n["_x"], YS[n["_d"]]
    if "action" in n:
        a = n["action"]
        dark = a >= 0.75
        ax.text(x, y, f"reserve\n{int(100 * a)}%", ha="center", va="center", fontsize=8.2, fontweight="bold",
                color="white" if dark else INK, linespacing=1.0,
                bbox=dict(boxstyle="round,pad=0.35", fc=TIERC[a], ec="none"))
        ax.text(x, y - 0.5, f"{100 * n['_n'] / len(ite):.1f}%\nP95 {100 * n['_q']:.0f}%",
                ha="center", va="center", fontsize=6.9, color=INK, linespacing=1.1)
        return
        ax.text(x, y, "",
                ha="center", va="center", fontsize=7.4, color="white" if dark else INK, linespacing=1.15,
                bbox=dict(boxstyle="round,pad=0.35", fc=TIERC[a], ec="none"))
        return
    for ch, lab in ((n["left"], "yes"), (n["right"], "no")):
        cx, cy = ch["_x"], YS[ch["_d"]]
        ax.plot([x, cx], [y - 0.2, cy + 0.27], color=GREY, lw=0.8, zorder=0)
        ax.text(x + 0.42 * (cx - x), y - 0.2 + 0.42 * (cy + 0.27 - y + 0.2), lab, fontsize=6.8, color=GREY,
                ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.4))
        draw(ch)
    l1, l2 = LAB[n["feature"]]
    q = f"{l1}\n{l2} $\\leq$ {fmt_thr(n['feature'], n['threshold_le'])}?" if l2 else \
        f"{l1} $\\leq$ {fmt_thr(n['feature'], n['threshold_le'])}?"
    ax.text(x, y, q, ha="center", va="center", fontsize=7.1, color=INK, linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f3f2ee", ec="#8d8c88", lw=0.6))


draw(tree_js)
ax.set_xlim(-0.5, 7.5)
ax.set_ylim(-3.6, 0.3)
fig.savefig(OUT / "fig_opt_tree.pdf")
fig.savefig(OUT / "fig_opt_tree.png", dpi=200)
plt.close(fig)
leaves = []


def collect(n):
    if "action" in n:
        leaves.append((n["action"], round(100 * n["_n"] / len(ite), 1), round(100 * n["_q"])))
    else:
        collect(n["left"]); collect(n["right"])


collect(tree_js)
print("OPT leaves (tier, % test VMs, mean P95):", leaves)

# ============================================================== Figure 2: feature ablation
A = pd.read_csv(RES / "final_ablation.csv")
SETS = ["A: size + time (7)", "B: A + subscription history (top 8)", "C: B + deployment history (top 8, main)"]
SETLAB = ["size,\ntime", "+cust.\nhistory", "+depl.\nhistory"]
METH = [("OPT-exact", "exact OPT", BLUE), ("Greedy", "greedy tree", LBLUE),
        ("Cost-aware RF (all set features)", "cost-aware RF", AQUA)]
fig, axes = plt.subplots(1, 3, figsize=(6.6, 1.45), sharey=True)
for ax, b in zip(axes, ("2", "5", "10")):
    w = 0.26
    for j, (m, lab, c) in enumerate(METH):
        v = [A[(A.method == m) & (A.variant == s)][b].iloc[0] for s in SETS]
        bars = ax.bar(np.arange(3) + (j - 1) * w, v, width=w * 0.92, color=c, label=lab)
        if m == "OPT-exact":
            for xx, vv in zip(np.arange(3) + (j - 1) * w, v):
                ax.text(xx, vv + 0.5, f"{vv:.1f}", ha="center", va="bottom", fontsize=6.2, color=INK)
    ax.set_xticks(range(3), SETLAB, fontsize=6.8)
    ax.set_title(f"$\\beta={b}$")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 24)
axes[0].set_ylabel("% saving vs best fixed")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 1.07))
fig.tight_layout(w_pad=0.6)
fig.savefig(OUT / "fig_ablation.pdf")
fig.savefig(OUT / "fig_ablation.png", dpi=200)
plt.close(fig)

# ============================================================== Figure 3: beta sensitivity and reservation vs shortfall
S = pd.read_csv(RES / "sensitivity/s10_e2_saving_by_beta.csv").set_index("policy")
SW = pd.read_csv(RES / "sensitivity/s10_e1_best_fixed_switches.csv")
betas = np.array([float(c) for c in S.columns])
SHOW = [("OPT-exact (val-selected)", "exact OPT (depth 3)", BLUE, "o", "-"),
        ("Greedy (val-selected)", "greedy policy tree", LBLUE, "s", "--"),
        ("Cost-aware RF", "cost-aware RF", AQUA, "D", "-"),
        ("RF point + margin", "RF prediction + margin", ORANGE, "^", "-."),
        ("RF point -> tier", "RF prediction, rounded up", RED, "x", ":")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.4), gridspec_kw={"width_ratios": [1.12, 1]})
sw = SW[SW.split.str.startswith("val") & ~SW["best fixed tier"].str.startswith("pairwise")].sort_values("beta from")["beta to"].to_numpy()[:3]   # validation switches
assert np.allclose(np.round(sw, 2), [1.60, 2.29, 4.46])
edges = [1.35, *sw, 23]
for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
    a1.axvspan(lo, hi, color=["#f7f7f5", "#efeeea"][k % 2], zorder=0, lw=0)
    a1.text(np.sqrt(lo * hi), 23.3, f"{[25, 50, 75, 100][k]}%", ha="center", va="top", fontsize=6.3, color=GREY)
for key, lab, c, m, ls in SHOW:
    a1.plot(betas, S.loc[key].to_numpy(float), color=c, marker=m, ms=3.0, lw=1.2, ls=ls, label=lab, zorder=3)
a1.axhline(0, color=GREY, lw=0.6)
a1.set_xscale("log")
a1.set_xlim(1.35, 23)
a1.set_xticks([1.5, 2, 3, 5, 10, 20], ["1.5", "2", "3", "5", "10", "20"])
a1.minorticks_off()
a1.set_ylim(-8, 24)
a1.set_xlabel(r"penalty $\beta$ (log scale)")
a1.set_ylabel("% modeled-cost saving vs\nvalidation-selected fixed tier")
a1.set_title(r"(a) Saving across $\beta$ (test)")
a1.grid(axis="x", visible=False)

FR = pd.read_csv(RES / "extended_evaluation/ext_risk_frontier_grid.csv")
for nm, key in (("OPT depth 3", "OPT-exact (val-selected)"), ("Greedy depth 3", "Greedy (val-selected)"),
                ("Cost-aware RF", "Cost-aware RF"), ("RF point + margin", "RF point + margin")):
    _, lab, c, m, ls = next(s for s in SHOW if s[0] == key)
    g = FR[FR.method == nm].sort_values("beta")
    a2.plot(100 * g["E[sf]"], 100 * g["E[r]"], color=c, marker=m, ms=2.6, lw=1.1, ls=ls, zorder=3)
    r5 = g[g.beta == 5].iloc[0]
    a2.plot(100 * r5["E[sf]"], 100 * r5["E[r]"], marker=m, ms=5.5, color=c, mec=INK, mew=0.6, ls="none", zorder=4)
fx = FR[FR.method == "Fixed tier"].sort_values("tier")
a2.plot(100 * fx["E[sf]"], 100 * fx["E[r]"], color=GREY, ls="--", lw=0.9, marker="|", ms=5, zorder=2,
        label="fixed tiers and their\nrandom mixtures")
for _, r in fx.iterrows():
    if r.tier >= 0.5:
        a2.annotate(f"{int(100 * r.tier)}%", (100 * r["E[sf]"], 100 * r["E[r]"]), xytext=(3, 3),
                    textcoords="offset points", fontsize=6.3, color=GREY)
M = pd.read_csv(RES / "extended_evaluation/ext_savings_mechanism.csv")
o5 = M[(M.beta == 5) & (M.policy == "OPT depth 3")].iloc[0]
opt5 = FR[(FR.method == "OPT depth 3") & (FR.beta == 5)].iloc[0]
xr, yr = o5["E[(Y-r)+] same tier mix, random assignment (pp)"], 100 * opt5["E[r]"]
a2.plot(xr, yr, marker="X", ms=7, color=RED, mec="white", mew=0.5, ls="none", zorder=5,
        label="OPT's $\\beta=5$ tiers,\nassigned at random")
a2.annotate("", xy=(100 * opt5["E[sf]"] + 0.3, yr), xytext=(xr - 0.3, yr),
            arrowprops=dict(arrowstyle="->", color=RED, lw=0.8), zorder=4)
a2.set_xlim(-0.3, 15.5)
a2.set_ylim(38, 102)
a2.set_xlabel(r"expected shortfall $E[(Y-r)^+]$ (pp of capacity)")
a2.set_ylabel("mean reservation $E[r]$ (%)")
a2.set_title(r"(b) Reservation vs shortfall, $\beta$ = 1.5 to 20")
a2.legend(frameon=False, loc="upper right", fontsize=6.4, handlelength=1.4)
a2.text(0.6, 41, "outlined markers: $\\beta=5$", ha="left", fontsize=6.2, color=GREY)
h_, l_ = a1.get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=5, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.07),
           handlelength=2.2, columnspacing=1.2)
fig.tight_layout(w_pad=1.2, rect=(0, 0.06, 1, 1))
fig.savefig(OUT / "fig_beta_risk.pdf")
fig.savefig(OUT / "fig_beta_risk.png", dpi=200)
plt.close(fig)
print("ok")
