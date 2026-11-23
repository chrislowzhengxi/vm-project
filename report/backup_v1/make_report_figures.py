"""Report-specific figures, built only from saved results (no model is refit, no result file is modified).

fig_opt_tree.png      : figures/final_fig4_opt_tree.png with its title strip cropped (the LaTeX caption replaces it)
fig_beta_risk.pdf     : (a) % test saving vs beta, results/sensitivity/s10_e2_saving_by_beta.csv
                        (b) mean reservation vs expected shortfall, results/extended_evaluation/ext_risk_frontier_grid.csv
                            and ext_savings_mechanism.csv (random-assignment counterfactual)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

im = Image.open(ROOT / "figures" / "final_fig4_opt_tree.png")
w, h = im.size
im.crop((0, int(0.075 * h), w, int(0.93 * h))).save(OUT / "fig_opt_tree.png", dpi=(150, 150))

plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "savefig.bbox": "tight", "font.family": "serif"})
COL = {"OPT-exact (val-selected)": ("#2a78d6", "o", "exact OPT, depth 3"),
       "Greedy (val-selected)": ("#86b6ef", "s", "greedy policy tree"),
       "Cost-aware RF": ("#1baf7a", "D", "cost-aware RF"),
       "Forest-weighted prescription": ("#008300", "v", "forest-weighted prescription"),
       "RF point + margin": ("#eb6834", "^", "RF prediction + margin"),
       "RF point -> tier": ("#e34948", "x", "RF prediction, rounded up")}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.45), gridspec_kw={"width_ratios": [1.15, 1]})
S = pd.read_csv(ROOT / "results" / "sensitivity" / "s10_e2_saving_by_beta.csv").set_index("policy")
betas = np.array([float(c) for c in S.columns])
for k, (c, m, lab) in COL.items():
    a1.plot(betas, S.loc[k].to_numpy(float), color=c, marker=m, ms=3.2, lw=1.2, label=lab)
for x, lab in ((1.60, "25%|50%"), (2.29, "50%|75%"), (4.46, "75%|100%")):
    a1.axvline(x, color="#8a8984", lw=0.7, ls=":")
    a1.text(x, -0.8, lab, rotation=90, fontsize=6.3, color="#52514e", ha="right", va="top")
a1.axhline(0, color="#52514e", lw=0.7)
a1.set_xscale("log")
a1.set_xticks([1.5, 2, 3, 5, 10, 20], ["1.5", "2", "3", "5", "10", "20"])
a1.set_ylim(-10, 24.5)
a1.set_xlabel(r"under-provisioning penalty $\beta$ (log scale)")
a1.set_ylabel("% modeled-cost saving vs best fixed tier")
a1.set_title(r"(a) Savings across $\beta$ (test)")


FR = pd.read_csv(ROOT / "results" / "extended_evaluation" / "ext_risk_frontier_grid.csv")
FRM = {"OPT depth 3": "OPT-exact (val-selected)", "Greedy depth 3": "Greedy (val-selected)", "Cost-aware RF": "Cost-aware RF",
       "RF point + margin": "RF point + margin"}
for nm, key in FRM.items():
    g = FR[FR.method == nm].sort_values("beta")
    c, m, lab = COL[key]
    a2.plot(100 * g["E[sf]"], 100 * g["E[r]"], color=c, marker=m, ms=3.0, lw=1.1)
    r5 = g[g.beta == 5].iloc[0]
    a2.plot(100 * r5["E[sf]"], 100 * r5["E[r]"], marker=m, ms=6.5, color=c, mec="black", mew=0.6, ls="none")
fx = FR[FR.method == "Fixed tier"].sort_values("tier")
a2.plot(100 * fx["E[sf]"], 100 * fx["E[r]"], color="#52514e", ls="--", lw=1.0, marker="|", ms=6)
for _, r in fx.iterrows():
    if r.tier >= 0.5:
        a2.annotate(f"always {int(100 * r.tier)}%", (100 * r["E[sf]"], 100 * r["E[r]"]), xytext=(4, 2),
                    textcoords="offset points", fontsize=6.3, color="#52514e")
M = pd.read_csv(ROOT / "results" / "extended_evaluation" / "ext_savings_mechanism.csv")
o5 = M[(M.beta == 5) & (M.policy == "OPT depth 3")].iloc[0]
opt5 = FR[(FR.method == "OPT depth 3") & (FR.beta == 5)].iloc[0]
xr, yr = o5["E[(Y-r)+] same tier mix, random assignment (pp)"], 100 * opt5["E[r]"]
a2.plot(xr, yr, marker="o", ms=6.5, mfc="white", mec="#2a78d6", mew=1.2, ls="none")
a2.annotate("", xy=(100 * opt5["E[sf]"] + 0.25, yr), xytext=(xr - 0.25, yr),
            arrowprops=dict(arrowstyle="->", color="#2a78d6", lw=0.9))
a2.text(xr, yr - 8.5, "OPT's tier mix,\nassigned at random", fontsize=6.3, color="#2a78d6", ha="center")
a2.set_xlim(-0.3, 15.5)
a2.set_ylim(38, 102)
a2.set_xlabel(r"expected shortfall $E[(Y-r)^+]$ (pp of VM capacity)")
a2.set_ylabel("mean reservation $E[r]$ (% of VM)")
a2.set_title(r"(b) Reservation vs shortfall, $\beta\in[1.5,20]$")
a2.text(9.0, 92, r"large markers: $\beta=5$", fontsize=6.3, color="#52514e")
h_, l_ = a1.get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(w_pad=1.6)
fig.savefig(OUT / "fig_beta_risk.pdf")
fig.savefig(OUT / "fig_beta_risk.png", dpi=200)
print("ok")
