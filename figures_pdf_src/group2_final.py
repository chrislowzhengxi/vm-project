"""Vector, report-style redraws of figures/final_fig1-6 from saved outputs only (no model is trained or re-solved)."""
import pickle
import shutil

import numpy as np
import pandas as pd

from pubstyle import GREY, INK, PAL, PROC, RES, ROOT, STY, W, plt, save
import final_lib as fl

MT = pd.read_csv(RES / "final_main_table.csv")

# ---------------------------------------------------------------- fig1: benchmark at beta = 5
FAMC = {"fixed": GREY, "heuristic": PAL["violet"], "predict-then-decide": PAL["orange"], "cost-aware RF": PAL["green"],
        "tree (greedy)": PAL["lightblue"], "OPT": PAL["blue"], "oracle": INK}
SHOW = [("Best fixed (val)", "best fixed tier (always 100\\%)"), ("H2: customer critical-ratio rule", "H2: customer critical-ratio rule"),
        ("H1: customer mean P95 + margin", "H1: customer mean P95 + margin"), ("RF predict -> tier", "RF prediction, rounded up"),
        ("Ridge predict + margin", "Ridge prediction + margin"), ("RF predict + margin", "RF prediction + margin"),
        ("CART-MSE (26 feat.) [full]", "CART-MSE tree (26 features)"), ("CART-MSE [full]", "CART-MSE tree (8 features)"),
        ("Greedy [full]", "greedy policy tree"), ("OPT-IAI [200k]", "OPT, IAI local search (200k)"),
        ("OPT-exact [full]", "exact OPT"), ("Cost-aware RF", "cost-aware RF"), ("Oracle (hindsight)", "oracle (hindsight)")]
d5 = MT[MT.beta == 5].set_index("policy")
fig, ax = plt.subplots(figsize=(W, 3.5))
yy = np.arange(len(SHOW))[::-1]
for y, (p, lab) in zip(yy, SHOW):
    r = d5.loc[p]
    ax.barh(y, r["test cost"], color=FAMC[r.family], height=0.64)
    txt = f"{r['test cost']:.3f}" + ("" if p == "Best fixed (val)" else f"  ({r['% reduction vs best fixed']:.1f}\\% saving)")
    ax.text(r["test cost"] + 0.008, y, txt, va="center", fontsize=7.2, color=INK)
ax.set_yticks(yy, [s[1] for s in SHOW])
ax.set_xlim(0, 1.27)
ax.set_xlabel("mean modeled cost per test VM, $\\beta=5$ (lower is better)")
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
fams = ["fixed", "heuristic", "predict-then-decide", "tree (greedy)", "OPT", "cost-aware RF", "oracle"]
fig.legend([plt.Rectangle((0, 0), 1, 1, color=FAMC[f]) for f in fams], fams, loc="outside upper center", ncol=7,
           fontsize=7, handlelength=1.0, columnspacing=1.0)
save(fig, "final_fig1_benchmark_b5")

# ---------------------------------------------------------------- fig2: savings with cluster-bootstrap CIs across beta
BT = pd.read_csv(RES / "final_bootstrap.csv")
FP = [("H1: customer mean P95 + margin", "H1 customer rule", "h1"), ("RF predict + margin", "RF prediction + margin", "margin"),
      ("CART-MSE (26 feat.) [full]", "CART-MSE (26 feat.)", "cart"), ("Greedy [full]", "greedy policy tree", "greedy"),
      ("OPT-IAI [200k]", "OPT, IAI search", "iai"), ("OPT-exact [full]", "exact OPT", "opt"), ("Cost-aware RF", "cost-aware RF", "rf")]
fig, ax = plt.subplots(figsize=(W, 2.6))
w = 0.8 / len(FP)
for j, (p, lab, k) in enumerate(FP):
    for i, b in enumerate(fl.BETAS):
        r = BT[(BT.beta == b) & (BT["policy A"] == p) & (BT["policy B"] == "Best fixed (val)")].iloc[0]
        mid, lo, hi = -r["% of B"], -100 * r["95% CI high"] / r["cost B"], -100 * r["95% CI low"] / r["cost B"]
        x = i + (j - (len(FP) - 1) / 2) * w
        ax.bar(x, mid, width=w * 0.88, color=STY[k]["color"], label=lab if i == 0 else None)
        ax.errorbar(x, mid, yerr=[[mid - lo], [hi - mid]], color=INK, lw=0.7, capsize=1.5, capthick=0.6)
ax.set_xticks(range(3), [f"$\\beta={b}$" for b in fl.BETAS])
ax.set_ylabel("\\% saving vs best fixed tier\n(95\\% subscription-cluster CI)")
ax.grid(axis="x", visible=False)
fig.legend(loc="outside upper center", ncol=4, fontsize=7, handlelength=1.0, columnspacing=1.2)
save(fig, "final_fig2_beta_ci")

# ---------------------------------------------------------------- fig3: capacity vs under-provisioning frontier
F = pd.read_parquet(ROOT / "data/processed/features.parquet", columns=["split", "y_p95", "sub_hist_mean"])
ite = np.flatnonzero((F.split == "test").to_numpy())
qt = F.y_p95.to_numpy()[ite] / 100
npz = np.load(PROC / "final/rf_cost_shortfall_preds.npz")
assert np.array_equal(npz["test_idx"], ite)


def ev(a):
    return 100 * np.mean(qt > a + 1e-12), np.mean(a)


fig, ax = plt.subplots(figsize=(W, 3.4))
fx = [ev(np.full(len(qt), t)) for t in fl.TIERS]
ax.plot([f[0] for f in fx], [f[1] for f in fx], color=GREY, ls=(0, (3, 2)), lw=0.9, marker="s", ms=4, mfc="white",
        label="fixed tiers (and their mixtures)", zorder=2)
for t, f in zip(fl.TIERS, fx):
    ax.annotate(f"always {int(t * 100)}\\%", f, xytext=(5, 3) if t > 0.25 else (-50, 3), textcoords="offset points",
                fontsize=7, color=GREY)
Sh = npz["test"]
rf = [ev(fl.TIERS[np.argmin(np.column_stack([fl.TIERS[:3] + b * Sh, np.ones(len(Sh))]), axis=1)])
      for b in np.geomspace(1.1, 60, 30)]
ax.plot([e[0] for e in rf], [e[1] for e in rf], color=STY["rf"]["color"], lw=1.4, label="cost-aware RF ($\\beta$ sweep)",
        zorder=3)
E2P = pickle.load(open(PROC / "sensitivity/e2_policies.pkl", "rb"))["POLT_tier_index"]
grid = sorted({k[1] for k in E2P})
for key, k, lab in (("Greedy (val-selected)", "greedy", "greedy policy tree, depth $\\le$3"),
                    ("OPT-exact (val-selected)", "opt", "exact OPT, depth $\\le$3")):
    e = [ev(fl.TIERS[E2P[(key, g)]]) for g in grid]
    ax.plot([x[0] for x in e], [x[1] for x in e], color=STY[k]["color"], ls=STY[k]["ls"], marker=STY[k]["marker"],
            ms=3.6, lw=1.2, label=lab + " (12 validated $\\beta$)", zorder=4)
    if k == "opt":
        e5 = e[grid.index(5.0)]
        ax.annotate("$\\beta=5$", e5, xytext=(-6, -14), textcoords="offset points", fontsize=7, color=INK)
mt = F.sub_hist_mean.to_numpy()[ite]
h1 = [ev(fl.route(mt, d)) for d in np.arange(-30, 80, 2.5)]
ax.plot([e[0] for e in h1], [e[1] for e in h1], color=STY["h1"]["color"], lw=1.1, label="H1 customer mean + margin (margin sweep)",
        zorder=3)
o = ev(fl.TIERS[np.searchsorted(fl.TIERS, qt - 1e-12)])
ax.plot(*o, marker="*", ms=9, color=INK, ls="none", label="oracle (smallest sufficient tier)", zorder=5)
ax.set_xlabel("\\% of test VMs under-provisioned (P95 $>$ reservation)")
ax.set_ylabel("mean reserved capacity (fraction of VM)")
ax.grid(axis="x", visible=True)
ax.legend(loc="upper right", fontsize=7.2)
save(fig, "final_fig3_frontier")

# ---------------------------------------------------------------- fig4: identical saved tree, report design
shutil.copy(ROOT / "report/figures/fig_opt_tree.pdf", ROOT / "figures_pdf/final_fig4_opt_tree.pdf")
(ROOT / "figures_pdf/final_fig4_opt_tree_white.png").unlink(missing_ok=True)

# ---------------------------------------------------------------- fig5: sample size / stability and runtime
ST = pd.read_csv(RES / "final_stability.csv")
MIO = pd.read_csv(RES / "final_mio_crosscheck.csv")
MK = {"OPT-exact": ("opt", "exact OPT (DP)"), "Greedy": ("greedy", "greedy tree"), "OPT-IAI": ("iai", "OPT, IAI local search")}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.6))
for m, (k, lab) in MK.items():
    g = ST[ST.method == m].sort_values("n")
    st = STY[k]
    a1.errorbar(g.n, g.test_cost_mean, yerr=[g.test_cost_mean - g.test_cost_min, g.test_cost_max - g.test_cost_mean],
                color=st["color"], marker=st["marker"], ms=3.6, lw=1.2, ls=st["ls"], capsize=2, label=lab)
    a2.plot(g.n, g.runtime_s, color=st["color"], marker=st["marker"], ms=3.6, lw=1.2, ls=st["ls"], label=lab)
mm = MIO[(MIO.method == "Gurobi MIO")]
for d_, c in ((2, PAL["magenta"]), (3, PAL["red"])):
    g = mm[mm.depth == d_].sort_values("n")
    a2.plot(g.n, g.runtime_s, color=c, marker="P", ms=4, lw=1.0, ls=(0, (3, 1.5)),
            label=f"Gurobi MIO, depth {d_} ({'proved optimal' if d_ == 2 else 'hit 600 s limit'})")
for ax in (a1, a2):
    ax.set_xscale("log")
    ax.set_xlabel("training VMs (log scale)")
    ax.grid(axis="x", visible=False)
a1.set_ylabel("test cost, $\\beta=5$")
a1.set_title("(a) Test cost (mean, min--max over 5 samples)")
a2.set_yscale("log")
a2.set_ylabel("fit time per tree (s, log scale)")
a2.set_title("(b) Runtime: exact DP vs local search vs MIO")
h1_, l1_ = a2.get_legend_handles_labels()
fig.legend(h1_, l1_, loc="outside upper center", ncol=3, fontsize=7, handlelength=1.8, columnspacing=1.2)
save(fig, "final_fig5_opt_scaling")

# ---------------------------------------------------------------- fig6: segment-level savings at beta = 5
SG = pd.read_csv(RES / "final_segments.csv")
s5 = SG[SG.beta == 5].set_index("segment")
fig, ax = plt.subplots(figsize=(W, 3.5))
yy = np.arange(len(s5))[::-1]
ax.barh(yy + 0.19, s5["OPT % saving"], height=0.36, color=STY["opt"]["color"], label="exact OPT (depth 3)")
ax.barh(yy - 0.19, s5["RF % saving"], height=0.36, color=STY["rf"]["color"], label="cost-aware RF")
ax.axvline(0, color=GREY, lw=0.6)
lab = [f"{s}  ($n$ = {int(s5.loc[s, 'n']):,})".replace("<=", "$\\le$").replace(">=", "$\\ge$").replace(">", "$>$")
       .replace("%", "\\%") for s in s5.index]
ax.set_yticks(yy, lab, fontsize=7)
ax.set_xlabel("\\% saving vs best fixed tier within segment ($\\beta=5$, test)")
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
ax.legend(loc="lower right", fontsize=7.2)
save(fig, "final_fig6_segments")
print("group2 done")
