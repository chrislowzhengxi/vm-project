"""Vector PDF versions of the feasibility-study figures (originals: figures/fig1..fig8, drawn by src/01, 03, 04).
Built only from saved outputs; nothing is refit. Run from the repo root:
    /usr/local/bin/python3 figures_pdf_src/group1_feasibility.py
Skipped: fig5_depth3_tree and fig8_policy_tree_lambda5 (tree structures were never saved; only leaf summaries exist).
fig7: the 'direct cost-aware RF' sweep is omitted (its shortfall predictions were never saved).
"""
import numpy as np
import pandas as pd

from pubstyle import GREY, INK, PAL, PROC, RES, STY, W, plt, save
import final_lib as fl  # noqa: E402  (pubstyle puts src/ on the path)

TIERS = np.array([0.25, 0.5, 0.75, 1.0])

# ---------------------------------------------------------------- fig1: target distributions (new VMs in trace)
vm = pd.read_parquet(PROC / "vmtable.parquet", columns=["subscriptionid", "deploymentid", "vmcreated",
                                                         "p95maxcpu", "avgcpu", "maxcpu"])
newv = vm[vm.vmcreated > 0]
fig, axes = plt.subplots(1, 3, figsize=(W, 2.0), sharey=True)
bins = np.linspace(0, 100, 51)
for ax, col, c, t in zip(axes, ["p95maxcpu", "avgcpu", "maxcpu"], [PAL["blue"], PAL["orange"], PAL["green"]],
                         ["(a) P95 CPU (target)", "(b) average CPU", "(c) maximum CPU"]):
    ax.hist(newv[col], bins=bins, weights=np.full(len(newv), 100 / len(newv)), color=c, edgecolor="white", lw=0.3)
    for v in (25, 50, 75):
        ax.axvline(v, color=GREY, lw=0.6, ls=(0, (1, 1.5)))
    ax.set_title(t)
    ax.set_xlabel("CPU utilization (\\% of vCPUs)")
    ax.set_xticks([0, 25, 50, 75, 100])
axes[0].set_ylabel("\\% of VMs (2-pt bins)")
save(fig, "fig1_target_distributions")

# ---------------------------------------------------------------- fig2: VMs per subscription / deployment (CCDF)
vps = vm.groupby("subscriptionid", observed=True).size()
vpd = vm.groupby("deploymentid", observed=True).size()
fig, ax = plt.subplots(figsize=(3.2, 2.3))
for s, lab, c in [(vps, "subscriptions", PAL["blue"]), (vpd, "deployments", PAL["orange"])]:
    x = np.sort(s.values)
    ax.loglog(x, 1 - np.arange(len(x)) / len(x), color=c, drawstyle="steps-post", label=f"{lab} (n = {len(s):,})")
ax.set_xlabel("VMs in group (log scale)")
ax.set_ylabel("share of groups with $\\ge x$ VMs")
ax.grid(axis="both", which="major")
ax.legend(loc="lower left", fontsize=7)
save(fig, "fig2_vms_per_group")

# ---------------------------------------------------------------- fig3: creations over time with split boundaries
hc = (newv.vmcreated // 3600).value_counts().sort_index()
fig, ax = plt.subplots(figsize=(W, 1.8))
ax.plot(hc.index / 24, hc.values / 1000, color=PAL["blue"], lw=0.8)
for d, lab, dx, ha in [(21, "train $|$ validation", -0.2, "right"), (25, "validation $|$ test", 0.2, "left")]:
    ax.axvline(d, color=GREY, lw=0.8, ls=(0, (3, 2)))
    ax.text(d + dx, hc.max() / 1000 * 0.97, lab, color=GREY, fontsize=7, ha=ha, va="top")
ax.set_xlim(0, 30)
ax.set_xlabel("day in trace")
ax.set_ylabel("VMs created per hour\n(thousands)")
save(fig, "fig3_creation_time")

# ---------------------------------------------------------------- fig4: grouped permutation importance
g = pd.read_csv(RES / "03_grouped_importance.csv", index_col=0).sort_values("RF grouped perm. dMAE")
fig, ax = plt.subplots(figsize=(W, 1.9))
yy = np.arange(len(g))
ax.barh(yy, g["RF grouped perm. dMAE"], color=PAL["blue"], height=0.6)
for yi, v in zip(yy, g["RF grouped perm. dMAE"]):
    ax.text(v + 0.2, yi, f"{v:.1f}", va="center", fontsize=7, color=INK)
ax.set_yticks(yy, [s.replace("p95", "P95") for s in g.index])
ax.set_xlabel("increase in test MAE when the feature group is permuted (pp)")
ax.grid(axis="y", visible=False)
ax.grid(axis="x")
ax.set_xlim(0, g["RF grouped perm. dMAE"].max() * 1.1)
save(fig, "fig4_grouped_importance")

# ---------------------------------------------------------------- fig6: MAE by history segment; R2 by test day
seg = pd.read_csv(RES / "03_segments_by_history.csv", index_col=0)
st = pd.read_csv(RES / "03_stress_tests.csv", index_col=0)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw={"width_ratios": [1.35, 1]})
show = ["all test VMs", "no completed sub history (sub_n_hist = 0)", "1 <= sub_n_hist < 50", "sub_n_hist >= 50",
        "sub_n_hist >= 1000"]
lab = ["all", "no history", "1--49", "$\\ge$50", "$\\ge$1,000"]
xx = np.arange(len(show))
for j, (col, c, nm) in enumerate([("MAE mean", GREY, "training mean"), ("MAE RF A", PAL["orange"], "RF, size + time"),
                                  ("MAE RF C", PAL["blue"], "RF, + history")]):
    a1.bar(xx + (j - 1) * 0.26, seg.loc[show, col], width=0.24, color=c, label=nm)
a1.set_xticks(xx, [f"{l}\n{seg.loc[s, 'n']:,}" for l, s in zip(lab, show)], fontsize=6.8)
a1.set_xlabel("completed earlier VMs of the subscription (and test VMs)")
a1.set_ylabel("test MAE (pp)")
a1.set_ylim(0, 45)
a1.set_title("(a) Error by amount of customer history")
a1.legend(loc="upper center", ncol=3, fontsize=7, columnspacing=1.0, handlelength=1.2)
a1.grid(axis="x", visible=False)
dd = st.loc[[i for i in st.index if i.startswith("RF C, test day")]]
days = [int(i.split()[-1]) for i in dd.index]
a2.plot(days, dd.R2, marker="o", color=PAL["blue"], ms=4)
vr = st.loc["RF C, validation period (days 21-24)", "R2"]
a2.axhline(vr, color=GREY, ls=(0, (3, 2)), lw=0.8)
a2.text(29.0, vr + 0.003, "validation $R^2$", color=GREY, fontsize=7, ha="right")
a2.set_xticks(days)
a2.set_xlabel("test day")
a2.set_ylabel("$R^2$ (RF with history)")
a2.set_title("(b) Distribution shift by test day")
save(fig, "fig6_segments_and_shift")

# ---------------------------------------------------------------- fig7: capacity vs risk (feasibility test set)
P = pd.read_parquet(PROC / "predictions.parquet")
Pt = P[P.split == "test"]
qt = Pt.y_p95.to_numpy() / 100


def route(pred_pct, delta=0.0):
    return TIERS[np.minimum(np.searchsorted(TIERS, (pred_pct + delta) / 100 - 1e-12), 3).clip(0)]


def ev(a):
    sf = np.maximum(0, qt - a)
    return {"u": 100 * np.mean(qt > a + 1e-12), "s": sf.mean(), "r": a.mean()}


fig, axes = plt.subplots(1, 2, figsize=(W, 2.6))
fx = [ev(np.full(len(qt), t)) for t in TIERS]
sweeps = {nm: [ev(route(Pt[col].to_numpy(), dl)) for dl in np.arange(-40, 80.1, 2.5)]
          for nm, col in [("Ridge + margin", "pred_Ridge_C"), ("RF + margin", "pred_Random_Forest_C")]}
orc = ev(TIERS[np.searchsorted(TIERS, qt - 1e-12)])
for ax, k, xl, t in [(axes[0], "u", "\\% of VMs under-provisioned (P95 $>$ reservation)", "(a) Capacity vs.\\ risk"),
                     (axes[1], "s", "mean shortfall (fraction of capacity)", "(b) Capacity vs.\\ shortfall size")]:
    ax.plot([f[k] for f in fx], [f["r"] for f in fx], color=GREY, ls=(0, (3, 2)), lw=0.9, marker="s", ms=3,
            mfc="white", label="fixed tiers")
    for t_, f in zip(TIERS, fx):
        ax.annotate(f"{int(t_ * 100)}\\%", (f[k], f["r"]), xytext=(4, 2), textcoords="offset points", fontsize=7,
                    color=GREY)
    for nm, key in [("Ridge + margin", "margin"), ("RF + margin", "opt")]:
        e = sweeps[nm]
        ax.plot([x[k] for x in e], [x["r"] for x in e], color=PAL["orange"] if key == "margin" else PAL["blue"],
                lw=1.3, label=f"{nm} (margin swept)")
    ax.plot(orc[k], orc["r"], marker="*", ms=8, color=INK, ls="none", label="oracle")
    ax.set_xlabel(xl)
    ax.set_ylabel("mean reservation (fraction)")
    ax.set_title(t)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=4, fontsize=7, columnspacing=1.2)
save(fig, "fig7_capacity_vs_risk")
print("group1 done")
