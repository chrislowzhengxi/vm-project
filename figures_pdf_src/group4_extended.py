"""Vector PDF versions of figures/extended_evaluation/ext_fig1-4, redrawn in the report style from saved outputs only
(results/extended_evaluation/*.csv and saved test actions). No model is trained. Run from the repo root."""
import numpy as np
import pandas as pd

from pubstyle import GREY, INK, PROC, RES, STY, W, plt, save

EXT = RES / "extended_evaluation"
FR = pd.read_csv(EXT / "ext_risk_frontier_grid.csv")
OP = pd.read_csv(EXT / "ext_operational_metrics.csv")
MAP = {"OPT depth 3": "opt", "Greedy depth 3": "greedy", "Cost-aware RF": "rf", "Forest-weighted prescription": "fw",
       "RF point + margin": "margin", "H1 customer mean + margin": "h1", "RF point -> tier": "round"}
LAB = {"opt": "exact OPT (depth 3)", "greedy": "greedy tree", "rf": "cost-aware RF", "fw": "forest-weighted",
       "margin": "RF + margin", "h1": "H1 customer rule", "round": "RF rounded up"}
MK = {2: "s", 5: "o", 10: "^"}

# ================================================================== ext_fig1: reservation vs risk frontiers
fig, axes = plt.subplots(2, 2, figsize=(W, 5.0))
panels = [("E[sf]", "expected shortfall $E[(Y-r)^+]$ (pp)", "(a) Reservation vs expected shortfall"),
          ("P(Y>r)", "exceedance frequency $P(Y>r)$ (\\%)", "(b) Reservation vs exceedance"),
          ("P(sf>20pp)", "severe shortfall $P(Y-r>20$\\,pp$)$ (\\%)", "(c) Reservation vs severe shortfall")]
for ax, (risk, xl, title) in zip(axes.ravel()[:3], panels):
    for nm, k in MAP.items():
        if k == "round":
            continue
        st = STY[k]
        g = FR[FR.method == nm].sort_values("beta")
        ax.plot(100 * g[risk], 100 * g["E[r]"], color=st["color"], ls=st["ls"], marker=st["marker"], ms=2.6,
                lw=1.1, label=LAB[k], zorder=3)
        r5 = g[g.beta == 5].iloc[0]
        ax.plot(100 * r5[risk], 100 * r5["E[r]"], marker=st["marker"], ms=5.5, color=st["color"], mec=INK,
                mew=0.5, ls="none", zorder=4)
    fx = FR[FR.method == "Fixed tier"].sort_values("tier")
    ax.plot(100 * fx[risk], 100 * fx["E[r]"], color=GREY, ls=(0, (3, 2)), lw=0.9, marker="s", ms=3, mfc="white",
            zorder=2, label="fixed tiers and mixtures")
    o = OP[(OP.beta == 5) & (OP.policy == "Oracle (hindsight)")].iloc[0]
    ax.plot(100 * o[risk], 100 * o["E[r]"], marker="*", ms=8, color=INK, ls="none", label="oracle, $\\beta=5$")
    ax.set_xlabel(xl)
    ax.set_ylabel("mean reservation $E[r]$ (\\%)")
    ax.set_title(title)
    ax.set_ylim(35, 102)
    ax.set_xlim(left=-0.5)
axes[0, 0].set_xlim(-0.3, 19)
axes[0, 1].set_xlim(-1, 62)
axes[1, 0].set_xlim(-0.5, 45)
ax = axes[1, 1]
d = OP[(OP.beta == 5) & OP.policy.isin(list(MAP)) & (OP.policy != "RF point -> tier")]
OFF = {"opt": (5, -3), "greedy": (-5, -10), "rf": (5, -6), "fw": (5, 3), "margin": (5, 2), "h1": (-5, -8)}
for _, r in d.iterrows():
    k = MAP[r.policy]
    st = STY[k]
    ax.plot(100 * r["P(Y>r)"], r.cost, marker=st["marker"], ms=5.5, color=st["color"], ls="none")
    dx, dy = OFF[k]
    ax.annotate(LAB[k], (100 * r["P(Y>r)"], r.cost), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                color=GREY, ha="left" if dx > 0 else "right")
for nm, lab, dx in (("OPT 4-leaf (depth 2)", "OPT 4 leaves", -5), ("OPT depth 4 (certified)", "OPT depth 4", 5)):
    dy = 4 if dx < 0 else -3
    r = OP[(OP.beta == 5) & (OP.policy == nm)].iloc[0]
    ax.plot(100 * r["P(Y>r)"], r.cost, marker="o", ms=5, mfc="white", mec=STY["opt"]["color"], mew=1.0, ls="none")
    ax.annotate(lab, (100 * r["P(Y>r)"], r.cost), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                color=GREY, ha="left" if dx > 0 else "right")
ax.set_xlim(6.5, 12.8)
ax.set_xlabel("exceedance frequency $P(Y>r)$ (\\%)")
ax.set_ylabel("mean modeled cost, $\\beta=5$")
ax.set_title("(d) $\\beta=5$: cost vs exceedance")
ax.text(0.015, 0.25, "off panel:\nbest fixed (0\\%, 1.000)\nRF rounded up (29.9\\%, 0.858)", transform=ax.transAxes,
        va="bottom", fontsize=6.4, color=GREY, linespacing=1.15)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=4, fontsize=7.2, columnspacing=1.4)
save(fig, "extended_evaluation/ext_fig1_risk_frontier")

# ================================================================== ext_fig2: customer-level savings (beta = 5)
P = pd.read_csv(EXT / "ext_customer_per_subscription_b5.csv")
CU = pd.read_csv(EXT / "ext_customer_heterogeneity.csv")
CP = [("OPT depth 3", "opt"), ("Cost-aware RF", "rf"), ("Forest-weighted prescription", "fw"),
      ("RF point + margin", "margin")]
STR = ["1-4 test VMs", "5-19", "20-99", "100-999", ">= 1000"]
STRL = ["1--4", "5--19", "20--99", "100--\n999", "$\\ge$1,000"]
fig, (b1, b2, b3) = plt.subplots(1, 3, figsize=(W, 2.35), gridspec_kw={"width_ratios": [1.2, 1, 1]})
for key, k in CP:
    st = STY[k]
    v = np.sort(P[(P.policy == key) & (P["test VMs"] >= 5)]["% saving"].to_numpy())
    b1.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=st["color"], ls=st["ls"], lw=1.1, label=LAB[k])
b1.axvline(0, color=GREY, lw=0.6)
b1.set_xlim(-60, 80)
b1.set_ylim(0, 1.02)
b1.set_xlabel("\\% saving per subscription ($\\ge$5 test VMs)")
b1.set_ylabel("cumulative share of subscriptions")
b1.set_title("(a) Distribution")
b1.grid(axis="x")
xs = np.arange(len(STR))
w = 0.19
for j, (key, k) in enumerate(CP):
    st = STY[k]
    dd = CU[(CU.beta == 5) & (CU.policy == key)].set_index("subscriptions").loc[STR]
    x = xs + (j - 1.5) * w
    b2.errorbar(x, dd["median % saving"], yerr=[dd["median % saving"] - dd.P25, dd.P75 - dd["median % saving"]],
                fmt="o", ms=3, color=st["color"], lw=0.8, capsize=1.5, capthick=0.7)
    b2.plot(x, dd["VM-weighted % saving"], ls="none", marker="D", ms=3, mfc="white", mec=st["color"], mew=0.8)
    b3.bar(x, dd["% subs higher cost"], width=w * 0.92, color=st["color"])
b2.plot([], [], "ko", ms=3, label="median [IQR]")
b2.plot([], [], ls="none", marker="D", ms=3, mfc="white", mec="k", label="VM-weighted")
b2.legend(loc="upper right", fontsize=6.8, handletextpad=0.3, borderaxespad=0.1)
b2.axhline(0, color=GREY, lw=0.5)
b2.set_ylim(-3, 55)
b2.set_ylabel("\\% saving")
b2.set_title("(b) Saving by subscription size")
b3.set_ylabel("\\% of subscriptions")
b3.set_title("(c) Costlier than always-100\\%")
for ax in (b2, b3):
    ax.set_xticks(xs, STRL, fontsize=7)
    ax.set_xlabel("test VMs per subscription")
    ax.set_xlim(-0.55, len(STR) - 0.45)
h, l = b1.get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=4)
save(fig, "extended_evaluation/ext_fig2_customer_savings")

# ================================================================== ext_fig3: shortfall severity (CCDF from saved actions)
F = pd.read_parquet(PROC / "features.parquet", columns=["y_p95"])
PA = pd.read_parquet(PROC / "final" / "policy_actions_test.parquet")
HW = pd.read_parquet(PROC / "hw6" / "forest_actions_test.parquet")
assert np.array_equal(PA.index.to_numpy(), HW.index.to_numpy())
qt = F.y_p95.to_numpy()[PA.index.to_numpy()] / 100
NT = len(qt)
SRC = [("RF predict -> tier", "round"), ("RF predict + margin", "margin"), ("H1: customer mean P95 + margin", "h1"),
       ("Greedy [full]", "greedy"), ("OPT-exact [full]", "opt"), ("Cost-aware RF", "rf")]
xg = np.linspace(0, 0.75, 301)


def ccdf(a):
    sf = np.sort(np.maximum(0, qt - a))
    c = 1 - np.searchsorted(sf, xg + 1e-12, side="left") / NT
    return np.where(c > 0, 100 * c, np.nan)


# consistency with the stage-12 table (OPT at beta = 5)
chk = OP[(OP.beta == 5) & (OP.policy == "OPT depth 3")].iloc[0]
a5 = PA["OPT-exact [full]|b=5"].to_numpy(float)
assert np.isclose(np.mean(qt > a5 + 1e-12), chk["P(Y>r)"]) and np.isclose(np.maximum(0, qt - a5).mean(), chk["E[sf]"])
fig, axes = plt.subplots(1, 3, figsize=(W, 2.45), sharey=True)
for ax, b in zip(axes, (2, 5, 10)):
    for col, k in SRC:
        st = STY[k]
        ax.plot(100 * xg, ccdf(PA[f"{col}|b={b}"].to_numpy(float)), color=st["color"], ls=st["ls"], lw=1.1,
                label=LAB[k])
    a = HW[f"C: forest-weighted prescription [main forest, msl=50]|b={b}"].to_numpy(float)
    ax.plot(100 * xg, ccdf(a), color=STY["fw"]["color"], ls=STY["fw"]["ls"], lw=1.1, label=LAB["fw"])
    ax.plot(100 * xg, ccdf(PA[f"Oracle (hindsight)|b={b}"].to_numpy(float)), color=INK, ls=(0, (1, 1)), lw=0.9,
            label="oracle (hindsight)")
    bf = PA[f"Best fixed (val)|b={b}"].to_numpy(float)
    if bf[0] < 1:
        ax.plot(100 * xg, ccdf(bf), color=GREY, ls=(0, (3, 2)), lw=1.0, label="best fixed tier")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 100)
    ax.set_xlim(0, 76)
    ax.set_xlabel("shortfall threshold $x$ (pp)")
    tier = int(round(100 * bf[0]))
    ax.set_title(f"$\\beta={b}$ (best fixed tier {tier}\\%)")
    ax.grid(axis="both", which="major")
axes[0].set_ylabel("\\% of VMs with $Y-r>x$")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=5, fontsize=7.2, columnspacing=1.3)
save(fig, "extended_evaluation/ext_fig3_shortfall_severity")

# ================================================================== ext_fig4: RF calibration and tier mix
CAL = pd.read_csv(EXT / "ext_rf_calibration.csv")
fig, (c1, c2) = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw={"width_ratios": [1, 1.25]})
m = CAL["mean predicted"]
c1.fill_between(m, CAL["realized P10"], CAL["realized P90"], color=STY["opt"]["color"], alpha=0.15, lw=0,
                label="realized P10--P90")
c1.plot(m, CAL["mean realized"], color=STY["opt"]["color"], marker="o", ms=3.2, label="mean realized")
c1.plot(m, CAL["realized P50"], color=STY["opt"]["color"], ls=(0, (3, 1.5)), lw=1.0, label="median realized")
c1.plot([0, 100], [0, 100], color=GREY, lw=0.7, label="perfect calibration")
for t in (25, 50, 75):
    c1.axhline(t, color="#c9c8c3", lw=0.6, zorder=0)
c1.set_xlim(0, 100)
c1.set_ylim(0, 100)
c1.set_xlabel("RF predicted P95 (\\%), bin mean")
c1.set_ylabel("realized P95 (\\%)")
c1.set_title("(a) Calibration by prediction bin")
c1.legend(loc="upper left", fontsize=6.6, borderaxespad=0.2)
xs = np.arange(len(CAL))
bottom = np.zeros(len(CAL))
for col, lab, colr in (("% Y <= 25", "realized $\\le$25\\%", "#d6e6fa"), ("% 25 < Y <= 50", "25--50\\%", "#9cc3f0"),
                       ("% 50 < Y <= 75", "50--75\\%", "#4a8ad8"), ("% Y > 75", "$>$75\\%", "#173f73")):
    c2.bar(xs, CAL[col], bottom=bottom, color=colr, width=0.8, label=lab)
    bottom += CAL[col].to_numpy()
c2.plot(xs, CAL["E[r] % point->tier (b=5)"], color=STY["round"]["color"], marker="x", ms=3.5, lw=1.1,
        label="reservation: rounded up")
c2.plot(xs, CAL["E[r] % forest-weighted (b=5)"], color=STY["fw"]["color"], marker="D",
        ms=3.2, lw=1.1, label="reservation: forest-weighted")
c2.set_xticks(xs, [s.replace("[", "").replace(")", "").replace(", ", "--") for s in CAL["predicted P95 bin (%)"]],
              fontsize=6.6, rotation=35, ha="right", rotation_mode="anchor")
c2.set_xlabel("RF predicted P95 bin (\\%)")
c2.set_ylabel("\\% of VMs / mean reservation (\\%)")
c2.set_title("(b) Realized tier mix and chosen reservation ($\\beta=5$)")
c2.set_ylim(0, 100)
h2, l2 = c2.get_legend_handles_labels()
fig.legend(h2, l2, loc="outside upper center", ncol=6, fontsize=6.8, columnspacing=1.0, handlelength=1.6)
c2.grid(axis="y")
save(fig, "extended_evaluation/ext_fig4_rf_calibration")
print("ok")
