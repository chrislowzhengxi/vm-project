"""Vector PDF versions of the HW6 and sensitivity figures (report style), from saved outputs only.

No model is trained, refit or re-solved. Saved OCT trees (data/processed/hw6/iai_oct/oct_{1,2}.json = depth 3, 4) are
applied to the (training-imputed) test features to get leaf shares, exactly as src/09_hw6_clustering.py did.
hw6_fig_forest_example is NOT rebuilt: its weight histograms need the fitted P95 forest, which was not saved.
Run from the repo root:  /usr/local/bin/python3 figures_pdf_src/group3_hw6_sens.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "figures_pdf_src"))
import final_setup as fs  # noqa: E402  (imported before pubstyle: common.py sets its own rcParams)
from pubstyle import GREY, GRID, INK, PAL, PROC, RES, STY, W, plt, save  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SENS, HW = RES / "sensitivity", RES / "hw6"
BETAS = (2, 5, 10)
TIERC = {0.25: "#d6e6fa", 0.5: "#9cc3f0", 0.75: "#4a8ad8", 1.0: "#173f73"}


def lstyle(k, **kw):
    s = STY[k]
    return dict(color=s["color"], ls=s["ls"], marker=s["marker"], **kw)


# =================================================================== HW6: forest rules (same data as original)
T = pd.read_csv(HW / "hw6_forest_results.csv")
show = [("A: point prediction -> tier [val-selected forest]", "A: point $\\to$ tier", PAL["red"]),
        ("B: point prediction + margin [val-selected forest]", "B: point + margin", PAL["orange"]),
        ("C: forest-weighted prescription [val-selected forest]", "C: forest-weighted", PAL["darkgreen"]),
        ("Cost-aware RF (existing)", "cost-aware RF", PAL["green"]),
        ("OPT-exact (existing)", "exact OPT", PAL["blue"])]
fig, ax = plt.subplots(figsize=(W, 2.45))
wd = 0.155
for j, (p, lab, c) in enumerate(show):
    v = [T[(T.beta == b) & (T.policy == p)]["% reduction vs best fixed"].iloc[0] for b in BETAS]
    xs = np.arange(3) + (j - 2) * wd
    ax.bar(xs, v, width=wd * 0.9, color=c, label=lab, zorder=3)
    for x, y in zip(xs, v):
        ax.text(x, y + (0.4 if y >= 0 else -0.5), f"{y:.1f}", ha="center", va="bottom" if y >= 0 else "top",
                fontsize=6.6, color=INK)
ax.axhline(0, color=GREY, lw=0.6)
ax.set_xticks(range(3), [f"$\\beta={b}$" for b in BETAS])
ax.set_ylim(-14.5, 25)
ax.set_ylabel("\\% saving vs best fixed (test)")
ax.grid(axis="x", visible=False)
fig.legend(loc="outside upper center", ncol=5)
save(fig, "hw6/hw6_fig_forest_rules")

# =================================================================== HW6: cluster profiles (from the saved profile table)
P = pd.read_csv(HW / "hw6_cluster_profiles.csv")          # rows already ordered by realized mean P95
COLS = [("share of train VMs %", "VMs\n(\\%)", "{:.0f}"), ("median vCPU", "median\nvCPUs", "{:.0f}"),
        ("median memory GB", "median\nmem. (GB)", "{:.0f}"),
        ("mean sub mean past P95 %", "cust. mean\nP95 (\\%)", "{:.0f}"),
        ("mean sub share past VMs >75% %", "cust. VMs\n$>$75\\% (\\%)", "{:.0f}"),
        ("median # completed sub VMs", "cust. past\nVMs (\\#)", "{:,.0f}"),
        ("mean dep mean past P95 %", "depl. mean\nP95 (\\%)", "{:.0f}"),
        ("realized mean P95 % (train, not a feature)", "realized\nP95$^\\dagger$ (\\%)", "{:.0f}")]
vals = P[[c for c, _, _ in COLS]].to_numpy(float)
z = (vals - vals.mean(0)) / vals.std(0)
z[:, 5] = (np.log10(vals[:, 5]) - np.log10(vals[:, 5]).mean()) / np.log10(vals[:, 5]).std()   # counts: log scale
fig, ax = plt.subplots(figsize=(W, 2.75))
K, J = vals.shape
ax.pcolormesh(np.arange(J + 1), np.arange(K + 1), z, cmap="RdBu_r", vmin=-2.2, vmax=2.2, edgecolors="white",
              linewidth=0.8)
for i in range(K):
    for j, (_, _, f) in enumerate(COLS):
        ax.text(j + 0.5, i + 0.5, f.format(vals[i, j]).replace(",", "{,}"), ha="center", va="center", fontsize=7,
                color="white" if abs(z[i, j]) > 1.3 else INK)
    for jj, b in enumerate(BETAS):
        t = float(P.iloc[i][f"action b={b}"].rstrip("%")) / 100
        ax.add_patch(plt.Rectangle((J + 0.15 + jj * 0.62, i + 0.12), 0.56, 0.76, fc=TIERC[t], ec="none"))
        ax.text(J + 0.43 + jj * 0.62, i + 0.5, f"{int(100 * t)}\\%", ha="center", va="center", fontsize=6.6,
                color="white" if t >= 0.75 else INK)
for jj, b in enumerate(BETAS):
    ax.text(J + 0.43 + jj * 0.62, -0.08, f"$\\beta={b}$", ha="center", va="bottom", fontsize=7)
ax.text(J + 0.15 + 0.93, -0.55, "reservation", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(np.arange(J) + 0.5, [l for _, l, _ in COLS], fontsize=6.9)
fig.supxlabel("$^\\dagger$ training VMs, not a clustering feature. Colors: z-score across clusters (log scale for past VMs).", fontsize=6.4, color=GREY)
ax.xaxis.tick_top()
ax.set_yticks(np.arange(K) + 0.5, [f"cluster {c}" for c in P.cluster])
ax.set_xlim(0, J + 2.1)
ax.set_ylim(K, 0)
ax.tick_params(length=0)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
save(fig, "hw6/hw6_fig_cluster_profiles")

# =================================================================== HW6: OCT trees (saved IAI trees, depth 3 and 4)
OCTJ = json.loads((HW / "hw6_oct_tree.json").read_text())
FEATS = OCTJ["features"]
ACT5 = OCTJ["cluster_actions"]["5"]
F, M = fs.load_strict()                                      # training-only imputation, as in stage 9
ite = np.flatnonzero(M["test"])
Xo = F[FEATS].to_numpy(float)[ite]


def iai_tree(path):
    js = json.loads(Path(path).read_text())
    nodes = {n["id"]: n for n in js["nodes"]}

    def rec(k):
        n = nodes[k]
        if n["leaf"]:
            return {"leaf": True, "label": int(n["label"])}
        return {"leaf": False, "feature": n["feature"] - 1, "threshold": float(n["threshold"]),
                "left": rec(n["lower"]), "right": rec(n["upper"])}
    return rec(1)


def same(a, b):
    if a["leaf"] or b["leaf"]:
        return a["leaf"] and b["leaf"] and a["label"] == b["label"]
    return (a["feature"] == b["feature"] and np.isclose(a["threshold"], b["threshold"]) and same(a["left"], b["left"])
            and same(a["right"], b["right"]))


OCT = {3: iai_tree(PROC / "hw6/iai_oct/oct_1.json"), 4: iai_tree(PROC / "hw6/iai_oct/oct_2.json")}
assert same(OCT[4], OCTJ["tree"]), "cached depth-4 OCT differs from the saved selected tree"
QL = {"cores": ("vCPUs", "$<$ {:.0f}?"), "memory_gb": ("memory", "$<$ {:.0f} GB?"),
      "sub_hist_mean": ("cust. mean P95", "$<$ {:.1f}\\%?"),
      "sub_hist_median": ("cust. median P95", "$<$ {:.1f}\\%?"),
      "sub_hist_frac_le75": ("share P95$\\le$75\\%", "$<$ {:.2f}?"),
      "sub_hist_frac_le50": ("share P95$\\le$50\\%", "$<$ {:.2f}?"),
      "sub_n_hist": ("cust. past VMs", "$<$ {:,.0f}?"),
      "dep_hist_mean": ("depl. mean P95", "$<$ {:.1f}\\%?")}


def draw_oct(tree, name, h):
    def lay(n, d, pos, idx):
        n["_d"], n["_n"] = d, len(idx)
        if n["leaf"]:
            n["_x"] = pos[0]
            pos[0] += 1
            return
        go = Xo[idx, n["feature"]] < n["threshold"]
        lay(n["left"], d + 1, pos, idx[go])
        lay(n["right"], d + 1, pos, idx[~go])
        n["_x"] = (n["left"]["_x"] + n["right"]["_x"]) / 2

    pos = [0]
    lay(tree, 0, pos, np.arange(len(ite)))
    nl = pos[0]
    leaves = []
    fig = plt.figure(figsize=(W, h), layout="none")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    big = nl > 10
    fsz = 6.4 if big else 7.3

    def draw(n):
        x, y = n["_x"], -n["_d"]
        if n["leaf"]:
            t = ACT5[n["label"]]
            leaves.append((n["label"], round(100 * n["_n"] / len(ite), 1)))
            ax.text(x, y, f"\\textbf{{{int(100 * t)}\\%}}", ha="center", va="center", fontsize=fsz + 0.8,
                    color="white" if t >= 0.75 else INK,
                    bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.15", fc=TIERC[t], ec="none"))
            ax.text(x, y - 0.34, f"cluster {n['label']}\n{100 * n['_n'] / len(ite):.1f}\\%", ha="center", va="top",
                    fontsize=fsz - 0.4, color=GREY, linespacing=1.2)
            return
        for ch, lab in ((n["left"], "yes"), (n["right"], "no")):
            cx, cy = ch["_x"], -ch["_d"]
            y0, y1 = y - 0.2, cy + (0.17 if ch["leaf"] else 0.22)
            ax.plot([x, cx], [y0, y1], color="#9a9994", lw=0.6, zorder=0)
            ax.text(x + 0.45 * (cx - x), y0 + 0.45 * (y1 - y0), lab, fontsize=fsz - 0.8, color=GREY, ha="center",
                    va="center", bbox=dict(fc="white", ec="none", pad=0.4))
            draw(ch)
        l1, l2 = QL[FEATS[n["feature"]]]
        ax.text(x, y, "\\shortstack{" + l1 + "\\\\" + l2.format(n["threshold"]).replace(",", "{,}") + "}",
                ha="center", va="center", fontsize=fsz, color=INK,
                bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.15", fc="#f4f3ef", ec="#a8a7a2", lw=0.5))

    draw(tree)
    ax.set_xlim(-0.6, nl - 0.4)
    md = max_d(tree)
    ax.set_ylim(-md - 0.85, 0.32)
    save(fig, name)
    return leaves


def max_d(n):
    return 0 if n["leaf"] else 1 + max(max_d(n["left"]), max_d(n["right"]))


lv3 = draw_oct(OCT[3], "hw6/hw6_fig_cluster_oct_depth3", 2.6)
assert [s for _, s in lv3] == [8.4, 29.7, 8.5, 14.4, 6.3, 7.7, 16.4, 8.6], lv3      # matches the original figure
lv4 = draw_oct(OCT[4], "hw6/hw6_fig_cluster_oct", 3.2)

# =================================================================== sensitivity data
E2 = pd.read_csv(SENS / "s10_e2_reoptimized.csv")
GRID_B = sorted(E2.beta.unique())
FX = pd.read_csv(SENS / "s10_e4_fixed_beta5_policies.csv")
REG = pd.read_csv(SENS / "s10_e4_regret.csv")
SW = pd.read_csv(SENS / "s10_e1_best_fixed_switches.csv")
sw = SW[SW.split.str.startswith("val") & ~SW["best fixed tier"].str.startswith("pairwise")].sort_values("beta from")
SWX = sw["beta to"].to_numpy()[:3]
SWT = sw["best fixed tier"].to_numpy()[1:]
assert np.allclose(np.round(SWX, 2), [1.60, 2.29, 4.46])


def beta_axis(ax, ticks=(1.5, 2, 3, 5, 10, 20), lo=1.4, hi=21.5):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xticks(list(ticks), [f"{t:g}" for t in ticks])
    ax.minorticks_off()


# =================================================================== sens fig 1: fixed vs re-optimized; wrong-beta regret
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.9))
for pol, k, lab in (("OPT-exact (val-selected)", "opt", "exact OPT"), ("Greedy (val-selected)", "greedy", "greedy tree"),
                    ("Cost-aware RF", "rf", "cost-aware RF"), ("Forest-weighted prescription", "fw", "forest-weighted"),
                    ("H1 customer mean + margin", "h1", "H1 customer rule")):
    d = E2[E2.policy == pol]
    a1.plot(d.beta, d["% saving vs best fixed (val)"], label=f"{lab} (re-tuned)", **lstyle(k, ms=3))
for nm, k, lab in (("OPT-exact built for beta=5 (fixed)", "opt", "exact OPT built for $\\beta=5$"),
                   ("Cost-aware RF decided at beta=5 (fixed)", "rf", "cost-aware RF at $\\beta=5$")):
    d = FX[FX.policy == nm]
    a1.plot(d.beta, d["% saving vs best fixed (val)"], color=STY[k]["color"], ls=(0, (2, 1.2)), lw=1.0, marker=None,
            alpha=0.85, label=lab)
for x, t in zip(SWX, SWT):
    a1.axvline(x, color=GREY, lw=0.5, ls=":")
    a1.text(x * 1.03, -11.5, f"$\\to${t}", fontsize=6.6, color=GREY, va="bottom")
a1.axhline(0, color=GREY, lw=0.5)
beta_axis(a1)
a1.set_ylim(-12.5, 24)
a1.set_xlabel("$\\beta$ (log scale)")
a1.set_ylabel("\\% saving vs best fixed tier (test)")
a1.set_title("(a) Re-tuned at each $\\beta$ vs fixed at $\\beta=5$")
h1_, l1_ = a1.get_legend_handles_labels()
fig.legend(h1_, l1_, loc="outside lower center", ncol=4, fontsize=6.6, handlelength=2.0, columnspacing=1.0)
for pol, split, k, ls, col, lab in (
        ("OPT-exact", "test", "opt", "-", "regret vs val-tuned %", "exact OPT, test"),
        ("OPT-exact", "val", "opt", (0, (1, 1.2)), "regret vs val-tuned %", "exact OPT, validation"),
        ("OPT-exact", "train", None, (0, (4, 1.5, 1, 1.5)), "regret vs same-class %", "exact OPT, train (vs class optimum)"),
        ("Cost-aware RF (decision rule)", "test", "rf", "-", "regret vs val-tuned %", "cost-aware RF, test"),
        ("Best fixed tier", "test", "fixed", "-", "regret vs val-tuned %", "best fixed tier, test")):
    d = REG[(REG.policy == pol) & (REG.split == split)]
    c = GREY if k is None else STY[k]["color"]
    if k == "fixed":
        c = PAL["orange"]
    a2.plot(d.beta, d[col], color=c, ls=ls, marker="o", ms=2.6, label=lab)
a2.axhline(0, color=GREY, lw=0.5)
a2.axvline(5, color=GREY, lw=0.5, ls=":")
beta_axis(a2)
a2.set_xlabel("true $\\beta$ (policy built for $\\beta=5$)")
a2.set_ylabel("regret: \\% extra cost")
a2.set_title("(b) Cost of assuming the wrong $\\beta$")
a2.legend(loc="upper center", fontsize=6.4, handlelength=1.8)
save(fig, "sensitivity/sens_fig1_fixed_vs_reoptimized")

# =================================================================== sens fig 2: tier shares across beta (vector heatmaps)
fig, axes = plt.subplots(1, 3, figsize=(W, 2.85), sharey=True)
TC = ["% at 25%", "% at 50%", "% at 75%", "% at 100%"]
for ax, (pol, ttl) in zip(axes, (("OPT-exact (val-selected)", "(a) exact OPT"), ("Cost-aware RF", "(b) cost-aware RF"),
                                 ("Oracle (hindsight)", "(c) oracle (hindsight)"))):
    d = E2[E2.policy == pol].set_index("beta").loc[GRID_B, TC].to_numpy(float)
    ax.pcolormesh(np.arange(5), np.arange(len(GRID_B) + 1), d, cmap="Blues", vmin=0, vmax=100, edgecolors="white",
                  linewidth=0.6)
    for i in range(len(GRID_B)):
        for j in range(4):
            ax.text(j + 0.5, i + 0.5, f"{d[i, j]:.0f}", ha="center", va="center", fontsize=6.6,
                    color="white" if d[i, j] > 55 else INK)
    ax.set_xticks(np.arange(4) + 0.5, ["25\\%", "50\\%", "75\\%", "100\\%"])
    ax.set_title(ttl)
    ax.set_ylim(len(GRID_B), 0)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
axes[0].set_yticks(np.arange(len(GRID_B)) + 0.5, [f"{g:g}" for g in GRID_B])
axes[0].set_ylabel("$\\beta$")
bfs = [E2[(E2.policy == "Best fixed (val)") & (E2.beta == g)]["best fixed (val) tier"].iloc[0] for g in GRID_B]
for i, t in enumerate(bfs):
    axes[2].text(4.55, i + 0.5, t.replace("%", "\\%"), fontsize=6.6, ha="center", va="center", color=GREY, clip_on=False)
axes[2].text(4.55, -0.25, "best\nfixed", fontsize=6.6, ha="center", va="bottom", color=GREY, clip_on=False)
for ax in axes:
    ax.set_xlabel("reservation tier (\\% of test VMs)")
save(fig, "sensitivity/sens_fig2_tier_shares")

# =================================================================== sens fig 3: exact change points of the optimal tree
SEG = pd.read_csv(SENS / "s10_e3_change_points.csv")
fig, axes = plt.subplots(1, 2, figsize=(W, 2.6))
handles = []
for frac, c in ((5.0, PAL["blue"]), (1.0, PAL["orange"])):
    s_ = SEG[SEG["min leaf %"] == frac]
    xs = np.r_[s_["beta from"].to_numpy(), s_["beta to"].iloc[-1]]
    for ax, var, sc in zip(axes, ("reserved", "shortfall"), (100, 100)):
        tr = ax.get_xaxis_transform()
        y0r = 0.0 if frac == 5.0 else 0.045
        for x in s_["beta from"].iloc[1:]:
            ax.plot([x, x], [y0r, y0r + 0.035], transform=tr, color=c, lw=0.5, zorder=1, clip_on=False)
        for split, ls in (("train", "-"), ("test", (0, (3, 1.5)))):
            ys = sc * np.r_[s_[f"{split} {var}"].to_numpy(), s_[f"{split} {var}"].iloc[-1]]
            h, = ax.step(xs, ys, where="post", color=c, ls=ls, lw=1.1,
                         label=f"min leaf {frac:.0f}\\%, {'train (exact optimum)' if split == 'train' else 'test (same trees)'}")
            if ax is axes[0]:
                handles.append(h)
for ax, lab, ttl in zip(axes, ("mean reservation $E[r]$ (\\%)", "expected shortfall $E[(Y-r)^+]$ (pp)"),
                        ("(a) Reservation of the optimal tree", "(b) Shortfall of the optimal tree")):
    beta_axis(ax, ticks=(1, 1.5, 2, 3, 5, 7, 10, 15, 20), lo=1, hi=20)
    ax.set_xlabel("$\\beta$ (log scale)")
    ax.set_ylabel(lab)
    ax.set_title(ttl)
axes[0].set_ylim(17, 88)
axes[1].set_ylim(-3.5, 31)
fig.legend(handles=handles, loc="outside upper center", ncol=2, fontsize=7)
save(fig, "sensitivity/sens_fig3_opt_change_points")

# =================================================================== sens fig 4: cost vs complexity
H = pd.read_csv(SENS / "s11_e6_lambda_hull.csv")
FR = pd.read_csv(SENS / "s11_e5_frontier.csv")
SEL = pd.read_csv(SENS / "s11_e6_selection.csv")
fig, axes = plt.subplots(2, 3, figsize=(W, 3.9), sharex=True)
for j, b in enumerate(BETAS):
    for i, split in enumerate(("val", "test")):
        ax = axes[i, j]
        h = H[H.beta == b].sort_values("leaves")
        ax.plot(h.leaves, h[f"{split} cost"], color=PAL["blue"], marker="o", ms=3, lw=1.1,
                label="exact OPT, leaf-penalty path")
        f5 = FR[(FR.source == "stage 5") & (FR.beta == b) & (FR["train set"] == "full")]
        for meth, k in (("OPT-exact", "opt"), ("Greedy", "greedy"), ("CART-MSE", "cart")):
            d = f5[f5.method == meth].sort_values("leaves")
            c, mk = STY[k]["color"], STY[k]["marker"]
            d3 = d[d["max depth"] <= 3]
            ax.plot(d3.leaves, d3[f"{split} cost"], ls="none", marker=mk, ms=5, mfc="white", mec=c, mew=0.9,
                    label=f"{meth.replace('OPT-exact', 'exact OPT').replace('Greedy', 'greedy')}, depth 1--3")
            d4 = d[d["max depth"] == 4]
            if len(d4):
                ax.plot(d4.leaves, d4[f"{split} cost"], ls="none", marker=mk, ms=5, color=c,
                        label=f"{meth.replace('OPT-exact', 'exact OPT').replace('Greedy', 'greedy')}, depth 4")
        for pol, k, lab in (("Cost-aware RF", "rf", "cost-aware RF"), ("H1 customer mean + margin", "h1", "H1 customer rule")):
            v = E2[(E2.beta == b) & (E2.policy == pol)][f"{split} cost"].iloc[0]
            ax.axhline(v, color=STY[k]["color"], ls=(0, (3, 1.5)), lw=0.9, label=lab)
        x = SEL[(SEL.beta == b) & SEL["selection rule"].str.startswith("simplest")].iloc[0]
        ax.plot(x.leaves, x[f"{split} cost"], marker="*", ms=9, color=PAL["red"], ls="none",
                label="validation-selected penalized tree")
        if i == 0:
            ax.set_title(f"$\\beta={b}$")
        if j == 0:
            ax.set_ylabel(f"{'validation' if split == 'val' else 'test'} cost")
        if i == 1:
            ax.set_xlabel("leaves")
        ax.set_xticks(range(1, 17, 3))
h_, l_ = axes[0, 0].get_legend_handles_labels()
seen = dict(zip(l_, h_))
fig.legend(list(seen.values()), list(seen.keys()), loc="outside upper center", ncol=3, fontsize=6.8)
save(fig, "sensitivity/sens_fig4_complexity")

# =================================================================== sens fig 5: exact depth 4 vs depth 3 and alternatives
runs = [json.loads(p.read_text()) for p in sorted((SENS / "depth4").glob("run_*.json"))]
MT = pd.read_csv(RES / "final_main_table.csv")
S5 = pd.read_csv(RES / "final_tree_selected.csv")
fig, axes = plt.subplots(1, 3, figsize=(W, 2.3), sharey=True)
for j, (ax, b) in enumerate(zip(axes, BETAS)):
    r = min([x for x in runs if x["beta"] == b and x["bins"] == 32 and x["status"].startswith("OPTIMAL")],
            key=lambda x: x["depth4"]["val_cost"])
    items = []
    for pol, lab, k in (("OPT-exact [full]", "exact OPT, depth 3", "opt"), ("Greedy depth-4 [full]", "greedy, depth 4", "greedy"),
                        ("OPT-IAI depth-4 [200k]", "IAI OPT, depth 4", "iai"), ("Cost-aware RF", "cost-aware RF", "rf")):
        m = MT[(MT.beta == b) & (MT.policy == pol)].iloc[0]
        row = S5[(S5.policy == pol) & (S5.beta == b)]
        lv = f"{int(row.leaves.iloc[0])}" if len(row) else "--"
        items.append((lab, lv, m["val cost"], m["test cost"], STY[k]["color"]))
    items.insert(1, ("exact OPT, depth 4", f"{r['depth4']['leaves']}", r["depth4"]["val_cost"], r["depth4"]["test_cost"],
                     "#173f73"))
    ys = np.arange(len(items))[::-1]
    for y, (lab, lv, v, t, c) in zip(ys, items):
        ax.plot([v, t], [y, y], color=c, lw=0.6, alpha=0.6)
        ax.plot(v, y, marker="o", mfc="white", mec=c, mew=1.0, ms=5, ls="none")
        ax.plot(t, y, marker="o", color=c, ms=5, ls="none")
        if lv != "--":
            ax.annotate(f"{lv} leaves", (t, y), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6.2,
                        color=GREY)
    if j == 0:
        ax.set_yticks(ys, [it[0] for it in items])
    ax.set_ylim(-0.6, len(items) - 0.3)
    ax.set_title(f"$\\beta={b}$")
    ax.set_xlabel("mean cost")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.margins(x=0.12)
axes[2].plot([], [], marker="o", mfc="white", mec=GREY, ls="none", label="validation")
axes[2].plot([], [], marker="o", color=GREY, ls="none", label="test")
axes[2].legend(loc="lower right", fontsize=6.6, handletextpad=0.2)
save(fig, "sensitivity/sens_fig5_depth4")
print("done; OCT depth-3 leaves", lv3, "depth-4 leaves", lv4)
