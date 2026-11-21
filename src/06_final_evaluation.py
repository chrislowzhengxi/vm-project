"""Stage 6 (final project): every benchmark under the strict chronological setup, evaluated on the full test period.

Policies (all tuning on the strict validation set, separately for each beta):
  Always 100% / fixed tiers / best fixed;  oracle = hindsight cost-minimizing tier (a lower bound under the modeled
  cost, not an achievable policy);  customer heuristics H1 (subscription mean past P95 + tuned margin) and
  H2 (critical-ratio rule, no tuning: smallest tier covering >= 1 - 1/beta of the customer's completed VMs);
  Ridge / CART / RF prediction of P95 -> smallest covering tier (naive and with a validation-tuned margin);
  cost-aware RF (predicted shortfall per tier -> cheapest tier);  policy trees from stage 5.
Outputs: results/final_main_table*.{csv,md}, results/final_cost_by_beta.*, results/final_weighted_costs.*,
         figures/final_fig1_benchmark_b5.png, figures/final_fig3_frontier.png, figures/final_fig4_opt_tree.png,
         data/processed/final/policy_actions_{val,test}.parquet
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

import final_lib as fl
import final_setup as fs
from common import C, FEATS_C, FIG, RES, plt, to_md

t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)


F, M = fs.load_strict()
q = fs.q_of(F)
itr, iva, ite = (np.flatnonzero(M[k]) for k in ("train", "val", "test"))
qv, qt = q[iva], q[ite]
DELTAS = np.arange(-50, 100.1, 2.5)


def cost(a, qq, beta, w=None):
    return float(np.average(a + beta * np.maximum(0, qq - a), weights=w))


# Each policy: {beta: (val_actions, test_actions)} with actions as fractions.
POL, COMPLEXITY, FAMILY = {}, {}, {}


def add(name, family, per_beta, complexity="-"):
    POL[name], FAMILY[name], COMPLEXITY[name] = per_beta, family, complexity


# ---------------------------------------------------------------- fixed, oracle
for t in fl.TIERS:
    add(f"Always {int(t * 100)}%", "fixed", {b: (np.full(len(iva), t), np.full(len(ite), t)) for b in fl.BETAS})
best_fixed = {b: min(fl.TIERS, key=lambda t: cost(np.full(len(iva), t), qv, b)) for b in fl.BETAS}
add("Best fixed (val)", "fixed", {b: (np.full(len(iva), best_fixed[b]), np.full(len(ite), best_fixed[b])) for b in fl.BETAS})
add("Oracle (hindsight)", "oracle", {b: (fl.oracle_actions(qv, b), fl.oracle_actions(qt, b)) for b in fl.BETAS})

# ---------------------------------------------------------------- customer-history heuristics
choices = []
h1 = {}
for b in fl.BETAS:
    mv, mt = F.sub_hist_mean.to_numpy()[iva], F.sub_hist_mean.to_numpy()[ite]
    dl = DELTAS[int(np.argmin([cost(fl.route(mv, d), qv, b) for d in DELTAS]))]
    h1[b] = (fl.route(mv, dl), fl.route(mt, dl))
    choices.append({"policy": "H1 customer mean + margin", "beta": b, "chosen on val": f"delta = {dl:+.1f} pts"})
add("H1: customer mean P95 + margin", "heuristic", h1, "1 parameter")
fr = F[["sub_hist_frac_le25", "sub_hist_frac_le50", "sub_hist_frac_le75"]].to_numpy()


def critical_ratio(rows, beta):
    tau = 1 - 1 / beta
    ok = np.column_stack([fr[rows] >= tau - 1e-12, np.ones(len(rows), bool)])
    return fl.TIERS[np.argmax(ok, axis=1)]


add("H2: customer critical-ratio rule", "heuristic", {b: (critical_ratio(iva, b), critical_ratio(ite, b)) for b in fl.BETAS},
    "0 parameters")
log("heuristics done")

# ---------------------------------------------------------------- predict-then-decide (regression on all 26 features)
LOG_COLS = ["dep_batch_size", "sub_age_days", "sub_n_prior_created", "sub_n_hist", "dep_n_prior_created", "dep_n_hist"]
X = F[FEATS_C].astype(float).copy()
for c_ in LOG_COLS:
    X[c_] = np.log1p(X[c_])
X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * F.hour / 24), np.cos(2 * np.pi * F.hour / 24)
size = F.cores.astype(int).astype(str) + "c_" + F.memory_gb.astype(int).astype(str) + "g"
X = pd.concat([X.drop(columns=["hour", "dow", "cores", "memory_gb", "mem_per_core"]),
               pd.get_dummies(size, prefix="size", drop_first=True, dtype=float),
               pd.get_dummies(F.dow, prefix="dow", drop_first=True, dtype=float)], axis=1)
mu, sd = X.iloc[itr].mean(), X.iloc[itr].std().replace(0, 1)
XR = ((X - mu) / sd).to_numpy()
Xt = F[FEATS_C].to_numpy(np.float32)
y = F.y_p95.to_numpy()

preds = {}
m = Ridge(alpha=1.0).fit(XR[itr], y[itr])
preds["Ridge"] = (m.predict(XR[iva]), m.predict(XR[ite]))
best = None
for L in (8, 32, 128, 512, 2048):
    mm = DecisionTreeRegressor(max_leaf_nodes=L, min_samples_leaf=500, random_state=fs.SEED).fit(Xt[itr], y[itr])
    r = mean_squared_error(y[iva], mm.predict(Xt[iva]))
    if best is None or r < best[0]:
        best = (r, L, mm)
preds["CART"] = (best[2].predict(Xt[iva]), best[2].predict(Xt[ite]))
cart_leaves = best[1]
m = RandomForestRegressor(n_estimators=100, min_samples_leaf=50, max_features=0.5, max_samples=300_000, n_jobs=-1,
                          random_state=fs.SEED).fit(Xt[itr], y[itr])
preds["RF"] = (m.predict(Xt[iva]), m.predict(Xt[ite]))
pred_quality = pd.DataFrame({k: {"val R2": 1 - mean_squared_error(y[iva], v[0]) / y[iva].var(),
                                 "test R2": 1 - mean_squared_error(y[ite], v[1]) / y[ite].var()}
                             for k, v in preds.items()}).T
to_md(pred_quality, RES / "final_prediction_quality", floatfmt=".3f")
log("regressions done")
cplx = {"Ridge": "linear, 50 coef.", "CART": f"{cart_leaves} leaves", "RF": "100 trees"}
for nm, (pv, pt) in preds.items():
    add(f"{nm} predict -> tier", "predict-then-decide", {b: (fl.route(pv), fl.route(pt)) for b in fl.BETAS}, cplx[nm])
    per = {}
    for b in fl.BETAS:
        dl = DELTAS[int(np.argmin([cost(fl.route(pv, d), qv, b) for d in DELTAS]))]
        per[b] = (fl.route(pv, dl), fl.route(pt, dl))
        choices.append({"policy": f"{nm} + margin", "beta": b, "chosen on val": f"delta = {dl:+.1f} pts"})
    add(f"{nm} predict + margin", "predict-then-decide", per, cplx[nm])

# ---------------------------------------------------------------- cost-aware RF (stage 5)
npz = np.load(fs.FINAL_DATA / "rf_cost_shortfall_preds.npz")
assert np.array_equal(npz["val_idx"], iva) and np.array_equal(npz["test_idx"], ite)


def rf_act(Shat, beta):
    return fl.TIERS[np.argmin(np.column_stack([fl.TIERS[:3] + beta * Shat, np.ones(len(Shat))]), axis=1)]


add("Cost-aware RF", "cost-aware RF", {b: (rf_act(npz["val"], b), rf_act(npz["test"], b)) for b in fl.BETAS}, "100 trees")

# ---------------------------------------------------------------- policy trees (stage 5)
TA_v = pd.read_parquet(fs.FINAL_DATA / "tree_actions_val.parquet")
TA_t = pd.read_parquet(fs.FINAL_DATA / "tree_actions_test.parquet")
assert np.array_equal(TA_v.index.to_numpy(), iva) and np.array_equal(TA_t.index.to_numpy(), ite)
SELT = pd.read_csv(RES / "final_tree_selected.csv")
for name in dict.fromkeys(c.split("|")[0] for c in TA_t.columns):
    fam = "OPT" if name.startswith("OPT") else "tree (greedy)"
    per = {b: (fl.TIERS[TA_v[f"{name}|b={b}"].to_numpy()], fl.TIERS[TA_t[f"{name}|b={b}"].to_numpy()]) for b in fl.BETAS}
    r5 = SELT[(SELT.policy == name) & (SELT.beta == fl.MAIN_BETA)].iloc[0]
    add(name, fam, per, f"depth {int(r5.depth)}, {int(r5.leaves)} leaves (b=5)")
pd.DataFrame(choices).to_csv(RES / "final_validation_choices.csv", index=False)

# ---------------------------------------------------------------- consolidated tables
ORDER = (["Always 100%", "Best fixed (val)", "H1: customer mean P95 + margin", "H2: customer critical-ratio rule",
          "Ridge predict -> tier", "CART predict -> tier", "RF predict -> tier", "Ridge predict + margin",
          "CART predict + margin", "RF predict + margin", "Cost-aware RF",
          "CART-MSE (26 feat.) [full]", "CART-MSE [full]", "Greedy [full]", "OPT-exact [full]",
          "CART-MSE [200k]", "Greedy [200k]", "OPT-exact [200k]", "OPT-IAI [200k]",
          "Greedy depth-4 [full]", "Greedy depth-4 [200k]", "OPT-IAI depth-4 [200k]",
          "Oracle (hindsight)"])
ORDER = [p for p in ORDER if p in POL]
long = []
for b in fl.BETAS:
    fixed_c = cost(np.full(len(ite), best_fixed[b]), qt, b)
    orac_c = cost(POL["Oracle (hindsight)"][b][1], qt, b)
    for p in ORDER:
        a = POL[p][b][1]
        ev = fl.evaluate(a, qt, betas=(b,))
        c_ = ev.pop(f"cost b={b}")
        long.append({"beta": b, "policy": p, "family": FAMILY[p], "test cost": c_,
                     "% reduction vs best fixed": 100 * (fixed_c - c_) / fixed_c,
                     "% of fixed-oracle gap closed": 100 * (fixed_c - c_) / (fixed_c - orac_c),
                     "val cost": cost(POL[p][b][0], qv, b)} | ev | {"complexity": COMPLEXITY[p]})
L = pd.DataFrame(long)
L.to_csv(RES / "final_main_table.csv", index=False)
cols = ["test cost", "% reduction vs best fixed", "% of fixed-oracle gap closed", "avg reserved", "% under-provisioned",
        "mean shortfall", "mean unused", "% at 25%", "% at 50%", "% at 75%", "% at 100%", "complexity"]
for b in fl.BETAS:
    to_md(L[L.beta == b].set_index("policy")[cols], RES / f"final_main_table_b{b}", floatfmt=".3f")
cbb = L.pivot(index="policy", columns="beta", values="test cost").loc[ORDER]
cbb.columns = [f"cost b={b}" for b in cbb.columns]
red = L.pivot(index="policy", columns="beta", values="% reduction vs best fixed").loc[ORDER]
red.columns = [f"% red. b={b}" for b in red.columns]
to_md(pd.concat([cbb, red], axis=1), RES / "final_cost_by_beta", floatfmt=".3f")
log("main tables done")

# ---------------------------------------------------------------- weighted outcomes (evaluation weights only)
WV, WT = fs.eval_weights(F, M["val"]), fs.eval_weights(F, M["test"])
wrows = []
for wname in ("vCPU-weighted", "vCPU-hour-weighted"):
    for b in fl.BETAS:
        bf = min(fl.TIERS, key=lambda t: cost(np.full(len(iva), t), qv, b, WV[wname]))
        fixed_c = cost(np.full(len(ite), bf), qt, b, WT[wname])
        orac_c = cost(POL["Oracle (hindsight)"][b][1], qt, b, WT[wname])
        for p in ORDER:
            pc = fixed_c if p == "Best fixed (val)" else cost(POL[p][b][1], qt, b, WT[wname])
            wrows.append({"weighting": wname, "beta": b, "policy": p, "test cost": pc,
                          "% reduction vs best fixed": 100 * (fixed_c - pc) / fixed_c,
                          "% of gap closed": 100 * (fixed_c - pc) / (fixed_c - orac_c),
                          "best fixed (weighted val)": f"{int(bf * 100)}%"})
W = pd.DataFrame(wrows)
W.to_csv(RES / "final_weighted_costs.csv", index=False)
WP = W.pivot_table(index="policy", columns=["weighting", "beta"], values="% reduction vs best fixed", sort=False)
to_md(WP.loc[[p for p in ORDER if p in WP.index]], RES / "final_weighted_reduction", floatfmt=".1f")

# ---------------------------------------------------------------- save all policy actions (for robustness stage)
for part, k in (("val", 0), ("test", 1)):
    pd.DataFrame({f"{p}|b={b}": POL[p][b][k].astype(np.float32) for p in ORDER for b in fl.BETAS},
                 index=iva if k == 0 else ite).to_parquet(fs.FINAL_DATA / f"policy_actions_{part}.parquet")

# ---------------------------------------------------------------- Fig 1: benchmark at beta = 5
FAM_COLOR = {"fixed": C["ink2"], "heuristic": C["violet"], "predict-then-decide": C["orange"],
             "cost-aware RF": C["aqua"], "tree (greedy)": "#86b6ef", "OPT": C["blue"], "oracle": C["ink"]}
show = [p for p in ["Always 100%", "Best fixed (val)", "H2: customer critical-ratio rule",
                    "H1: customer mean P95 + margin", "RF predict -> tier", "Ridge predict + margin",
                    "RF predict + margin", "CART-MSE (26 feat.) [full]", "CART-MSE [full]", "Greedy [full]",
                    "OPT-IAI [200k]", "OPT-exact [full]", "Cost-aware RF", "Oracle (hindsight)"] if p in POL]
d5 = L[L.beta == fl.MAIN_BETA].set_index("policy").loc[show]
fig, ax = plt.subplots(figsize=(8.6, 5.0))
yy = np.arange(len(show))[::-1]
ax.barh(yy, d5["test cost"], color=[FAM_COLOR[FAMILY[p]] for p in show], height=0.62)
for yi, p in zip(yy, show):
    v, r = d5.loc[p, "test cost"], d5.loc[p, "% reduction vs best fixed"]
    ax.text(v + 0.01, yi, f"{v:.3f}" + (f"  ({r:.1f}% saving)" if p not in ("Best fixed (val)", "Always 100%") else ""),
            va="center", color=C["ink"], fontsize=8.5)
ax.set_yticks(yy, show)
ax.set_xlim(0, 1.5)
ax.set_xlabel("mean modeled cost per test VM, beta = 5 (lower is better; saving = cost reduction vs best fixed)")
ax.set_title("Out-of-sample modeled allocation cost, strict chronological test (days 25-29, 426,846 VMs)")
ax.grid(axis="y", visible=False)
handles = [plt.Rectangle((0, 0), 1, 1, color=FAM_COLOR[k]) for k in FAM_COLOR]
ax.legend(handles, list(FAM_COLOR), frameon=False, fontsize=8, loc="lower right")
fig.savefig(FIG / "final_fig1_benchmark_b5.png")
plt.close(fig)

# ---------------------------------------------------------------- Fig 3: capacity vs under-provisioning frontier
binner_js = json.loads((fs.FINAL_DATA / "binner.json").read_text())
Xb = np.load(fs.FINAL_DATA / "Xb_opt.npy")
binner = fl.Binner(binner_js["max_bins"])
binner.names, binner.thresholds = binner_js["names"], [np.array(t) for t in binner_js["thresholds"]]
binner.n_bins = np.array([len(t) + 1 for t in binner.thresholds])
P, S, B = fl.aggregate(Xb[itr], q[itr], binner.n_bins)
r5 = SELT[(SELT.policy == "OPT-exact [full]") & (SELT.beta == fl.MAIN_BETA)].iloc[0]
SWEEP = [1.25, 1.5, 2, 3, 4, 5, 7, 10, 15, 25, 50]
opt_sweep = fl.solve_opt_exact(P, S, B, int(r5.max_depth), [(b, int(r5.min_leaf)) for b in SWEEP])
gr5 = SELT[(SELT.policy == "Greedy [full]") & (SELT.beta == fl.MAIN_BETA)].iloc[0]
gr_sweep = [fl.fit_greedy_tree(P, S, B, b, int(gr5.max_depth), int(gr5.min_leaf)) for b in SWEEP]
log("frontier sweeps done")
fig, ax = plt.subplots(figsize=(7.4, 4.6))
fx = [fl.evaluate(np.full(len(ite), t), qt) for t in fl.TIERS]
ax.plot([f["% under-provisioned"] for f in fx], [f["avg reserved"] for f in fx], color=C["ink2"], ls="--", lw=1.2,
        marker="s", ms=6, label="fixed reservations (and their mixtures)")
for t, f in zip(fl.TIERS, fx):
    ax.annotate(f"always {int(t * 100)}%", (f["% under-provisioned"], f["avg reserved"]),
                xytext=(-64 if t == 0.25 else 6, 4), textcoords="offset points", color=C["ink2"], fontsize=8)
rf_ev = [fl.evaluate(rf_act(npz["test"], b), qt) for b in np.geomspace(1.1, 60, 30)]
ax.plot([e["% under-provisioned"] for e in rf_ev], [e["avg reserved"] for e in rf_ev], color=C["aqua"], lw=2,
        label="cost-aware RF (sweep beta)")
for res, colr, lab, mk in [(gr_sweep, "#86b6ef", f"greedy policy tree, depth {int(gr5.max_depth)} (sweep beta)", "o"),
                           (opt_sweep, C["blue"], f"OPT-exact, depth {int(r5.max_depth)} (sweep beta)", "D")]:
    ev = [fl.evaluate(fl.TIERS[fl.predict_tree(r["tree"], Xb[ite])], qt) for r in res]
    ax.plot([e["% under-provisioned"] for e in ev], [e["avg reserved"] for e in ev], color=colr, lw=1.6, marker=mk,
            ms=5, label=lab)
    i5 = SWEEP.index(5)
    if mk == "D":
        ax.annotate("beta = 5", (ev[i5]["% under-provisioned"], ev[i5]["avg reserved"]), xytext=(-14, -22),
                    textcoords="offset points", fontsize=8, color=C["ink"])
h1_ev = [fl.evaluate(fl.route(F.sub_hist_mean.to_numpy()[ite], d), qt) for d in np.arange(-30, 80, 2.5)]
ax.plot([e["% under-provisioned"] for e in h1_ev], [e["avg reserved"] for e in h1_ev], color=C["violet"], lw=1.4,
        label="H1 customer mean + margin (sweep margin)")
o = fl.evaluate(fl.TIERS[np.searchsorted(fl.TIERS, qt - 1e-12)], qt)
ax.plot(o["% under-provisioned"], o["avg reserved"], marker="*", ms=13, color=C["ink"], ls="none",
        label="oracle (smallest sufficient tier)")
ax.set_xlabel("% of test VMs under-provisioned (P95 > reservation)")
ax.set_ylabel("average reserved capacity (fraction of VM)")
ax.set_title("Capacity vs. under-provisioning on the test period (down-left is better)")
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
fig.savefig(FIG / "final_fig3_frontier.png")
plt.close(fig)

# ---------------------------------------------------------------- Fig 4: the learned OPT (full data, beta = 5)
tree = fl.tree_from_json(json.loads((fs.MODELS / f"OPT-exact_full_b{fl.MAIN_BETA}.json").read_text()), binner)
leaf_of = np.zeros(len(ite), dtype=np.int64)
TIER_COL = {0: "#86b6ef", 1: "#3987e5", 2: "#1c5cab", 3: "#0d366b"}


def layout(node, depth=0, pos=[0]):
    if node["leaf"]:
        node["_x"] = pos[0]
        pos[0] += 1
    else:
        layout(node["left"], depth + 1, pos)
        layout(node["right"], depth + 1, pos)
        node["_x"] = (node["left"]["_x"] + node["right"]["_x"]) / 2
    node["_d"] = depth


def mark(node, idx):
    node["_n"], node["_q"] = len(idx), qt[idx].mean() if len(idx) else np.nan
    if not node["leaf"]:
        go = Xb[ite][idx, node["feature"]] <= node["t"]
        mark(node["left"], idx[go])
        mark(node["right"], idx[~go])


layout(tree, 0, [0])
mark(tree, np.arange(len(ite)))
nleaf = fl.tree_leaves(tree)
fig, ax = plt.subplots(figsize=(max(9, 1.9 * nleaf), 5.2))
ax.axis("off")


def draw(node):
    x, yv = node["_x"], -node["_d"]
    if not node["leaf"]:
        for child, lab in ((node["left"], "yes"), (node["right"], "no")):
            ax.plot([x, child["_x"]], [yv - 0.18, -child["_d"] + 0.22], color=C["ink2"], lw=1)
            ax.text((x + child["_x"]) / 2, (yv - child["_d"]) / 2, lab, fontsize=8, color=C["ink2"], ha="center",
                    bbox=dict(fc=C["surface"], ec="none", pad=0.5))
            draw(child)
        thr = binner.threshold_value(node["feature"], node["t"])
        txt = f"{fs.LABELS[binner.names[node['feature']]]}\n<= {thr:.3g} ?\n{100 * node['_n'] / len(ite):.0f}% of test VMs"
        ax.text(x, yv, txt, ha="center", va="center", fontsize=8.5, color=C["ink"],
                bbox=dict(boxstyle="round,pad=0.4", fc="#f1f0ec", ec=C["ink2"], lw=0.8))
    else:
        a = node["action"]
        txt = (f"reserve {int(fl.TIERS[a] * 100)}%\n{100 * node['_n'] / len(ite):.1f}% of test VMs\n"
               f"mean P95 {100 * node['_q']:.0f}%")
        ax.text(x, yv, txt, ha="center", va="center", fontsize=8.5, color="white" if a >= 1 else C["ink"],
                bbox=dict(boxstyle="round,pad=0.45", fc=TIER_COL[a], ec="none"))


draw(tree)
ax.set_xlim(-0.7, nleaf - 0.3)
ax.set_ylim(-fl.tree_depth(tree) - 0.6, 0.6)
ax.set_title(f"Globally optimal policy tree (exact DP, depth {fl.tree_depth(tree)}, beta = 5, trained on 1.70M "
             "strict-train VMs); shares and mean P95 on the test period", fontsize=10)
fig.savefig(FIG / "final_fig4_opt_tree.png")
plt.close(fig)

print(L[L.beta == fl.MAIN_BETA].set_index("policy")[cols[:7]].to_string())
print(pd.concat([cbb, red], axis=1).to_string())
print(WP.to_string())
log("Stage 6 done")
