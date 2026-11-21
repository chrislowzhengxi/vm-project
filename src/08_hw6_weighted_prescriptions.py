"""Stage 8 (HW6 refinement 2, Ch. 13): distribution-aware prescriptions from Random Forest leaf weights.

One P95 regression forest, three decision rules (controlled comparison on the SAME fitted forest):
  A  point prediction -> smallest covering tier
  B  point prediction + safety margin (margin tuned on strict validation per beta) -> tier
  C  forest-weighted prescription: argmin_a sum_i w_i(x) C(a, Y_i; beta), leaf weights over all strict training rows
Forest hyperparameters match the main pipeline (100 trees, max_features 0.5, 300k-row bootstrap samples, seed 0);
min_samples_leaf in {50, 500, 2000} is the only tuned complexity knob (selected on validation per method and beta).
min_samples_leaf = 50 reproduces the main pipeline's RF exactly (asserted against stage-6 actions).
Outputs: results/hw6/hw6_forest_*.{csv,md}, figures/hw6/hw6_fig_forest_*.png, data/processed/hw6/forest_actions_*.parquet
Original stage 1-7 outputs are read, never written.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import final_lib as fl
import final_setup as fs
import hw6_lib as h6
from common import C, FEATS_C, FIG, PROC, RES, plt, to_md

OUT, FOUT, DOUT = RES / "hw6", FIG / "hw6", PROC / "hw6"
for d in (OUT, FOUT, DOUT):
    d.mkdir(parents=True, exist_ok=True)
t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)


F, M = fs.load_strict()
q = fs.q_of(F)
itr, iva, ite = (np.flatnonzero(M[k]) for k in ("train", "val", "test"))
qv, qt = q[iva], q[ite]
X = F[FEATS_C].to_numpy(np.float32)
y = F.y_p95.to_numpy()
DELTAS = np.arange(-50, 100.1, 2.5)
MSL = (50, 500, 2000)
PA_t = pd.read_parquet(fs.FINAL_DATA / "policy_actions_test.parquet")
PA_v = pd.read_parquet(fs.FINAL_DATA / "policy_actions_val.parquet")


def cost(a, qq, b, w=None):
    return float(np.average(a + b * np.maximum(0, qq - a), weights=w))


grid, acts_t, acts_v, keep = [], {}, {}, {}
for msl in MSL:
    t0 = time.perf_counter()
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=msl, max_features=0.5, max_samples=300_000,
                               n_jobs=-1, random_state=fs.SEED).fit(X[itr], y[itr])
    t_fit = time.perf_counter() - t0
    pv, pt = rf.predict(X[iva]), rf.predict(X[ite])
    t0 = time.perf_counter()
    tabs = h6.leaf_tables(rf, X[itr], q[itr])
    Sv, St = h6.weighted_shortfall(rf, tabs, X[iva]), h6.weighted_shortfall(rf, tabs, X[ite])
    t_w = time.perf_counter() - t0
    leaves = int(np.mean([e.get_n_leaves() for e in rf.estimators_]))
    log(f"min_samples_leaf={msl}: fit {t_fit:.0f}s, leaf tables + weighted costs {t_w:.0f}s, {leaves} leaves/tree")
    for b in fl.BETAS:
        dl = DELTAS[int(np.argmin([cost(fl.route(pv, d), qv, b) for d in DELTAS]))]
        rules = {"A: point prediction -> tier": (fl.route(pv), fl.route(pt), "-"),
                 "B: point prediction + margin": (fl.route(pv, dl), fl.route(pt, dl), f"delta={dl:+.1f}"),
                 "C: forest-weighted prescription": (fl.TIERS[h6.prescribe(Sv, b)], fl.TIERS[h6.prescribe(St, b)], "-")}
        for rule, (av, at, note) in rules.items():
            grid.append({"rule": rule, "min_samples_leaf": msl, "beta": b, "val cost": cost(av, qv, b),
                         "test cost": cost(at, qt, b), "note": note, "leaves/tree": leaves,
                         "fit_s": t_fit, "weights_s": t_w if rule.startswith("C") else np.nan})
            acts_t[(rule, msl, b)], acts_v[(rule, msl, b)] = at, av
        if msl == 50:   # must reproduce the main pipeline's RF policies exactly
            assert np.array_equal(acts_t[("A: point prediction -> tier", 50, b)], PA_t[f"RF predict -> tier|b={b}"])
            assert np.array_equal(acts_t[("B: point prediction + margin", 50, b)], PA_t[f"RF predict + margin|b={b}"])
    # calibration diagnostic: forest-weighted expected shortfall vs realized, on test
    real = fl.shortfall_stats(qt)[:, 1:]
    keep[msl] = {"pred_test": pt, "S_test": St, "S_val": Sv, "rf": rf if msl == 50 else None, "tabs": tabs if msl == 50 else None,
                 "calib": {f"E[shortfall @{int(t * 100)}%] weighted": St[:, k].mean() for k, t in enumerate(fl.TIERS[:3])}
                 | {f"realized shortfall @{int(t * 100)}%": real[:, k].mean() for k, t in enumerate(fl.TIERS[:3])}}
log("forest grid done")
G = pd.DataFrame(grid)
G.to_csv(OUT / "hw6_forest_grid.csv", index=False)
to_md(pd.DataFrame({m: k["calib"] for m, k in keep.items()}).T.rename_axis("min_samples_leaf"),
      OUT / "hw6_forest_calibration", floatfmt=".4f")

# ---------------------------------------------------------------- validation selection per rule and beta
sel = G.loc[G.groupby(["rule", "beta"])["val cost"].idxmin()].sort_values(["beta", "rule"])
to_md(sel.set_index(["beta", "rule"])[["min_samples_leaf", "val cost", "test cost", "note", "leaves/tree"]],
      OUT / "hw6_forest_selected", floatfmt=".4f")
POL = {}
for _, r in sel.iterrows():
    POL[(r.rule + " [val-selected forest]", r.beta)] = acts_t[(r.rule, r.min_samples_leaf, r.beta)]
for rule in ("A: point prediction -> tier", "B: point prediction + margin", "C: forest-weighted prescription"):
    for b in fl.BETAS:
        POL[(rule + " [main forest, msl=50]", b)] = acts_t[(rule, 50, b)]
REF = {"Best fixed (val)": "Best fixed (val)", "Greedy policy tree": "Greedy [full]", "OPT-exact": "OPT-exact [full]",
       "Cost-aware RF": "Cost-aware RF", "Oracle (hindsight)": "Oracle (hindsight)"}
for name, col in REF.items():
    for b in fl.BETAS:
        POL[(name + " (existing)", b)] = PA_t[f"{col}|b={b}"].to_numpy(float)

rows = []
for b in fl.BETAS:
    bf = cost(POL[("Best fixed (val) (existing)", b)], qt, b)
    orc = cost(POL[("Oracle (hindsight) (existing)", b)], qt, b)
    for (name, bb), a in POL.items():
        if bb != b:
            continue
        ev = fl.evaluate(a, qt, betas=(b,))
        c_ = ev.pop(f"cost b={b}")
        rows.append({"beta": b, "policy": name, "test cost": c_, "% reduction vs best fixed": 100 * (bf - c_) / bf,
                     "% of gap closed": 100 * (bf - c_) / (bf - orc)} | ev)
T = pd.DataFrame(rows)
T.to_csv(OUT / "hw6_forest_results.csv", index=False)
for b in fl.BETAS:
    to_md(T[T.beta == b].drop(columns="beta").set_index("policy"), OUT / f"hw6_forest_results_b{b}", floatfmt=".4f")
pd.DataFrame({f"{n}|b={b}": a.astype(np.float32) for (n, b), a in POL.items() if "existing" not in n},
             index=ite).to_parquet(DOUT / "forest_actions_test.parquet")

# ---------------------------------------------------------------- paired subscription-cluster bootstrap
subs = F.key_sub.to_numpy()[ite]
CSEL = "C: forest-weighted prescription [val-selected forest]"
pairs = [(CSEL, "A: point prediction -> tier [val-selected forest]"),
         (CSEL, "B: point prediction + margin [val-selected forest]"),
         ("C: forest-weighted prescription [main forest, msl=50]", "A: point prediction -> tier [main forest, msl=50]"),
         ("C: forest-weighted prescription [main forest, msl=50]", "B: point prediction + margin [main forest, msl=50]"),
         (CSEL, "Cost-aware RF (existing)"), (CSEL, "OPT-exact (existing)"), (CSEL, "Best fixed (val) (existing)")]
bt = []
for b in fl.BETAS:
    for a_, b_ in pairs:
        ca, cb = POL[(a_, b)] + b * np.maximum(0, qt - POL[(a_, b)]), POL[(b_, b)] + b * np.maximum(0, qt - POL[(b_, b)])
        r = fl.cluster_bootstrap_diff(ca, cb, subs, n_boot=1000, seed=fs.SEED)
        bt.append({"beta": b, "A": a_, "B": b_, "cost A": ca.mean(), "cost B": cb.mean(), "A - B": r["mean diff"],
                   "95% CI low": r["ci low"], "95% CI high": r["ci high"], "% of B": 100 * r["mean diff"] / cb.mean()})
BT = pd.DataFrame(bt)
to_md(BT.set_index(["beta", "A", "B"]), OUT / "hw6_forest_bootstrap", floatfmt=".4f")
log("bootstrap done")

# ---------------------------------------------------------------- vCPU-hour weighted sensitivity at beta = 5
WT, WV = fs.eval_weights(F, M["test"])["vCPU-hour-weighted"], fs.eval_weights(F, M["val"])["vCPU-hour-weighted"]
b = fl.MAIN_BETA
bfw = min(fl.TIERS, key=lambda t: cost(np.full(len(iva), t), qv, b, WV))
cf = cost(np.full(len(ite), bfw), qt, b, WT)
wr = [{"policy": n, "vCPU-hour-weighted cost": cost(a, qt, b, WT),
       "% reduction vs weighted best fixed": 100 * (cf - cost(a, qt, b, WT)) / cf}
      for (n, bb), a in POL.items() if bb == b and "Oracle" not in n]
to_md(pd.DataFrame(wr).set_index("policy"), OUT / "hw6_forest_vcpu_hour_b5", floatfmt=".4f")

# ---------------------------------------------------------------- illustrative example (main forest, beta = 5)
rf, tabs, pt, St = keep[50]["rf"], keep[50]["tabs"], keep[50]["pred_test"], keep[50]["S_test"]
aA, aC = fl.route(pt), fl.TIERS[h6.prescribe(St, b)]
diff = np.flatnonzero(aA != aC)
# one VM where C reserves MORE than A and one where it reserves LESS; pick the largest expected-cost gap
expA = np.where(aA < 1, aA + b * St[np.arange(len(ite)), np.clip(np.searchsorted(fl.TIERS, aA), 0, 2)], 1.0)
expC = np.where(aC < 1, aC + b * St[np.arange(len(ite)), np.clip(np.searchsorted(fl.TIERS, aC), 0, 2)], 1.0)
gain = expA - expC
ex = []
for cond in (aC > aA, aC < aA):
    cand = diff[cond[diff]]
    if len(cand):
        ex.append(cand[np.argmax(gain[cand])])
ytr = q[itr]
fig, axes = plt.subplots(1, len(ex), figsize=(5.6 * len(ex), 3.8), squeeze=False)
ex_rows = []
for ax, j in zip(axes[0], ex):
    w = h6.explicit_weights(rf, X[itr], X[ite][j])
    assert np.isclose(w.sum(), 1) and np.allclose(w @ fl.shortfall_stats(ytr)[:, 1:], St[j])   # equals leaf aggregation
    ec = w @ fl.cost_matrix(ytr, b)
    ax.hist(100 * ytr, bins=np.linspace(0, 100, 41), weights=w * 100, color="#86b6ef", edgecolor=C["surface"])
    ax.axvline(pt[j], color=C["orange"], lw=2)
    ax.text(pt[j] + 1, ax.get_ylim()[1] * 0.55, f"point\nprediction\n{pt[j]:.1f}%", color=C["orange"], fontsize=8)
    for t in (25, 50, 75):
        ax.axvline(t, color=C["ink2"], lw=0.8, ls=":")
    ax.set_xlabel("P95 CPU of weighted similar training VMs (%)")
    ax.set_ylabel("forest weight (%)")
    ax.set_title(f"A reserves {int(aA[j] * 100)}%, C reserves {int(aC[j] * 100)}%  (realized P95 {100 * qt[j]:.0f}%)",
                 fontsize=9.5)
    box = "weighted expected cost (beta=5):\n" + "\n".join(
        f"  {int(t * 100)}%: {c_:.2f}" + (" <- C" if np.isclose(t, aC[j]) else "") + (" <- A" if np.isclose(t, aA[j]) else "")
        for t, c_ in zip(fl.TIERS, ec))
    left = pt[j] >= 50          # put the box on the side with less histogram mass
    ax.text(0.02 if left else 0.98, 0.97, box, transform=ax.transAxes, va="top", ha="left" if left else "right",
            fontsize=8, family="monospace")
    ex_rows.append({"test row": int(ite[j]), "point prediction %": pt[j], "A action": aA[j], "C action": aC[j],
                    "realized P95 %": 100 * qt[j], "weights > 0": int((w > 0).sum()),
                    **{f"E[cost] {int(t * 100)}%": c_ for t, c_ in zip(fl.TIERS, ec)},
                    "weighted P(P95 > 50%)": float(w @ (ytr > 0.5)), "weighted P(P95 > 75%)": float(w @ (ytr > 0.75))})
fig.suptitle("Same forest, different decisions: point prediction vs. forest-weighted expected cost", fontsize=10.5, y=1.02)
fig.savefig(FOUT / "hw6_fig_forest_example.png")
plt.close(fig)
to_md(pd.DataFrame(ex_rows).set_index("test row"), OUT / "hw6_forest_example", floatfmt=".3f")
share_diff = {b_: float(np.mean(acts_t[("A: point prediction -> tier", 50, b_)] != acts_t[("C: forest-weighted prescription", 50, b_)]))
              for b_ in fl.BETAS}
share_diffB = {b_: float(np.mean(acts_t[("B: point prediction + margin", 50, b_)] != acts_t[("C: forest-weighted prescription", 50, b_)]))
               for b_ in fl.BETAS}
to_md(pd.DataFrame({"% test VMs where A != C": {k: 100 * v for k, v in share_diff.items()},
                    "% test VMs where B != C": {k: 100 * v for k, v in share_diffB.items()}}).rename_axis("beta"),
      OUT / "hw6_forest_decision_disagreement", floatfmt=".1f")

# ---------------------------------------------------------------- summary figure across beta
show = ["A: point prediction -> tier [val-selected forest]", "B: point prediction + margin [val-selected forest]",
        CSEL, "Cost-aware RF (existing)", "OPT-exact (existing)"]
lab = ["A: point -> tier", "B: point + margin", "C: forest-weighted", "cost-aware RF (existing)", "OPT-exact (existing)"]
cols = [C["orange"], "#f4a582", C["violet"], C["aqua"], C["blue"]]
fig, ax = plt.subplots(figsize=(8.6, 3.6))
wdt = 0.16
for j, (p, l_, c_) in enumerate(zip(show, lab, cols)):
    vals = [T[(T.beta == b_) & (T.policy == p)]["% reduction vs best fixed"].iloc[0] for b_ in fl.BETAS]
    xs = np.arange(len(fl.BETAS)) + (j - 2) * wdt
    ax.bar(xs, vals, width=wdt * 0.92, color=c_, label=l_)
    for x_, v in zip(xs, vals):
        ax.text(x_, v + (0.4 if v >= 0 else -1.6), f"{v:.1f}", ha="center", fontsize=7)
ax.axhline(0, color=C["ink2"], lw=0.8)
ax.set_xticks(range(len(fl.BETAS)), [f"beta = {b_}" for b_ in fl.BETAS])
ax.set_ylabel("% cost reduction vs best fixed (test)")
ax.set_title("Same P95 forest, three decision rules (forests tuned on validation)")
ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
ax.grid(axis="x", visible=False)
fig.savefig(FOUT / "hw6_fig_forest_rules.png")
plt.close(fig)
print(sel.to_string())
print(T[T.policy.str.contains("val-selected|existing")].to_string())
print(BT.to_string())
log("Stage 8 done")
