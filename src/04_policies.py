"""Stage 4: does a right-sizing problem exist, and do individualized reservations beat fixed ones?

Units. q_i = p95maxcpu_i / 100 is the VM's P95 CPU demand as a FRACTION OF ITS OWN ALLOCATED vCPUs, and a reservation
a in {0.25, 0.5, 0.75, 1.0} is the fraction of those vCPUs we physically reserve. Both live in [0, 1], so
    Cost_i(a) = a + lambda * max(0, q_i - a)
is measured in "allocated-VM capacity units": reserving the full VM costs 1, and each unit of unmet P95 demand
costs lambda. A 100% reservation never falls short (p95maxcpu <= 100 in the data). With lambda <= 1 the cheapest
action is always 25% (a + max(0, q - a) = max(a, q)), so only lambda > 1 makes the problem non-trivial.

Policies compared on the TEST period (any tuning uses VALIDATION only):
  fixed 25/50/75/100; best fixed (chosen on validation per lambda); oracle (smallest tier >= q);
  naive threshold routing on predicted P95 (Ridge / CART / RF);
  margin-tuned routing: smallest tier >= (pred + delta), delta tuned on validation per lambda;
  direct cost-aware policy: multi-output RF / CART predicting E[max(0, q - a)] for each tier, then argmin of
  a + lambda * predicted shortfall (one model serves every lambda);
  depth-3 policy tree: regression tree on the lambda=5 cost vectors, leaf action = argmin mean leaf cost.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree

from common import C, FEATS_C, FIG, LAMBDAS, PROC, RES, SEED, TIERS, impute, load_features, plt, to_md

t0 = time.time()
F = impute(load_features())
P = pd.read_parquet(PROC / "predictions.parquet")
tr, va, te = ((F.split == s).to_numpy() for s in ("train", "val", "test"))
q_all = F.y_p95.to_numpy() / 100
qv, qt = q_all[va], q_all[te]
Pv, Pt = P[P.split == "val"], P[P.split == "test"]
assert np.allclose(Pv.y_p95.to_numpy() / 100, qv) and np.allclose(Pt.y_p95.to_numpy() / 100, qt)
w_cores_t = F.loc[te, "cores"].to_numpy()

# ---------------------------------------------------------------- Step 8: is there a right-sizing problem?
tier_of = np.searchsorted(TIERS, q_all - 1e-12)          # index of the smallest sufficient tier
shares = pd.DataFrame({sp: pd.Series(tier_of[(F.split == sp).to_numpy()]).value_counts(normalize=True).sort_index()
                       for sp in ("train", "val", "test")})
shares["all new VMs"] = pd.Series(tier_of).value_counts(normalize=True).sort_index()
shares.index = ["P95 <= 25%", "25% < P95 <= 50%", "50% < P95 <= 75%", "P95 > 75%"]
to_md(100 * shares, RES / "04_tier_shares_pct", floatfmt=".1f")


# ---------------------------------------------------------------- evaluation helpers
def evaluate(a, q, w=None):
    w = np.ones_like(q) if w is None else w
    sf, ws = np.maximum(0, q - a), np.maximum(0, a - q)
    r = {"avg reserved": np.average(a, weights=w), "% under-provisioned": 100 * np.average(q > a + 1e-12, weights=w),
         "mean shortfall": np.average(sf, weights=w), "mean waste": np.average(ws, weights=w)}
    for lam in LAMBDAS:
        r[f"cost l={lam}"] = np.average(a + lam * sf, weights=w)
    for t_ in TIERS:
        r[f"share {int(t_ * 100)}%"] = 100 * np.mean(np.isclose(a, t_))
    return r


def route(pred_pct, delta=0.0):
    """Smallest tier >= (prediction + delta), predictions in percent."""
    return TIERS[np.minimum(np.searchsorted(TIERS, (pred_pct + delta) / 100 - 1e-12), len(TIERS) - 1).clip(0)]


def cost(a, q, lam):
    return np.mean(a + lam * np.maximum(0, q - a))


def oracle_action(q, lam):
    """Hindsight cost-minimizing tier per VM (equals the smallest sufficient tier only when lambda is large)."""
    return TIERS[np.argmin(TIERS[None, :] + lam * np.maximum(0, q[:, None] - TIERS[None, :]), axis=1)]


DELTAS = np.arange(-50, 100.1, 2.5)
pred_cols = {"Ridge": "pred_Ridge_C", "CART": "pred_CART_C", "RF": "pred_Random_Forest_C"}

# ---------------------------------------------------------------- direct cost-aware models (predict shortfall per tier)
Xtr, Xva, Xte = (F.loc[m, FEATS_C].to_numpy(np.float32) for m in (tr, va, te))
S_tr = np.maximum(0, q_all[tr][:, None] - TIERS[None, :3])        # shortfall at 25/50/75 (100% never short)
rf_dir = RandomForestRegressor(n_estimators=100, min_samples_leaf=50, max_features=0.5, max_samples=300_000,
                               n_jobs=-1, random_state=SEED).fit(Xtr, S_tr)
Sv_rf, St_rf = rf_dir.predict(Xva), rf_dir.predict(Xte)
print(f"[{time.time() - t0:5.0f}s] direct RF fitted", flush=True)
cart_dir = {L: DecisionTreeRegressor(max_leaf_nodes=L, min_samples_leaf=500, random_state=SEED).fit(Xtr, S_tr)
            for L in (8, 32, 128, 512, 2048)}


def direct_action(S_hat, lam):
    exp_cost = np.column_stack([TIERS[:3] + lam * S_hat, np.ones(len(S_hat))])
    return TIERS[np.argmin(exp_cost, axis=1)]


# ---------------------------------------------------------------- Step 9: tradeoff table (no cost function needed)
rows = {}
for t_ in TIERS:
    rows[f"Always {int(t_ * 100)}%"] = evaluate(np.full(len(qt), t_), qt)
for nm, col in pred_cols.items():
    rows[f"Threshold routing on predicted P95 ({nm})"] = evaluate(route(Pt[col].to_numpy()), qt)
rows["Oracle (smallest sufficient tier)"] = evaluate(TIERS[tier_of[te]], qt)
t9 = pd.DataFrame(rows).T
to_md(t9[["avg reserved", "% under-provisioned", "mean shortfall", "mean waste", "share 25%", "share 50%", "share 75%",
          "share 100%"]], RES / "04_policy_tradeoff_table")

# ---------------------------------------------------------------- Step 10: lambda costs (policies tuned on validation)
cost_rows, choice_rows, actions_lam5, best_fixed = [], [], {}, {}
for lam in LAMBDAS:
    r = {"lambda": lam}
    fixed_val = {t_: cost(np.full(len(qv), t_), qv, lam) for t_ in TIERS}
    a_fix = best_fixed[lam] = min(fixed_val, key=fixed_val.get)
    r["Best fixed (val)"] = cost(np.full(len(qt), a_fix), qt, lam)
    choice_rows.append({"lambda": lam, "policy": "Best fixed", "chosen on val": f"{int(a_fix * 100)}%"})
    for nm, col in pred_cols.items():
        r[f"{nm} naive threshold"] = cost(route(Pt[col].to_numpy()), qt, lam)
        dv = [cost(route(Pv[col].to_numpy(), dl), qv, lam) for dl in DELTAS]
        dl = DELTAS[int(np.argmin(dv))]
        a = route(Pt[col].to_numpy(), dl)
        r[f"{nm} + margin (val-tuned)"] = cost(a, qt, lam)
        choice_rows.append({"lambda": lam, "policy": f"{nm} + margin", "chosen on val": f"delta={dl:+.1f} pts"})
        if lam == 5:
            actions_lam5[f"{nm} + margin"] = a
    L = min(cart_dir, key=lambda L_: cost(direct_action(cart_dir[L_].predict(Xva), lam), qv, lam))
    a = direct_action(cart_dir[L].predict(Xte), lam)
    r["Direct cost-aware CART"] = cost(a, qt, lam)
    choice_rows.append({"lambda": lam, "policy": "Direct CART", "chosen on val": f"{L} leaves"})
    a = direct_action(St_rf, lam)
    r["Direct cost-aware RF"] = cost(a, qt, lam)
    if lam == 5:
        actions_lam5["Direct RF"] = a
        actions_lam5["Direct CART"] = direct_action(cart_dir[L].predict(Xte), lam)
    # depth-3 greedy policy tree on this lambda's cost vectors
    G_tr = TIERS[None, :] + lam * np.maximum(0, q_all[tr][:, None] - TIERS[None, :])
    pt3 = DecisionTreeRegressor(max_depth=3, min_samples_leaf=500, random_state=SEED).fit(Xtr, G_tr)
    a = TIERS[np.argmin(pt3.predict(Xte), axis=1)]
    r["Depth-3 policy tree"] = cost(a, qt, lam)
    if lam == 5:
        pt3_lam5, actions_lam5["Depth-3 policy tree"] = pt3, a
    r["Oracle (hindsight cost-optimal)"] = cost(oracle_action(qt, lam), qt, lam)
    cost_rows.append(r)
    print(f"[{time.time() - t0:5.0f}s] lambda={lam} done", flush=True)

ct = pd.DataFrame(cost_rows).set_index("lambda")
to_md(ct.T, RES / "04_policy_costs_by_lambda", floatfmt=".4f")
to_md(pd.DataFrame(choice_rows).set_index(["lambda", "policy"]), RES / "04_validation_choices")
ORC = "Oracle (hindsight cost-optimal)"
gap = pd.DataFrame({lam: {p: 100 * (ct.loc[lam, "Best fixed (val)"] - ct.loc[lam, p]) /
                          (ct.loc[lam, "Best fixed (val)"] - ct.loc[lam, ORC])
                          if ct.loc[lam, "Best fixed (val)"] - ct.loc[lam, ORC] > 1e-9 else np.nan
                          for p in ct.columns if p not in ("Best fixed (val)", ORC)} for lam in LAMBDAS})
to_md(gap, RES / "04_pct_of_gap_to_oracle_closed", floatfmt=".1f")
imp = pd.DataFrame({lam: {p: 100 * (ct.loc[lam, "Best fixed (val)"] - ct.loc[lam, p]) / ct.loc[lam, "Best fixed (val)"]
                          for p in ct.columns if p != "Best fixed (val)"} for lam in LAMBDAS})
to_md(imp, RES / "04_pct_cost_reduction_vs_best_fixed", floatfmt=".1f")

# Operating characteristics of the lambda=5 policies (action mix = does the policy personalize at all?)
lam5 = pd.DataFrame({k: evaluate(v, qt) for k, v in actions_lam5.items()}).T
lam5.loc["Best fixed (val)"] = evaluate(np.full(len(qt), best_fixed[5]), qt)
lam5.loc["Oracle (cost-optimal, lambda=5)"] = evaluate(oracle_action(qt, 5), qt)
to_md(lam5, RES / "04_lambda5_policy_details")

# Cluster bootstrap over subscriptions: uncertainty of (best fixed - direct RF) cost difference on test
rng = np.random.default_rng(SEED)
subs_t = F.loc[te, "key_sub"].to_numpy()
codes, inv = np.unique(subs_t, return_inverse=True)
boot = []
for lam in LAMBDAS:
    a_fix = best_fixed[lam]
    diff = (a_fix + lam * np.maximum(0, qt - a_fix)) - (lambda a: a + lam * np.maximum(0, qt - a))(direct_action(St_rf, lam))
    sd, n_ = np.bincount(inv, diff), np.bincount(inv)
    reps = []
    for _ in range(500):
        s = rng.integers(0, len(codes), len(codes))
        reps.append(sd[s].sum() / n_[s].sum())
    boot.append({"lambda": lam, "mean cost saving vs best fixed": diff.mean(),
                 "95% CI low (sub-cluster bootstrap)": np.percentile(reps, 2.5),
                 "95% CI high": np.percentile(reps, 97.5),
                 "% saving": 100 * diff.mean() / np.mean(a_fix + lam * np.maximum(0, qt - a_fix))})
to_md(pd.DataFrame(boot).set_index("lambda"), RES / "04_bootstrap_direct_rf_vs_best_fixed", floatfmt=".4f")

# Weighted robustness: cost per allocated vCPU (weight = cores) and per vCPU-hour (weight = cores x lifetime).
# Lifetime is used ONLY as an evaluation weight here; policies are unchanged, best fixed is re-chosen on val.
life_w = lambda m: F.loc[m, "cores"].to_numpy() * np.maximum(F.loc[m, "post_lifetime_h"].to_numpy(), 5 / 60)
cw_tabs = []
for wname, wv, wt in [("vCPU-weighted", F.loc[va, "cores"].to_numpy(), w_cores_t),
                      ("vCPU-hour-weighted", life_w(va), life_w(te))]:
    cw = {}
    for lam in LAMBDAS:
        fv = {t_: np.average(t_ + lam * np.maximum(0, qv - t_), weights=wv) for t_ in TIERS}
        a_fix = min(fv, key=fv.get)
        cw[lam] = {"Best fixed (val, same weighting)": evaluate(np.full(len(qt), a_fix), qt, wt)[f"cost l={lam}"],
                   "Direct cost-aware RF": evaluate(direct_action(St_rf, lam), qt, wt)[f"cost l={lam}"],
                   "Oracle (cost-optimal)": evaluate(oracle_action(qt, lam), qt, wt)[f"cost l={lam}"]}
    cw = pd.DataFrame(cw)
    cw.loc["% saving of direct RF"] = 100 * (cw.iloc[0] - cw.iloc[1]) / cw.iloc[0]
    cw_tabs.append(pd.concat({wname: cw}))
cw = pd.concat(cw_tabs)
to_md(cw, RES / "04_weighted_costs", floatfmt=".4f")

# ---------------------------------------------------------------- Fig 7: capacity vs under-provisioning risk
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
for ax, xm, xl in [(axes[0], "% under-provisioned", "% of VMs under-provisioned (P95 > reservation)"),
                   (axes[1], "mean shortfall", "mean shortfall (fraction of VM capacity)")]:
    fx = [evaluate(np.full(len(qt), t_), qt) for t_ in TIERS]
    ax.plot([f[xm] for f in fx], [f["avg reserved"] for f in fx], color=C["ink2"], ls="--", lw=1.2, marker="s", ms=7)
    for t_, f in zip(TIERS, fx):
        ax.annotate(f"always {int(t_ * 100)}%", (f[xm], f["avg reserved"]), xytext=(-62 if t_ == 0.25 else 6, 4),
                    textcoords="offset points",
                    color=C["ink2"], fontsize=8)
    for nm, col, colr in [("Ridge + margin", "pred_Ridge_C", C["orange"]), ("RF + margin", "pred_Random_Forest_C", C["blue"])]:
        ev = [evaluate(route(Pt[col].to_numpy(), dl), qt) for dl in np.arange(-40, 80.1, 2.5)]
        ax.plot([e[xm] for e in ev], [e["avg reserved"] for e in ev], color=colr, lw=2, label=nm)
    ev = [evaluate(direct_action(St_rf, lam), qt) for lam in np.geomspace(0.9, 60, 30)]
    ax.plot([e[xm] for e in ev], [e["avg reserved"] for e in ev], color=C["aqua"], lw=2, label="Direct cost-aware RF (sweep lambda)")
    o = evaluate(TIERS[tier_of[te]], qt)
    ax.plot(o[xm], o["avg reserved"], marker="*", ms=13, color=C["ink"], ls="none", label="oracle")
    ax.set_xlabel(xl)
    ax.set_ylabel("average reserved capacity (fraction of VM)")
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
axes[0].set_title("Capacity vs. risk (test set)")
axes[1].set_title("Capacity vs. shortfall size (test set)")
fig.text(0.5, -0.03, "Down-and-left is better. The dashed line joins fixed rules; randomizing between adjacent fixed rules "
         "traces exactly that line, so any learned curve below it is a genuine gain from personalization.",
         ha="center", color=C["ink2"], fontsize=8.5)
fig.savefig(FIG / "fig7_capacity_vs_risk.png")
plt.close(fig)

# Fig 8: depth-3 policy tree for lambda = 5
fig, ax = plt.subplots(figsize=(18, 6.5))
annos = plot_tree(pt3_lam5, feature_names=FEATS_C, impurity=False, proportion=True, precision=1, fontsize=9, ax=ax)
tree_ = pt3_lam5.tree_
node_annos = [a_ for a_ in annos if "samples" in a_.get_text()]   # drop the True/False edge labels
if len(node_annos) == tree_.node_count:     # replace the raw cost vector with the decision it implies
    for i, ann in enumerate(node_annos):
        v = tree_.value[i][:, 0]
        head = [ln for ln in ann.get_text().split("\n") if not ln.lstrip().startswith(("value", "["))]
        ann.set_text("\n".join(head + [f"reserve {int(TIERS[np.argmin(v)] * 100)}%  (cost {v.min():.2f})"]))
ax.set_title("Depth-3 greedy policy tree, lambda = 5: each node shows the reservation tier with the lowest mean "
             "training cost a + 5*max(0, q - a)")
fig.savefig(FIG / "fig8_policy_tree_lambda5.png")
plt.close(fig)
leaf = pt3_lam5.apply(Xte)
lt = pd.DataFrame({"leaf": leaf, "q": qt}).groupby("leaf").agg(n=("q", "size"), mean_p95=("q", "mean"))
lt["action"] = [int(100 * TIERS[np.argmin(pt3_lam5.tree_.value[l_][:, 0])]) for l_ in lt.index]
to_md(lt, RES / "04_policy_tree_lambda5_leaves")

print(shares.to_string())
print(t9.to_string())
print(ct.T.to_string())
print(gap.to_string())
print(lam5.to_string())
print(pd.DataFrame(boot).to_string())
print(cw.to_string())
print(lt.to_string())
print(f"Stage 4 done in {time.time() - t0:.0f}s")
