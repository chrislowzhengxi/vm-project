"""Stage 5 (final project): policy trees under the strict chronological setup.

1. Cost-aware Random Forest on all 26 features (black-box benchmark); its validation permutation importance selects
   the compact feature set used by every tree (top 8; no test data involved).
2. One candidate split set (training quantile thresholds, <= 32 bins per feature) shared by all tree methods.
3. Tree methods, each tuned on the strict validation set over depth {1,2,3} x min-leaf {0.1,0.5,1,2,5}% of the
   training rows, separately for beta in {2, 5, 10}:
     CART-MSE (26 feat.) - feasibility-study policy tree: multi-output regression tree on the cost vectors
     CART-MSE            - same, restricted to the binned 8-feature split set (isolates the training objective)
     Greedy              - objective-aligned greedy tree (split = largest one-step drop in modeled cost)
     OPT-exact           - globally optimal tree by exhaustive dynamic programming (certified over the split set)
     OPT-IAI             - Interpretable AI OptimalTreePolicyMinimizer (local search; 200k-row sample only)
   Full training set (1.70M strict rows) and an equal 200k-row sample (seed 0) for the like-for-like comparison.
Outputs: results/final_feature_selection.*, results/final_tree_grid.csv, results/final_tree_selected.*,
         results/final_models/*.json|txt, data/processed/final/{Xb_opt.npy, binner.json, tree_actions.parquet,
         rf_cost_shortfall_preds.npz}
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import final_lib as fl
import final_setup as fs
from common import RES, to_md

t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:7.0f}s] {msg}", flush=True)


F, M = fs.load_strict()
q = fs.q_of(F)
itr, iva, ite = (np.flatnonzero(M[k]) for k in ("train", "val", "test"))
log(f"strict rows: train {len(itr):,}  val {len(iva):,}  test {len(ite):,}")
Cval = {b: fl.cost_matrix(q[iva], b) for b in fl.BETAS}
Ctest = {b: fl.cost_matrix(q[ite], b) for b in fl.BETAS}


def act_cost(actions, Cm):
    return float(Cm[np.arange(len(actions)), actions].mean())


# ---------------------------------------------------------------- 1. cost-aware RF + validation feature selection
Xc = F[fs.OPT_CANDIDATES].to_numpy(np.float32)
rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=50, max_features=0.5, max_samples=300_000,
                           n_jobs=-1, random_state=fs.SEED).fit(Xc[itr], fl.shortfall_stats(q[itr])[:, 1:])
Sv, St = rf.predict(Xc[iva]), rf.predict(Xc[ite])
np.savez(fs.FINAL_DATA / "rf_cost_shortfall_preds.npz", val=Sv, test=St, val_idx=iva, test_idx=ite)
log("cost-aware RF fitted")


def rf_actions(Shat, beta):
    exp_cost = np.column_stack([fl.TIERS[:3] + beta * Shat, np.ones(len(Shat))])
    return np.argmin(exp_cost, axis=1)


rng = np.random.default_rng(fs.SEED)
vs = rng.choice(len(iva), 100_000, replace=False)
Xv, Cv5 = Xc[iva][vs], Cval[fl.MAIN_BETA][vs]
base = act_cost(rf_actions(rf.predict(Xv), fl.MAIN_BETA), Cv5)
imp = []
for j, f in enumerate(fs.OPT_CANDIDATES):
    d = []
    for _ in range(2):
        Xp = Xv.copy()
        Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
        d.append(act_cost(rf_actions(rf.predict(Xp), fl.MAIN_BETA), Cv5) - base)
    imp.append({"feature": f, "label": fs.LABELS[f], "val cost increase when permuted (beta=5)": np.mean(d)})
imp = pd.DataFrame(imp).sort_values("val cost increase when permuted (beta=5)", ascending=False).reset_index(drop=True)
SEL = imp.feature.iloc[:fs.N_OPT_FEATURES].tolist()
imp["selected for trees"] = imp.feature.isin(SEL)
to_md(imp.set_index("feature"), RES / "final_feature_selection", floatfmt=".4f")
log(f"selected features: {SEL}")

# ---------------------------------------------------------------- 2. shared candidate split set
binner = fl.Binner(fs.MAX_BINS).fit(F.loc[M["train"], SEL].to_numpy(float), SEL)
Xb = binner.transform(F[SEL].to_numpy(float))
np.save(fs.FINAL_DATA / "Xb_opt.npy", Xb)
(fs.FINAL_DATA / "binner.json").write_text(json.dumps({"names": SEL, "max_bins": fs.MAX_BINS,
                                                        "thresholds": [t.tolist() for t in binner.thresholds]}))
log(f"bins per feature: {dict(zip(SEL, binner.n_bins.tolist()))}")

# ---------------------------------------------------------------- 3. tree grids
train_sets = {"full": itr, "200k": fs.sample_indices(itr, 200_000, fs.SEED)}
grid, trees = [], {}


def record(method, tset, beta, depth, frac, m, tree, train_cost, runtime, status, gap=np.nan, val_act=None,
           test_act=None, n_leaves=None, depth_act=None, extra=None):
    if val_act is None:
        val_act, test_act = fl.predict_tree(tree, Xb[iva]), fl.predict_tree(tree, Xb[ite])
    row = {"method": method, "train_set": tset, "beta": beta, "max_depth": depth, "min_leaf_frac": frac,
           "min_leaf": m, "train_cost": train_cost, "val_cost": act_cost(val_act, Cval[beta]),
           "test_cost": act_cost(test_act, Ctest[beta]), "leaves": n_leaves or fl.tree_leaves(tree),
           "depth": depth_act if depth_act is not None else fl.tree_depth(tree), "runtime_s": runtime,
           "status": status, "gap": gap, "signature": fl.tree_signature(tree) if tree is not None else ""}
    trees[len(grid)] = (tree, val_act, test_act)      # keyed by row number of the grid table
    grid.append(row | (extra or {}))


def sk_tree_to_policy(sk):
    t = sk.tree_

    def rec(i):
        if t.children_left[i] == -1:
            return fl.leaf_node(int(np.argmin(t.value[i][:, 0])))
        return fl.split_node(int(t.feature[i]), int(np.floor(t.threshold[i])), rec(t.children_left[i]),
                             rec(t.children_right[i]))
    return rec(0)


for tset, idx in train_sets.items():
    P, S, B = fl.aggregate(Xb[idx], q[idx], binner.n_bins)
    n = len(idx)
    leafs = {f: max(1, int(round(f * n))) for f in fs.MIN_LEAF_FRACS}
    log(f"[{tset}] n={n:,}, unique patterns={len(P):,}")
    # OPT-exact (depth 3 shares one histogram pass across all configs)
    for depth in fs.DEPTHS:
        cfgs = [(b, m) for b in fl.BETAS for m in leafs.values()]
        for r in fl.solve_opt_exact(P, S, B, depth, cfgs):
            frac = [f for f, m in leafs.items() if m == r["min_leaf"]][0]
            record("OPT-exact", tset, r["beta"], depth, frac, r["min_leaf"], r["tree"], r["train_mean_cost"],
                   r["runtime_s"], "OPTIMAL", 0.0)
        log(f"[{tset}] OPT-exact depth {depth} done")
    # Greedy (depth <= 3 primary; depth 4 reported as an extension)
    for depth in fs.DEPTHS + (4,):
        for b in fl.BETAS:
            for frac, m in leafs.items():
                r = fl.fit_greedy_tree(P, S, B, b, depth, m)
                record("Greedy", tset, b, depth, frac, m, r["tree"], r["train_mean_cost"], r["runtime_s"], "greedy")
    log(f"[{tset}] greedy done")
    # CART-MSE on the same binned features and leaf sizes
    for depth in fs.DEPTHS:
        for b in fl.BETAS:
            Ctr = fl.cost_matrix(q[idx], b)
            for frac, m in leafs.items():
                t0 = time.perf_counter()
                sk = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=m, random_state=fs.SEED).fit(Xb[idx], Ctr)
                rt = time.perf_counter() - t0
                tree = sk_tree_to_policy(sk)
                record("CART-MSE", tset, b, depth, frac, m, tree, fl.tree_train_cost(tree, P, S, b) / n, rt, "greedy (MSE)")
    log(f"[{tset}] CART-MSE done")

# Feasibility-study policy tree: CART-MSE on all 26 unbinned features, depth 3, min_samples_leaf = 500
for b in fl.BETAS:
    t0 = time.perf_counter()
    sk = DecisionTreeRegressor(max_depth=3, min_samples_leaf=500, random_state=fs.SEED).fit(Xc[itr], fl.cost_matrix(q[itr], b))
    rt = time.perf_counter() - t0
    va_a, te_a = sk.predict(Xc[iva]).argmin(1), sk.predict(Xc[ite]).argmin(1)
    tr_cost = act_cost(sk.predict(Xc[itr]).argmin(1), fl.cost_matrix(q[itr], b))
    record("CART-MSE (26 feat.)", "full", b, 3, np.nan, 500, None, tr_cost, rt, "greedy (MSE)", val_act=va_a,
           test_act=te_a, n_leaves=sk.get_n_leaves(), depth_act=sk.get_depth())
log("feasibility CART done")

# OPT-IAI on the 200k sample (full-data runs abort inside IAI 3.2.2: "no descent" assertion; see FINAL_FINDINGS)
if fl.iai_available():
    idx = train_sets["200k"]
    P, S, B = fl.aggregate(Xb[idx], q[idx], binner.n_bins)
    leafs = {f: max(1, int(round(f * len(idx)))) for f in fs.MIN_LEAF_FRACS}
    for b in fl.BETAS:
        cfgs = [{"depth": d, "minbucket": m, "seed": 1, "frac": f} for d in (2, 3, 4) for f, m in leafs.items()]
        out = fl.run_iai_policy_trees(Xb[idx], fl.cost_matrix(q[idx], b), cfgs, fs.FINAL_DATA / f"iai_b{b}", threads=10)
        for o in out:
            record("OPT-IAI", "200k", b, o["depth"], o["frac"], o["minbucket"], o["tree"],
                   fl.tree_train_cost(o["tree"], P, S, b) / len(idx), o["runtime_s"], "local search (no certificate)")
        log(f"OPT-IAI beta={b} done")
else:
    log("IAI not available: OPT-IAI skipped")

G = pd.DataFrame(grid)
G.to_csv(RES / "final_tree_grid.csv", index=False)

# ---------------------------------------------------------------- 4. validation selection (depth <= 3 primary)
sel_rows, actions = [], {}
for (method, tset, b), g in G[G.max_depth <= 3].groupby(["method", "train_set", "beta"], sort=False):
    r = g.loc[g.val_cost.idxmin()]
    tree, va_a, te_a = trees[g.val_cost.idxmin()]
    name = f"{method} [{tset}]"
    sel_rows.append(r.to_dict() | {"policy": name})
    actions[(name, b, "val")], actions[(name, b, "test")] = va_a, te_a
    if tree is not None:
        stem = f"{method.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')}_{tset}_b{b}"
        (fs.MODELS / f"{stem}.json").write_text(json.dumps(fl.tree_to_json(tree, binner), indent=1))
        (fs.MODELS / f"{stem}.txt").write_text(fl.tree_to_text(tree, binner))
for (method, tset, b), g in G[(G.max_depth == 4)].groupby(["method", "train_set", "beta"], sort=False):
    r = g.loc[g.val_cost.idxmin()]
    tree, va_a, te_a = trees[g.val_cost.idxmin()]
    name = f"{method} depth-4 [{tset}]"
    sel_rows.append(r.to_dict() | {"policy": name})
    actions[(name, b, "val")], actions[(name, b, "test")] = va_a, te_a
sel = pd.DataFrame(sel_rows)
to_md(sel.set_index(["policy", "beta"])[["max_depth", "min_leaf_frac", "min_leaf", "depth", "leaves", "train_cost",
                                          "val_cost", "test_cost", "runtime_s", "status"]],
      RES / "final_tree_selected", floatfmt=".4f")

A = {}
for (name, b, part), a in actions.items():
    A.setdefault(part, {})[f"{name}|b={b}"] = a.astype(np.int8)
pd.DataFrame(A["val"], index=iva).to_parquet(fs.FINAL_DATA / "tree_actions_val.parquet")
pd.DataFrame(A["test"], index=ite).to_parquet(fs.FINAL_DATA / "tree_actions_test.parquet")
print(sel.set_index(["policy", "beta"])[["max_depth", "min_leaf_frac", "leaves", "train_cost", "val_cost",
                                          "test_cost"]].to_string())
log("Stage 5 done")
