"""Stage 10: beta sensitivity of the reservation policies (E1-E4). Reads finalized outputs, never writes them.

E1  Fixed-policy sensitivity: every saved action vector is a line in beta (E[r] + beta E[(Y-r)+]); best-fixed tier
    switch points (train / validation / test), pairwise break-evens, leaf-level switch points of the saved OPT trees,
    and the robustness of the beta = 5 headline to the test-hindsight best fixed tier.
E2  Re-optimized sensitivity on beta in {1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20}: OPT-exact / greedy / CART-MSE
    re-solved (depth 1-3 x 5 min-leaf sizes, validation-selected per beta); beta-free models (cost-aware RF,
    forest-weighted prescription) re-decided; margins re-tuned on validation; paired cluster bootstrap per beta.
E3  Exact change points of the optimal depth-3 tree in beta (min leaf 1% and 5%) via the concave envelope V(beta).
E4  Wrong-beta regret of policies built for beta = 5 (train / validation / test), incl. best fixed and cost-aware RF.
beta is an economic input: costs at different beta are never compared to choose beta; within-beta comparisons use
% saving vs the best fixed rule chosen on validation. Test data is used only for retrospective evaluation.
Outputs: results/sensitivity/s10_*, figures/sensitivity/sens_fig{1,2,3}_*.png, data/processed/sensitivity/
"""
import argparse
import json
import pickle
import resource
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import final_lib as fl
import final_setup as fs
import hw6_lib as h6
import sensitivity_lib as sl
from common import C, FEATS_C, FIG, PROC, RES, plt, to_md

ap = argparse.ArgumentParser()
ap.add_argument("--e3-max-solves", type=int, default=150)
ap.add_argument("--e3-time-limit", type=float, default=1200, help="seconds per E3 path")
ap.add_argument("--reuse-e3", action="store_true", help="reuse cached E3 paths if present")
ap.add_argument("--e3-only", type=float, default=None, help="compute and cache only the E3 path for this min-leaf "
                "fraction (0.01 or 0.05), then exit; lets the two paths run as parallel processes")
args = ap.parse_args()

OUT, FOUT, DOUT = RES / "sensitivity", FIG / "sensitivity", PROC / "sensitivity"
for d in (OUT, FOUT, DOUT):
    d.mkdir(parents=True, exist_ok=True)
GRID = [1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
LO, HI = 1.0, 20.0
t_start = time.time()
runlog = {}


def log(msg):
    print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)


F, M = fs.load_strict()
q = fs.q_of(F)
itr, iva, ite = (np.flatnonzero(M[k]) for k in ("train", "val", "test"))
qtr, qv, qt = q[itr], q[iva], q[ite]
PA_v = pd.read_parquet(fs.FINAL_DATA / "policy_actions_val.parquet")
PA_t = pd.read_parquet(fs.FINAL_DATA / "policy_actions_test.parquet")
js = json.loads((fs.FINAL_DATA / "binner.json").read_text())
NAMES, THR = js["names"], [np.array(t) for t in js["thresholds"]]
nb = np.array([len(t) + 1 for t in THR])
Xb = np.load(fs.FINAL_DATA / "Xb_opt.npy")
P, S, B = fl.aggregate(Xb[itr], qtr, nb)
NTR = S[:, 0].sum()
subs = F.key_sub.to_numpy()[ite]
FR = {f: max(1, int(round(f * len(itr)))) for f in fs.MIN_LEAF_FRACS}


def cost(a, qq, b, w=None):
    return float(np.average(a + b * np.maximum(0, qq - a), weights=w))


def describe(tree):
    if tree["leaf"]:
        return "single leaf"
    f, t = tree["feature"], tree["t"]
    return f"{fs.LABELS[NAMES[f]]} <= {THR[f][t]:.3g}"


def load_tree(path):
    return fl.tree_from_json(json.loads(path.read_text()), type("Bn", (), {"names": NAMES})())


if args.e3_only is not None:      # standalone E3 path (cached for the main run, which then uses --reuse-e3)
    frac, m = args.e3_only, FR[args.e3_only]

    def solve(beta):
        r = fl.solve_opt_exact(P, S, B, 3, [(beta, m)])[0]
        A_, B_ = sl.tree_line(r["tree"], P, S)
        return {"A": A_, "B": B_, "tree": r["tree"], "sig": fl.tree_signature(r["tree"])}
    path = sl.parametric_path(solve, LO, HI, max_solves=args.e3_max_solves, time_limit=args.e3_time_limit)
    pickle.dump(path, open(DOUT / f"e3_path_m{frac}.pkl", "wb"))
    log(f"E3 m={frac}: {len(path['segments'])} segments, {path['n_solves']} solves, complete={path['complete']}, "
        f"{path['runtime_s']:.0f}s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30:.2f} GB")
    raise SystemExit

# ==================================================================== E1: fixed-policy sensitivity (algebra only)
t0 = time.time()
rows = []
for col in PA_t.columns:
    pol, b = col.split("|b=")
    lv = sl.line_coeffs(PA_v[col].to_numpy(float), qv)
    lt = sl.line_coeffs(PA_t[col].to_numpy(float), qt)
    rows.append({"policy": pol, "trained/decided at beta": int(b), "val E[r]": lv[0], "val E[(Y-r)+]": lv[1],
                 "test E[r]": lt[0], "test E[(Y-r)+]": lt[1]})
LINES = pd.DataFrame(rows)
LINES.to_csv(OUT / "s10_e1_policy_lines.csv", index=False)

# best fixed tier as a function of beta on each split (validation is the selection set)
env_rows, tier_lines = [], {}
for split, qq in (("train", qtr), ("validation", qv), ("test (retrospective)", qt)):
    lines = {float(t): sl.line_coeffs(np.full(len(qq), t), qq) for t in fl.TIERS}
    tier_lines[split] = lines
    for s0, e0, lab in sl.lower_envelope(lines, LO, HI):
        env_rows.append({"split": split, "beta from": s0, "beta to": e0, "best fixed tier": f"{int(lab * 100)}%"})
    for t_lo, t_hi in zip(fl.TIERS[:-1], fl.TIERS[1:]):
        env_rows.append({"split": split, "beta from": sl.break_even(lines[float(t_lo)], lines[float(t_hi)]),
                         "beta to": np.nan, "best fixed tier": f"pairwise break-even {int(t_lo * 100)}% vs {int(t_hi * 100)}%"})
to_md(pd.DataFrame(env_rows).set_index(["split", "best fixed tier"]), OUT / "s10_e1_best_fixed_switches", floatfmt=".3f")

# pairwise break-evens of the beta = 5 policies (validation and test lines)
key5 = ["OPT-exact [full]", "Greedy [full]", "Cost-aware RF", "CART-MSE (26 feat.) [full]", "H1: customer mean P95 + margin",
        "RF predict + margin", "Always 100%"]
L5 = LINES[(LINES["trained/decided at beta"] == 5) & LINES.policy.isin(key5)].set_index("policy")
be = []
for i, a_ in enumerate(key5):
    for b_ in key5[i + 1:]:
        for split, cols in (("validation", ("val E[r]", "val E[(Y-r)+]")), ("test", ("test E[r]", "test E[(Y-r)+]"))):
            ca, cb = tuple(L5.loc[a_, list(cols)]), tuple(L5.loc[b_, list(cols)])
            x = sl.break_even(ca, cb)
            be.append({"policy A (beta=5 version)": a_, "policy B (beta=5 version)": b_, "split": split, "break-even beta": x,
                       "in [1, 20]": bool(np.isfinite(x) and LO <= x <= HI),
                       "cheaper above break-even": a_ if ca[1] < cb[1] else b_})
for split, lines in tier_lines.items():   # also: OPT(beta=5) vs each fixed tier on validation / test
    if split == "train":
        continue
    cols = ("val E[r]", "val E[(Y-r)+]") if split == "validation" else ("test E[r]", "test E[(Y-r)+]")
    co = tuple(L5.loc["OPT-exact [full]", list(cols)])
    for t, ct in lines.items():
        x = sl.break_even(co, ct)
        be.append({"policy A (beta=5 version)": "OPT-exact [full]", "policy B (beta=5 version)": f"Always {int(t * 100)}%",
                   "split": split.split(" ")[0], "break-even beta": x, "in [1, 20]": bool(np.isfinite(x) and LO <= x <= HI),
                   "cheaper above break-even": "OPT-exact [full]" if co[1] < ct[1] else f"Always {int(t * 100)}%"})
to_md(pd.DataFrame(be).set_index(["policy A (beta=5 version)", "policy B (beta=5 version)", "split"]),
      OUT / "s10_e1_break_evens", floatfmt=".3f")

# headline robustness: savings vs best fixed chosen on validation AND vs the test-hindsight best fixed tier
hl = []
for b in fl.BETAS:
    bf_val = PA_t[f"Best fixed (val)|b={b}"].iloc[0]
    bf_test = min(fl.TIERS, key=lambda t: cost(np.full(len(ite), t), qt, b))
    cv, ch = cost(np.full(len(ite), bf_val), qt, b), cost(np.full(len(ite), bf_test), qt, b)
    for pol in ["OPT-exact [full]", "Greedy [full]", "Cost-aware RF", "OPT-IAI [200k]", "H1: customer mean P95 + margin"]:
        c_ = cost(PA_t[f"{pol}|b={b}"].to_numpy(float), qt, b)
        hl.append({"beta": b, "policy": pol, "test cost": c_, "best fixed (val) tier": f"{int(bf_val * 100)}%",
                   "% saving vs best fixed (val)": 100 * (cv - c_) / cv,
                   "test-hindsight best fixed tier": f"{int(bf_test * 100)}%",
                   "% saving vs test-hindsight best fixed (retrospective)": 100 * (ch - c_) / ch})
to_md(pd.DataFrame(hl).set_index(["beta", "policy"]), OUT / "s10_e1_headline_robustness", floatfmt=".3f")

# leaf-level switch points of the saved OPT trees (training statistics; validation / test shown for comparison)
Pv, Sv_ = fl.aggregate(Xb[iva], qv, nb)[:2]
Pt, St_ = fl.aggregate(Xb[ite], qt, nb)[:2]
lf = []
for b in fl.BETAS:
    tree = load_tree(fs.MODELS / f"OPT-exact_full_b{b}.json")
    for i, (tr_, va_, te_) in enumerate(zip(sl.leaf_table(tree, P, S, NAMES, THR), sl.leaf_table(tree, Pv, Sv_, NAMES, THR),
                                            sl.leaf_table(tree, Pt, St_, NAMES, THR))):
        row = {"tree": f"OPT-exact beta={b}", "leaf": i + 1, "path": tr_["path"], "stored action": f"{int(tr_['action'] * 100)}%",
               "train share %": 100 * tr_["n"] / NTR}
        for split, d in (("train", tr_), ("val", va_), ("test", te_)):
            if d["n"] > 0:
                segs = sl.leaf_action_segments(d["n"], d["S25"], d["S50"], d["S75"], LO, HI)
                row[f"{split} switch points"] = "; ".join(f"{int(lab * 100)}% on [{s0:.2f}, {e0:.2f}]" for s0, e0, lab in segs)
        lf.append(row)
to_md(pd.DataFrame(lf).set_index(["tree", "leaf"]), OUT / "s10_e1_leaf_switch_points", floatfmt=".1f")
runlog["E1_s"] = time.time() - t0
log("E1 done")

# ==================================================================== E2: re-optimized grid
t0 = time.time()
grid, trees = [], {}


def rec(method, b, depth, frac, tree, train_cost, rt, va_a=None, te_a=None, leaves=None):
    if va_a is None:
        va_a, te_a = fl.predict_tree(tree, Xb[iva]), fl.predict_tree(tree, Xb[ite])
    trees[len(grid)] = (tree, va_a, te_a)
    grid.append({"method": method, "beta": b, "max_depth": depth, "min_leaf_frac": frac, "train_cost": train_cost,
                 "val_cost": cost(fl.TIERS[va_a], qv, b), "test_cost": cost(fl.TIERS[te_a], qt, b),
                 "leaves": leaves if leaves is not None else fl.tree_leaves(tree), "runtime_s": rt,
                 "signature": fl.tree_signature(tree), "root": describe(tree)})


cfgs = [(b, m) for b in GRID for m in FR.values()]
for depth in (1, 2, 3):
    tt = time.time()
    for r in fl.solve_opt_exact(P, S, B, depth, cfgs):
        frac = [f for f, m in FR.items() if m == r["min_leaf"]][0]
        rec("OPT-exact", r["beta"], depth, frac, r["tree"], r["train_mean_cost"], r["runtime_s"])
    log(f"E2 OPT-exact depth {depth}: {time.time() - tt:.0f}s")
for depth in (1, 2, 3):
    for b in GRID:
        for frac, m in FR.items():
            r = fl.fit_greedy_tree(P, S, B, b, depth, m)
            rec("Greedy", b, depth, frac, r["tree"], r["train_mean_cost"], r["runtime_s"])
tt = time.time()
for depth in (1, 2, 3):
    for b in GRID:
        Ctr = fl.cost_matrix(qtr, b)
        for frac, m in FR.items():
            sk = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=m, random_state=fs.SEED).fit(Xb[itr], Ctr)
            t_ = sk.tree_

            def conv(i):
                if t_.children_left[i] == -1:
                    return fl.leaf_node(int(np.argmin(t_.value[i][:, 0])))
                return fl.split_node(int(t_.feature[i]), int(np.floor(t_.threshold[i])), conv(t_.children_left[i]),
                                     conv(t_.children_right[i]))
            tree = conv(0)
            rec("CART-MSE", b, depth, frac, tree, fl.tree_train_cost(tree, P, S, b) / NTR, np.nan)
log(f"E2 greedy + CART-MSE: {time.time() - tt:.0f}s")
G = pd.DataFrame(grid)
G.to_csv(OUT / "s10_e2_tree_grid.csv", index=False)
SEL, POLT, POLV, TREE_SEL = {}, {}, {}, {}
for (meth, b), g in G.groupby(["method", "beta"]):
    i = g.val_cost.idxmin()
    SEL[(meth, b)] = G.loc[i]
    TREE_SEL[(meth, b)] = trees[i][0]
    POLV[(f"{meth} (val-selected)", b)], POLT[(f"{meth} (val-selected)", b)] = fl.TIERS[trees[i][1]], fl.TIERS[trees[i][2]]
# integrity: every selected policy's stored actions must reproduce its recorded validation / test cost
for (meth, b), s_ in SEL.items():
    assert np.isclose(cost(POLV[(f"{meth} (val-selected)", b)], qv, b), s_.val_cost, rtol=1e-12), (meth, b)
    assert np.isclose(cost(POLT[(f"{meth} (val-selected)", b)], qt, b), s_.test_cost, rtol=1e-12), (meth, b)
# consistency with stage 5 at the original betas (same grid, same data -> identical selections)
S5 = pd.read_csv(RES / "final_tree_selected.csv")
for meth, nm in (("OPT-exact", "OPT-exact [full]"), ("Greedy", "Greedy [full]"), ("CART-MSE", "CART-MSE [full]")):
    for b in fl.BETAS:
        assert np.isclose(SEL[(meth, b)].test_cost, S5[(S5.policy == nm) & (S5.beta == b)].test_cost.iloc[0], rtol=1e-10)

# beta-free models: cost-aware RF (saved shortfall predictions) and the HW6 forest (one deterministic refit)
npz = np.load(fs.FINAL_DATA / "rf_cost_shortfall_preds.npz")
tt = time.time()
X = F[FEATS_C].to_numpy(np.float32)
rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=50, max_features=0.5, max_samples=300_000, n_jobs=-1,
                           random_state=fs.SEED).fit(X[itr], F.y_p95.to_numpy()[itr])
pv, pt = rf.predict(X[iva]), rf.predict(X[ite])
tabs = h6.leaf_tables(rf, X[itr], qtr)
Wv, Wt = h6.weighted_shortfall(rf, tabs, X[iva]), h6.weighted_shortfall(rf, tabs, X[ite])
np.savez(DOUT / "forest_msl50_preds.npz", pred_val=pv, pred_test=pt, wshort_val=Wv, wshort_test=Wt)
runlog["forest_refit_s"] = time.time() - tt
HW = pd.read_parquet(PROC / "hw6" / "forest_actions_test.parquet")
DELTAS = np.arange(-50, 100.1, 2.5)
for b in GRID:
    POLV[("Cost-aware RF", b)], POLT[("Cost-aware RF", b)] = (fl.TIERS[h6.prescribe(npz["val"], b)],
                                                              fl.TIERS[h6.prescribe(npz["test"], b)])
    POLV[("Forest-weighted prescription", b)], POLT[("Forest-weighted prescription", b)] = (
        fl.TIERS[h6.prescribe(Wv, b)], fl.TIERS[h6.prescribe(Wt, b)])
    POLV[("RF point -> tier", b)], POLT[("RF point -> tier", b)] = fl.route(pv), fl.route(pt)
    dl = DELTAS[int(np.argmin([cost(fl.route(pv, d), qv, b) for d in DELTAS]))]
    POLV[("RF point + margin", b)], POLT[("RF point + margin", b)] = fl.route(pv, dl), fl.route(pt, dl)
    mv, mt = F.sub_hist_mean.to_numpy()[iva], F.sub_hist_mean.to_numpy()[ite]
    dh = DELTAS[int(np.argmin([cost(fl.route(mv, d), qv, b) for d in DELTAS]))]
    POLV[("H1 customer mean + margin", b)], POLT[("H1 customer mean + margin", b)] = fl.route(mv, dh), fl.route(mt, dh)
    fr = F[["sub_hist_frac_le25", "sub_hist_frac_le50", "sub_hist_frac_le75"]].to_numpy()
    for part, rows_ in (("v", iva), ("t", ite)):
        ok = np.column_stack([fr[rows_] >= 1 - 1 / b - 1e-12, np.ones(len(rows_), bool)])
        (POLV if part == "v" else POLT)[("H2 critical-ratio rule", b)] = fl.TIERS[np.argmax(ok, axis=1)]
    bfv = min(fl.TIERS, key=lambda t: cost(np.full(len(iva), t), qv, b))
    POLV[("Best fixed (val)", b)], POLT[("Best fixed (val)", b)] = np.full(len(iva), bfv), np.full(len(ite), bfv)
    POLV[("Oracle (hindsight)", b)], POLT[("Oracle (hindsight)", b)] = fl.oracle_actions(qv, b), fl.oracle_actions(qt, b)
for b in fl.BETAS:   # consistency with stage 6 and HW6 (identical forest, identical rules)
    assert np.array_equal(POLT[("Cost-aware RF", b)], PA_t[f"Cost-aware RF|b={b}"].to_numpy())
    assert np.array_equal(POLT[("RF point + margin", b)], PA_t[f"RF predict + margin|b={b}"].to_numpy())
    assert np.array_equal(POLT[("Forest-weighted prescription", b)],
                          HW[f"C: forest-weighted prescription [main forest, msl=50]|b={b}"].to_numpy())
log(f"E2 beta-free models + heuristics done (forest refit {runlog['forest_refit_s']:.0f}s)")

E2_ORDER = ["Best fixed (val)", "H2 critical-ratio rule", "H1 customer mean + margin", "RF point -> tier",
            "RF point + margin", "CART-MSE (val-selected)", "Greedy (val-selected)", "OPT-exact (val-selected)",
            "Forest-weighted prescription", "Cost-aware RF", "Oracle (hindsight)"]
res = []
for b in GRID:
    bf = cost(POLT[("Best fixed (val)", b)], qt, b)
    bft = min(cost(np.full(len(ite), t), qt, b) for t in fl.TIERS)
    orc = cost(POLT[("Oracle (hindsight)", b)], qt, b)
    for pol in E2_ORDER:
        a = POLT[(pol, b)]
        ev = fl.evaluate(a, qt, betas=(b,))
        c_ = ev.pop(f"cost b={b}")
        row = {"beta": b, "policy": pol, "test cost": c_, "val cost": cost(POLV[(pol, b)], qv, b),
               "% saving vs best fixed (val)": 100 * (bf - c_) / bf,
               "% saving vs test-hindsight best fixed": 100 * (bft - c_) / bft,
               "% of gap to oracle closed": 100 * (bf - c_) / (bf - orc), "best fixed (val) tier": f"{int(POLT[('Best fixed (val)', b)][0] * 100)}%"} | ev
        meth = pol.replace(" (val-selected)", "")
        if (meth, b) in SEL:
            s_ = SEL[(meth, b)]
            row |= {"depth": int(s_.max_depth), "min leaf %": 100 * s_.min_leaf_frac, "leaves": int(s_.leaves),
                    "root split": s_.root, "features used": ", ".join(sorted(NAMES[j] for j in fl.tree_features(TREE_SEL[(meth, b)])))}
        res.append(row)
E2 = pd.DataFrame(res)
E2.to_csv(OUT / "s10_e2_reoptimized.csv", index=False)
show = ["test cost", "% saving vs best fixed (val)", "% saving vs test-hindsight best fixed", "% of gap to oracle closed",
        "avg reserved", "% under-provisioned", "mean shortfall", "% at 25%", "% at 50%", "% at 75%", "% at 100%"]
to_md(E2.pivot(index="policy", columns="beta", values="% saving vs best fixed (val)").loc[E2_ORDER],
      OUT / "s10_e2_saving_by_beta", floatfmt=".2f")
to_md(E2[E2.policy == "OPT-exact (val-selected)"].set_index("beta")[show + ["depth", "min leaf %", "leaves", "root split",
                                                                             "features used"]],
      OUT / "s10_e2_opt_by_beta", floatfmt=".3f")
to_md(E2[E2.policy.isin(["OPT-exact (val-selected)", "Greedy (val-selected)", "Cost-aware RF", "Forest-weighted prescription",
                         "Best fixed (val)", "Oracle (hindsight)"])].set_index(["beta", "policy"])[show],
      OUT / "s10_e2_main_policies", floatfmt=".3f")

bt = []
for b in GRID:
    co = POLT[("OPT-exact (val-selected)", b)]
    for other in ("Cost-aware RF", "Greedy (val-selected)", "Best fixed (val)", "Forest-weighted prescription"):
        r = fl.cluster_bootstrap_diff(co + b * np.maximum(0, qt - co), POLT[(other, b)] + b * np.maximum(0, qt - POLT[(other, b)]),
                                      subs, n_boot=1000, seed=fs.SEED)
        base = cost(POLT[(other, b)], qt, b)
        bt.append({"beta": b, "comparison": f"OPT - {other}", "diff": r["mean diff"], "ci low": r["ci low"],
                   "ci high": r["ci high"], "% of comparator": 100 * r["mean diff"] / base})
BT = pd.DataFrame(bt)
to_md(BT.set_index(["comparison", "beta"]), OUT / "s10_e2_bootstrap", floatfmt=".4f")
tier_idx = lambda d: {k: np.searchsorted(fl.TIERS, v).astype(np.int8) for k, v in d.items()}   # compact: tier index
pickle.dump({"POLT_tier_index": tier_idx(POLT), "POLV_tier_index": tier_idx(POLV), "TREE_SEL": TREE_SEL,
             "SEL": {k: v.to_dict() for k, v in SEL.items()}}, open(DOUT / "e2_policies.pkl", "wb"))
runlog["E2_s"] = time.time() - t0
log("E2 done")

# ==================================================================== E3: exact change points of the optimal tree
t0 = time.time()
NV, NT = Sv_[:, 0].sum(), St_[:, 0].sum()
E3 = {}
for frac in (0.01, 0.05):
    cache = DOUT / f"e3_path_m{frac}.pkl"
    if args.reuse_e3 and cache.exists():
        E3[frac] = pickle.load(open(cache, "rb"))
        log(f"E3 m={frac}: reused cache")
        continue
    m = FR[frac]

    def solve(beta, m=m):
        r = fl.solve_opt_exact(P, S, B, 3, [(beta, m)])[0]
        A_, B_ = sl.tree_line(r["tree"], P, S)
        return {"A": A_, "B": B_, "tree": r["tree"], "sig": fl.tree_signature(r["tree"])}
    path = sl.parametric_path(solve, LO, HI, max_solves=args.e3_max_solves, time_limit=args.e3_time_limit)
    E3[frac] = path
    pickle.dump(path, open(cache, "wb"))
    log(f"E3 m={frac}: {len(path['segments'])} segments, {path['n_solves']} solves, complete={path['complete']}, "
        f"{path['runtime_s']:.0f}s")
seg_rows = []
for frac, path in E3.items():
    prev = None
    for k, (s0, e0, r) in enumerate(path["segments"]):
        va_, te_ = sl.tree_line(r["tree"], Pv, Sv_), sl.tree_line(r["tree"], Pt, St_)
        row = {"min leaf %": 100 * frac, "segment": k + 1, "beta from": s0, "beta to": e0, "leaves": fl.tree_leaves(r["tree"]),
               "root split": describe(r["tree"]), "train reserved": r["A"] / NTR, "train shortfall": r["B"] / NTR,
               "val reserved": va_[0] / NV, "val shortfall": va_[1] / NV, "test reserved": te_[0] / NT,
               "test shortfall": te_[1] / NT}
        if prev is not None:   # out-of-sample crossover of the two adjacent trees (where their validation / test costs tie)
            row["val crossover with previous"] = sl.break_even((prev["val reserved"], prev["val shortfall"]),
                                                               (row["val reserved"], row["val shortfall"]))
            row["test crossover with previous"] = sl.break_even((prev["test reserved"], prev["test shortfall"]),
                                                                (row["test reserved"], row["test shortfall"]))
        seg_rows.append(row)
        prev = row
SEG = pd.DataFrame(seg_rows)
SEG.to_csv(OUT / "s10_e3_change_points.csv", index=False)
to_md(SEG.set_index(["min leaf %", "segment"]), OUT / "s10_e3_change_points", floatfmt=".3f")
mono = []
for frac in E3:
    s_ = SEG[SEG["min leaf %"] == 100 * frac]
    for split in ("train", "val", "test"):
        mono.append({"min leaf %": 100 * frac, "split": split,
                     "reserved non-decreasing in beta": bool((np.diff(s_[f"{split} reserved"]) >= -1e-12).all()),
                     "shortfall non-increasing in beta": bool((np.diff(s_[f"{split} shortfall"]) <= 1e-12).all()),
                     "segments": len(s_), "complete": E3[frac]["complete"], "solves": E3[frac]["n_solves"],
                     "unresolved intervals": len(E3[frac]["unresolved"])})
# concavity check of V(beta) on a fine grid (training objective = envelope of the segment lines)
for frac, path in E3.items():
    bb = np.linspace(LO, HI, 400)
    V = np.array([min(r["A"] + x * r["B"] for _, _, r in path["segments"]) for x in bb]) / NTR
    mono.append({"min leaf %": 100 * frac, "split": "train V(beta)",
                 "reserved non-decreasing in beta": "concave: " + str(bool((np.diff(V, 2) <= 1e-12).all())),
                 "shortfall non-increasing in beta": "", "segments": len(path["segments"]), "complete": path["complete"]})
to_md(pd.DataFrame(mono).set_index(["min leaf %", "split"]), OUT / "s10_e3_monotonicity")
runlog["E3_s"] = time.time() - t0
log("E3 done")

# ==================================================================== E4: wrong-beta regret (policies built for beta = 5)
t0 = time.time()
opt5 = load_tree(fs.MODELS / "OPT-exact_full_b5.json")
A5 = {"train": sl.tree_line(opt5, P, S), "val": sl.tree_line(opt5, Pv, Sv_), "test": sl.tree_line(opt5, Pt, St_)}
NN = {"train": NTR, "val": NV, "test": NT}
PS = {"train": (P, S), "val": (Pv, Sv_), "test": (Pt, St_)}
seg5 = E3[0.05]["segments"]
reg = []
for b in GRID:
    for split in ("train", "val", "test"):
        Pp, Sp = PS[split]
        fixed = (A5[split][0] + b * A5[split][1]) / NN[split]
        same = [r for s0, e0, r in seg5 if s0 <= b <= e0][0]["tree"]          # optimal depth-3 / 5% tree at beta
        same_c = sum(np.multiply((1, b), sl.tree_line(same, Pp, Sp))) / NN[split]
        retuned = TREE_SEL[("OPT-exact", b)]
        ret_c = sum(np.multiply((1, b), sl.tree_line(retuned, Pp, Sp))) / NN[split]
        reg.append({"beta": b, "split": split, "policy": "OPT-exact",
                    "fixed (built for beta=5)": fixed, "re-optimized, same class (depth 3, 5%)": same_c,
                    "re-optimized, validation-tuned": ret_c,
                    "regret vs same-class %": 100 * (fixed - same_c) / same_c,
                    "regret vs val-tuned %": 100 * (fixed - ret_c) / ret_c})
        qq = {"train": qtr, "val": qv, "test": qt}[split]
        f5 = cost(np.full(len(qq), PA_t["Best fixed (val)|b=5"].iloc[0]), qq, b)
        bfb = cost(np.full(len(qq), POLT[("Best fixed (val)", b)][0]), qq, b)
        reg.append({"beta": b, "split": split, "policy": "Best fixed tier", "fixed (built for beta=5)": f5,
                    "re-optimized, validation-tuned": bfb, "regret vs val-tuned %": 100 * (f5 - bfb) / bfb})
        if split != "train":
            Sh = npz["val"] if split == "val" else npz["test"]
            a5_, ab_ = fl.TIERS[h6.prescribe(Sh, 5)], fl.TIERS[h6.prescribe(Sh, b)]
            reg.append({"beta": b, "split": split, "policy": "Cost-aware RF (decision rule)",
                        "fixed (built for beta=5)": cost(a5_, qq, b), "re-optimized, validation-tuned": cost(ab_, qq, b),
                        "regret vs val-tuned %": 100 * (cost(a5_, qq, b) - cost(ab_, qq, b)) / cost(ab_, qq, b)})
REG = pd.DataFrame(reg)
REG.to_csv(OUT / "s10_e4_regret.csv", index=False)
to_md(REG.pivot_table(index=["policy", "split"], columns="beta", values="regret vs val-tuned %", sort=False),
      OUT / "s10_e4_regret_vs_tuned", floatfmt=".2f")
to_md(REG[REG.policy == "OPT-exact"].pivot_table(index="split", columns="beta", values="regret vs same-class %", sort=False),
      OUT / "s10_e4_regret_same_class", floatfmt=".2f")
# fixed-beta=5 OPT vs re-optimized OPT, % saving vs best fixed (val) at each beta (test, retrospective)
fx = []
for b in GRID:
    bf = cost(POLT[("Best fixed (val)", b)], qt, b)
    for nm, a in (("OPT-exact built for beta=5 (fixed)", PA_t["OPT-exact [full]|b=5"].to_numpy(float)),
                  ("Cost-aware RF decided at beta=5 (fixed)", PA_t["Cost-aware RF|b=5"].to_numpy(float))):
        fx.append({"beta": b, "policy": nm, "test cost": cost(a, qt, b), "% saving vs best fixed (val)": 100 * (bf - cost(a, qt, b)) / bf})
FX = pd.DataFrame(fx)
FX.to_csv(OUT / "s10_e4_fixed_beta5_policies.csv", index=False)
runlog["E4_s"] = time.time() - t0
log("E4 done")

# ==================================================================== figures
# Fig 1: fixed vs re-optimized across beta + wrong-beta regret
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
ax = axes[0]
sty = {"OPT-exact (val-selected)": (C["blue"], "-", "o"), "Greedy (val-selected)": ("#86b6ef", "-", "o"),
       "Cost-aware RF": (C["aqua"], "-", "s"), "Forest-weighted prescription": (C["violet"], "-", "^"),
       "H1 customer mean + margin": (C["orange"], "-", "v")}
for pol, (colr, ls, mk) in sty.items():
    d = E2[E2.policy == pol]
    ax.plot(d.beta, d["% saving vs best fixed (val)"], color=colr, ls=ls, marker=mk, ms=4, label=f"{pol} (re-optimized)")
for nm, colr in (("OPT-exact built for beta=5 (fixed)", C["blue"]), ("Cost-aware RF decided at beta=5 (fixed)", C["aqua"])):
    d = FX[FX.policy == nm]
    ax.plot(d.beta, d["% saving vs best fixed (val)"], color=colr, ls="--", lw=1.6, label=nm)
for s0, e0, lab in sl.lower_envelope(tier_lines["validation"], LO, HI):
    if s0 > LO:
        ax.axvline(s0, color=C["ink2"], lw=0.7, ls=":")
        ax.text(s0, 0.40, f" best fixed\n -> {int(lab * 100)}%", fontsize=7, color=C["ink2"],
                transform=ax.get_xaxis_transform())
ax.set_xscale("log")
ax.set_xticks(GRID, [str(g) for g in GRID], fontsize=7)
ax.set_xlabel("beta (under-provisioning penalty; log scale)")
ax.set_ylabel("% cost saving vs best fixed (val) at that beta")
ax.set_title("Fixed (built for beta=5) vs re-optimized policies, test period (retrospective)", fontsize=9.5)
ax.legend(frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
ax = axes[1]
for pol, split, colr, ls, col in (("OPT-exact", "test", C["blue"], "-", "regret vs val-tuned %"),
                                  ("OPT-exact", "val", C["blue"], ":", "regret vs val-tuned %"),
                                  ("OPT-exact", "train", C["ink2"], "-.", "regret vs same-class %"),
                                  ("Cost-aware RF (decision rule)", "test", C["aqua"], "-", "regret vs val-tuned %"),
                                  ("Best fixed tier", "test", C["orange"], "-", "regret vs val-tuned %")):
    d = REG[(REG.policy == pol) & (REG.split == split)]
    lab = f"{pol}, {split}" + (" (vs same-class optimum; >= 0 by construction)" if col.endswith("same-class %") else "")
    ax.plot(d.beta, d[col], color=colr, ls=ls, marker="o", ms=3, label=lab)
ax.axhline(0, color=C["ink2"], lw=0.8)
ax.axvline(5, color=C["ink2"], lw=0.8, ls=":")
ax.set_xscale("log")
ax.set_xticks(GRID, [str(g) for g in GRID], fontsize=7)
ax.set_xlabel("true beta (policy built assuming beta = 5)")
ax.set_ylabel("regret: % extra cost vs re-optimized policy")
ax.set_title("Cost of assuming the wrong beta", fontsize=9.5)
ax.legend(frameon=False, fontsize=6.8, loc="upper center")
fig.savefig(FOUT / "sens_fig1_fixed_vs_reoptimized.png")
plt.close(fig)

# Fig 2: reservation-tier shares across beta (test, retrospective) for OPT, cost-aware RF and the oracle
fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
for ax, pol in zip(axes, ("OPT-exact (val-selected)", "Cost-aware RF", "Oracle (hindsight)")):
    d = E2[E2.policy == pol].set_index("beta")[["% at 25%", "% at 50%", "% at 75%", "% at 100%"]]
    im = ax.imshow(d.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(4), ["25%", "50%", "75%", "100%"])
    ax.set_yticks(range(len(GRID)), [str(g) for g in GRID])
    for i in range(len(GRID)):
        for j in range(4):
            v = d.iloc[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 55 else C["ink"])
    ax.set_title(pol, fontsize=9.5)
    ax.grid(False)
axes[0].set_ylabel("beta")
bfs = [E2[(E2.policy == "Best fixed (val)") & (E2.beta == g)]["best fixed (val) tier"].iloc[0] for g in GRID]
axes[2].text(4.05, -0.9, "best fixed\n(val)", fontsize=7.5, ha="center", va="bottom", color=C["ink2"])
for i, t in enumerate(bfs):
    axes[2].text(4.05, i, t, fontsize=7.5, ha="center", va="center", color=C["ink2"], clip_on=False)
fig.colorbar(im, ax=axes, fraction=0.02, label="% of test VMs")
fig.suptitle("Share of test VMs at each reservation tier as beta increases (test period, retrospective)", fontsize=10.5)
fig.savefig(FOUT / "sens_fig2_tier_shares.png")
plt.close(fig)

# Fig 3: exact OPT change points (training objective) with out-of-sample values of the same trees
fig, axes = plt.subplots(1, 2, figsize=(12, 4.1))
for frac, colr in ((0.05, C["blue"]), (0.01, C["orange"])):
    s_ = SEG[SEG["min leaf %"] == 100 * frac]
    for ax, var in zip(axes, ("reserved", "shortfall")):
        xs = np.r_[s_["beta from"].to_numpy(), s_["beta to"].iloc[-1]]
        for split, ls in (("train", "-"), ("test", "--")):
            ys = np.r_[s_[f"{split} {var}"].to_numpy(), s_[f"{split} {var}"].iloc[-1]]
            ax.step(xs, ys, where="post", color=colr, ls=ls, lw=1.6,
                    label=f"min leaf {int(frac * 100)}%: {split}" + (" (exact optimum)" if split == "train" else " (same trees)"))
        for x in s_["beta from"].iloc[1:]:
            ax.axvline(x, color=colr, lw=0.5, alpha=0.35)
for ax, lab in zip(axes, ("mean reserved capacity of the optimal tree", "mean shortfall E[(Y - r)+] of the optimal tree")):
    ax.set_xscale("log")
    ax.set_xticks([1, 1.5, 2, 3, 5, 7, 10, 15, 20], ["1", "1.5", "2", "3", "5", "7", "10", "15", "20"])
    ax.set_xlabel("beta (log scale)")
    ax.set_ylabel(lab)
axes[0].legend(frameon=False, fontsize=7.5)
fig.suptitle("Exact change points of the optimal depth-3 tree in beta (thin vertical lines; training objective). "
             "Dashed = the same trees evaluated on the test period", fontsize=10)
fig.savefig(FOUT / "sens_fig3_opt_change_points.png")
plt.close(fig)

runlog["peak_rss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
runlog["total_s"] = time.time() - t_start
(OUT / "s10_runlog.json").write_text(json.dumps(runlog, indent=1))
print(json.dumps(runlog, indent=1))
log("Stage 10 done")
