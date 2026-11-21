"""Stage 11: OPT complexity (E5 frontier, E6 explicit leaf penalty, E7 exact depth-4). Reads finalized outputs only.

  --part frontier        E5: validation / test cost vs depth, leaves and splits from the existing tree grids
                         (stage 5 grid at beta 2/5/10 and the stage-10 grid at 12 betas); 4-leaf vs 8-leaf OPT.
  --part penalty         E6: exact lambda-path of  mean cost + lambda * #leaves  (depth 3, min leaf 0.1%) for beta
                         2/5/10; every tree on the lower convex hull of (leaves, training cost); lambda chosen on
                         validation, preferring the simpler tree when validation cost is within 0.1%.
  --part depth4          E7: ONE exact depth-4 solve (flags: --beta --min-leaf-frac --bins --budget-min), written to
                         results/sensitivity/depth4/; stops at the wall-clock budget with an uncertified incumbent.
  --part depth4-summary  E7: compares finished depth-4 runs with depth-3 OPT, greedy depth 4 and IAI depth 4.
Outputs: results/sensitivity/s11_*, figures/sensitivity/sens_fig{4,5}_*.png, results/sensitivity/depth4/*.json
"""
import argparse
import json
import pickle
import resource
import time

import numpy as np
import pandas as pd

import final_lib as fl
import final_setup as fs
import sensitivity_lib as sl
from common import C, FIG, PROC, RES, plt, to_md

ap = argparse.ArgumentParser()
ap.add_argument("--part", required=True, choices=["frontier", "penalty", "depth4", "depth4-summary"])
ap.add_argument("--beta", type=float, default=5)
ap.add_argument("--min-leaf-frac", type=float, default=0.05)
ap.add_argument("--bins", type=int, default=32, choices=[16, 32])
ap.add_argument("--budget-min", type=float, default=30)
ap.add_argument("--reuse", action="store_true", help="E6: reuse cached lambda paths if present")
args = ap.parse_args()

OUT, FOUT = RES / "sensitivity", FIG / "sensitivity"
D4 = OUT / "depth4"
for d in (OUT, FOUT, D4):
    d.mkdir(parents=True, exist_ok=True)
t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)


def load_data(bins):
    """Strict masks, labels and the binned 8 OPT features (32 = the original candidate set; 16 = coarser re-binning)."""
    F = pd.read_parquet(PROC / "features.parquet", columns=["split", "post_vmdeleted", "y_p95", "key_sub", "cores",
                                                            "post_lifetime_h"] + fs.OPT_CANDIDATES)
    M = fs.masks_for(F, strict=True)
    q = fs.q_of(F)
    js = json.loads((fs.FINAL_DATA / "binner.json").read_text())
    if bins == 32:
        Xb = np.load(fs.FINAL_DATA / "Xb_opt.npy")
        names, thr = js["names"], [np.array(t) for t in js["thresholds"]]
    else:
        from common import impute
        Fi = impute(pd.read_parquet(PROC / "features.parquet"), train_mask=M["train"])   # same imputation as stage 5
        bn = fl.Binner(bins).fit(Fi.loc[M["train"], js["names"]].to_numpy(float), js["names"])
        Xb = bn.transform(Fi[js["names"]].to_numpy(float))
        names, thr = js["names"], bn.thresholds
    nb = np.array([len(t) + 1 for t in thr])
    return F, M, q, Xb, names, thr, nb


def tree_json(tree, names, thr):
    if tree["leaf"]:
        return {"action": float(fl.TIERS[tree["action"]])}
    f, t = tree["feature"], tree["t"]
    return {"feature": names[f], "threshold_le": float(thr[f][t]), "bin_t": t,
            "left": tree_json(tree["left"], names, thr), "right": tree_json(tree["right"], names, thr)}


def eval_tree(tree, Xb, idx, q, b):
    a = fl.TIERS[fl.predict_tree(tree, Xb[idx])]
    return float(np.mean(a + b * np.maximum(0, q[idx] - a))), a


# ==================================================================== E5: complexity frontier from existing grids
if args.part == "frontier":
    G5 = pd.read_csv(RES / "final_tree_grid.csv")
    G10 = pd.read_csv(OUT / "s10_e2_tree_grid.csv")
    rows = []
    for src, G, tset in (("stage 5", G5[G5.train_set.isin(["full", "200k"])], None), ("stage 10", G10, "full")):
        for keys, g in G.groupby(["method", "train_set", "beta", "max_depth"] if tset is None else ["method", "beta", "max_depth"]):
            r = g.loc[g.val_cost.idxmin()]
            ts = r.train_set if tset is None else tset
            rows.append({"source": src, "method": r.method, "train set": ts, "beta": r.beta, "max depth": int(r.max_depth),
                         "leaves": int(r.leaves), "splits": int(r.leaves) - 1, "min leaf % (val-selected)": 100 * r.min_leaf_frac,
                         "train cost": r.train_cost, "val cost": r.val_cost, "test cost": r.test_cost})
    FR = pd.DataFrame(rows)
    FR.to_csv(OUT / "s11_e5_frontier.csv", index=False)
    to_md(FR[(FR.source == "stage 5") & (FR.beta.isin([2, 5, 10]))].set_index(["beta", "method", "train set", "max depth"])
          .sort_index(), OUT / "s11_e5_frontier_b2_5_10", floatfmt=".4f")
    # 4-leaf (depth 2) vs 8-leaf (depth 3) exact OPT, full data, all 12 betas; bootstrap at 2 / 5 / 10
    F, M, q, Xb, names, thr, nb = load_data(32)
    itr, iva, ite = (np.flatnonzero(M[k]) for k in ("train", "val", "test"))
    P, S, B = fl.aggregate(Xb[itr], q[itr], nb)
    PA = pd.read_parquet(fs.FINAL_DATA / "policy_actions_test.parquet")
    E2 = pd.read_csv(OUT / "s10_e2_reoptimized.csv")
    subs = F.key_sub.to_numpy()[ite]
    fv = []
    for b in sorted(G10.beta.unique()):
        g = FR[(FR.source == "stage 10") & (FR.method == "OPT-exact") & (FR.beta == b)].set_index("max depth")
        bf = E2[(E2.beta == b) & (E2.policy == "Best fixed (val)")]["test cost"].iloc[0]
        row = {"beta": b, "test cost 2 leaves": g.loc[1, "test cost"], "test cost 4 leaves": g.loc[2, "test cost"],
               "test cost 8 leaves (depth 3)": g.loc[3, "test cost"],
               "% saving 4 leaves": 100 * (bf - g.loc[2, "test cost"]) / bf, "% saving depth 3": 100 * (bf - g.loc[3, "test cost"]) / bf,
               "val cost 4 leaves": g.loc[2, "val cost"], "val cost depth 3": g.loc[3, "val cost"]}
        row["share of depth-3 saving captured by 4 leaves %"] = 100 * row["% saving 4 leaves"] / row["% saving depth 3"]
        if b in (2, 5, 10):
            m2 = int(round(g.loc[2, "min leaf % (val-selected)"] / 100 * len(itr)))
            t2 = fl.solve_opt_exact(P, S, B, 2, [(b, m2)])[0]["tree"]
            c2, a2 = eval_tree(t2, Xb, ite, q, b)
            assert np.isclose(c2, g.loc[2, "test cost"])
            a3 = PA[f"OPT-exact [full]|b={int(b)}"].to_numpy(float)
            r = fl.cluster_bootstrap_diff(a2 + b * np.maximum(0, q[ite] - a2), a3 + b * np.maximum(0, q[ite] - a3), subs,
                                          n_boot=1000, seed=fs.SEED)
            row |= {"4-leaf minus depth-3 (test)": r["mean diff"], "95% CI low": r["ci low"], "95% CI high": r["ci high"]}
            (OUT / f"s11_e5_opt_depth2_b{int(b)}.txt").write_text(json.dumps(tree_json(t2, names, thr), indent=1))
        fv.append(row)
    to_md(pd.DataFrame(fv).set_index("beta"), OUT / "s11_e5_four_vs_eight_leaves", floatfmt=".4f")
    log("E5 done")

# ==================================================================== E6: explicit leaf penalty (exact lambda path)
if args.part == "penalty":
    F, M, q, Xb, names, thr, nb = load_data(32)
    itr, iva, ite = (np.flatnonzero(M[k]) for k in ("train", "val", "test"))
    P, S, B = fl.aggregate(Xb[itr], q[itr], nb)
    n = S[:, 0].sum()
    m = max(1, int(round(0.001 * len(itr))))          # smallest grid value: lambda is the complexity control
    PA = pd.read_parquet(fs.FINAL_DATA / "policy_actions_test.parquet")
    subs = F.key_sub.to_numpy()[ite]
    hull, sel_rows = [], []
    for b in fl.BETAS:
        t0 = time.time()

        def solve(lam, b=b):
            r = fl.solve_opt_exact(P, S, B, 3, [(b, m, lam)])[0]
            return {"A": r["train_total_cost"], "B": n * r["leaves"], "tree": r["tree"], "leaves": r["leaves"]}
        cache = PROC / "sensitivity" / f"e6_lambda_path_b{b}.pkl"
        if args.reuse and cache.exists():
            path = pickle.load(open(cache, "rb"))
        else:
            path = sl.parametric_path(solve, 0.0, 0.5, max_solves=120, time_limit=1800)
            cache.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(path, open(cache, "wb"))
        log(f"E6 beta={b}: {len(path['segments'])} hull trees, {path['n_solves']} solves, complete={path['complete']}, "
            f"{time.time() - t0:.0f}s")
        recs = []
        for s0, e0, r in path["segments"]:
            cv, _ = eval_tree(r["tree"], Xb, iva, q, b)
            ct, at = eval_tree(r["tree"], Xb, ite, q, b)
            recs.append({"beta": b, "lambda from": s0, "lambda to": e0, "leaves": r["leaves"],
                         "train cost (unpenalized)": r["A"] / n, "penalized objective at lambda-from": (r["A"] + s0 * r["B"]) / n,
                         "val cost": cv, "test cost": ct, "root": "-" if r["tree"]["leaf"] else
                         f"{names[r['tree']['feature']]} <= {thr[r['tree']['feature']][r['tree']['t']]:.3g}",
                         "complete path": path["complete"], "_tree": r["tree"], "_a": at})
        hull += recs
        best_v = min(x["val cost"] for x in recs)
        chosen_min = min(recs, key=lambda x: x["val cost"])
        chosen_simple = min((x for x in recs if x["val cost"] <= best_v * 1.001), key=lambda x: x["leaves"])
        a3 = PA[f"OPT-exact [full]|b={b}"].to_numpy(float)
        for rule, x in (("min validation cost", chosen_min), ("simplest within 0.1% of min validation cost", chosen_simple)):
            bt = fl.cluster_bootstrap_diff(x["_a"] + b * np.maximum(0, q[ite] - x["_a"]), a3 + b * np.maximum(0, q[ite] - a3),
                                           subs, n_boot=1000, seed=fs.SEED)
            sel_rows.append({"beta": b, "selection rule": rule, "leaves": x["leaves"], "lambda range": f"[{x['lambda from']:.5f}, {x['lambda to']:.5f}]",
                             "train cost": x["train cost (unpenalized)"], "val cost": x["val cost"], "test cost": x["test cost"],
                             "test minus original OPT": bt["mean diff"], "95% CI low": bt["ci low"], "95% CI high": bt["ci high"]})
            (OUT / f"s11_e6_selected_b{b}_{'min' if rule.startswith('min') else 'simple'}.json").write_text(
                json.dumps(tree_json(x["_tree"], names, thr), indent=1))
    H = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in hull])
    H.to_csv(OUT / "s11_e6_lambda_hull.csv", index=False)
    to_md(H.set_index(["beta", "leaves"]), OUT / "s11_e6_lambda_hull", floatfmt=".5f")
    to_md(pd.DataFrame(sel_rows).set_index(["beta", "selection rule"]), OUT / "s11_e6_selection", floatfmt=".4f")

    # Figure: cost vs complexity (E5 + E6), validation and test, beta 2 / 5 / 10
    FR = pd.read_csv(OUT / "s11_e5_frontier.csv")
    E2 = pd.read_csv(OUT / "s10_e2_reoptimized.csv")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for j, b in enumerate(fl.BETAS):
        for i, split in enumerate(("val", "test")):
            ax = axes[i, j]
            h = H[H.beta == b].sort_values("leaves")
            ax.plot(h.leaves, h[f"{split} cost"], color=C["blue"], marker="D", ms=5, lw=1.5,
                    label="OPT-exact, lambda-penalized hull (depth <= 3, min leaf 0.1%)")
            f5 = FR[(FR.source == "stage 5") & (FR.beta == b) & (FR["train set"] == "full")]
            for meth, mk, colr in (("OPT-exact", "o", C["ink"]), ("Greedy", "s", "#86b6ef"), ("CART-MSE", "^", C["orange"])):
                d = f5[f5.method == meth].sort_values("leaves")
                d3 = d[d["max depth"] <= 3]
                ax.plot(d3.leaves, d3[f"{split} cost"], ls="none", marker=mk, ms=7, mfc="none", mec=colr,
                        label=f"{meth} depth 1-3 (val-selected per depth)")
                d4 = d[d["max depth"] == 4]
                if len(d4):
                    ax.plot(d4.leaves, d4[f"{split} cost"], ls="none", marker=mk, ms=7, color=colr,
                            label=f"{meth} depth 4 (different class)")
            for pol, ls, colr in (("Cost-aware RF", "--", C["aqua"]), ("H1 customer mean + margin", ":", C["violet"])):
                v = E2[(E2.beta == b) & (E2.policy == pol)][f"{split} cost"].iloc[0]
                ax.axhline(v, color=colr, ls=ls, lw=1.2, label=pol)
            sel = pd.DataFrame(sel_rows)
            x = sel[(sel.beta == b) & (sel["selection rule"].str.startswith("simplest"))].iloc[0]
            ax.plot(x.leaves, x[f"{split} cost"], marker="*", ms=14, color=C["red"], ls="none",
                    label="validation-selected penalized tree (simplest within 0.1%)")
            ax.set_title(f"beta = {b}: {'validation' if split == 'val' else 'test (retrospective)'}", fontsize=9.5)
            if j == 0:
                ax.set_ylabel(f"mean modeled cost ({split})")
            if i == 1:
                ax.set_xlabel("number of leaves")
    for ax in axes.ravel():
        ax.set_xticks(range(1, 17, 2))
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Allocation cost vs. tree complexity: exact OPT (depth grid and leaf-penalty hull) vs greedy / CART", fontsize=11)
    fig.savefig(FOUT / "sens_fig4_complexity.png")
    plt.close(fig)
    log("E6 done")

# ==================================================================== E7: one exact depth-4 run
if args.part == "depth4":
    F, M, q, Xb, names, thr, nb = load_data(args.bins)
    itr, iva, ite = (np.flatnonzero(M[k]) for k in ("train", "val", "test"))
    P, S, B = fl.aggregate(Xb[itr], q[itr], nb)
    b, m = args.beta, max(1, int(round(args.min_leaf_frac * len(itr))))
    tag = f"b{args.beta:g}_m{args.min_leaf_frac:g}_bins{args.bins}"
    prog_file = D4 / f"progress_{tag}.json"
    t0 = time.time()

    def progress(done, total):
        prog_file.write_text(json.dumps({"evaluated": done, "root_candidates": total, "elapsed_s": time.time() - t0,
                                         "peak_rss_GB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30}))
    d3 = fl.solve_opt_exact(P, S, B, 3, [(b, m)])[0]
    r = fl.solve_opt_exact(P, S, B, 4, [(b, m)], time_limit=args.budget_min * 60, progress=progress)[0]
    out = {"beta": b, "min_leaf_frac": args.min_leaf_frac, "min_leaf": m, "bins": args.bins, "budget_min": args.budget_min,
           "status": r["status"], "root_candidates": r["root_candidates"], "evaluated": r["evaluated"],
           "runtime_s": time.time() - t0, "peak_rss_GB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30,
           "patterns": len(P)}
    for nm, rr in (("depth4", r), ("depth3", d3)):
        cv, _ = eval_tree(rr["tree"], Xb, iva, q, b)
        ct, _ = eval_tree(rr["tree"], Xb, ite, q, b)
        out[nm] = {"train_cost": rr["train_mean_cost"], "val_cost": cv, "test_cost": ct, "leaves": rr["leaves"],
                   "depth": fl.tree_depth(rr["tree"]), "tree": tree_json(rr["tree"], names, thr)}
    if args.bins == 16:      # greedy depth 4 / 3 on the same coarse candidate set, for a like-for-like comparison
        for d_ in (3, 4):
            g = fl.fit_greedy_tree(P, S, B, b, d_, m)
            cv, _ = eval_tree(g["tree"], Xb, iva, q, b)
            ct, _ = eval_tree(g["tree"], Xb, ite, q, b)
            out[f"greedy_depth{d_}"] = {"train_cost": g["train_mean_cost"], "val_cost": cv, "test_cost": ct,
                                        "leaves": fl.tree_leaves(g["tree"]), "tree": tree_json(g["tree"], names, thr)}
    (D4 / f"run_{tag}.json").write_text(json.dumps(out, indent=1))
    log(f"depth-4 {tag}: {out['status']} ({out['evaluated']}/{out['root_candidates']} root splits, {out['runtime_s']:.0f}s)")

# ==================================================================== E7 summary
if args.part == "depth4-summary":
    runs = [json.loads(p.read_text()) for p in sorted(D4.glob("run_*.json"))]
    if not runs:
        log("no depth-4 runs found")
        raise SystemExit
    G5 = pd.read_csv(RES / "final_tree_grid.csv")
    rows = []
    for r in runs:
        b = r["beta"]
        base = {"beta": b, "bins": r["bins"], "min leaf %": 100 * r["min_leaf_frac"], "status": r["status"],
                "root splits evaluated": f"{r['evaluated']}/{r['root_candidates']}", "runtime min": r["runtime_s"] / 60,
                "peak RSS GB": r["peak_rss_GB"]}
        for nm in ("depth3", "depth4", "greedy_depth3", "greedy_depth4"):
            if nm in r:
                rows.append(base | {"model": {"depth3": "OPT-exact depth 3", "depth4": "OPT-exact depth 4",
                                              "greedy_depth3": "greedy depth 3", "greedy_depth4": "greedy depth 4"}[nm]}
                            | {k: r[nm][k] for k in ("train_cost", "val_cost", "test_cost", "leaves")})
        if r["bins"] == 32:   # existing 32-bin greedy / IAI depth-4 (validation-selected over min leaf) for context
            for meth, ts in (("Greedy", "full"), ("OPT-IAI", "200k")):
                g = G5[(G5.method == meth) & (G5.train_set == ts) & (G5.beta == b) & (G5.max_depth == 4)]
                x = g.loc[g.val_cost.idxmin()]
                rows.append(base | {"status": "existing: " + ("greedy (no guarantee)" if meth == "Greedy" else
                                                             "IAI local search (no certificate)"),
                                    "model": f"{meth} depth 4 [{ts}] (existing, val-selected min leaf "
                                             f"{100 * x.min_leaf_frac:g}%)", "train_cost": x.train_cost,
                                    "val_cost": x.val_cost, "test_cost": x.test_cost, "leaves": int(x.leaves)})
    T = pd.DataFrame(rows)
    to_md(T.set_index(["beta", "bins", "min leaf %", "model"]), OUT / "s11_e7_depth4", floatfmt=".4f")

    # bootstrap: depth-4 OPT vs depth-3 OPT (same bins / min leaf) on test, for finished certified runs
    F, M, q, Xb32, names, thr32, nb32 = load_data(32)
    ite = np.flatnonzero(M["test"])
    subs = F.key_sub.to_numpy()[ite]
    bt = []
    for r in runs:
        if r["bins"] != 32:
            continue
        b = r["beta"]
        bn = type("Bn", (), {"names": names})()
        acts = {nm: fl.TIERS[fl.predict_tree(fl.tree_from_json(r[nm]["tree"], bn), Xb32[ite])] for nm in ("depth3", "depth4")}
        cst = {k: a + b * np.maximum(0, q[ite] - a) for k, a in acts.items()}
        x = fl.cluster_bootstrap_diff(cst["depth4"], cst["depth3"], subs, n_boot=1000, seed=fs.SEED)
        bt.append({"beta": b, "min leaf %": 100 * r["min_leaf_frac"], "status": r["status"][:8],
                   "depth4 - depth3 (test)": x["mean diff"], "95% CI low": x["ci low"], "95% CI high": x["ci high"]})
    if bt:
        to_md(pd.DataFrame(bt).set_index(["beta", "min leaf %"]), OUT / "s11_e7_bootstrap", floatfmt=".4f")
    # validation-selected exact depth-4 (over the finished certified 32-bin runs at each beta) vs the original
    # depth-3 OPT and the existing greedy / IAI depth-4 trees (saved test actions); paired cluster bootstrap
    PA = pd.read_parquet(fs.FINAL_DATA / "policy_actions_test.parquet")
    sel_rows = []
    for b in sorted({r["beta"] for r in runs if r["bins"] == 32 and r["status"].startswith("OPTIMAL")}):
        cand = [r for r in runs if r["beta"] == b and r["bins"] == 32 and r["status"].startswith("OPTIMAL")]
        r = min(cand, key=lambda x: x["depth4"]["val_cost"])
        bn = type("Bn", (), {"names": names})()
        a4 = fl.TIERS[fl.predict_tree(fl.tree_from_json(r["depth4"]["tree"], bn), Xb32[ite])]
        c4 = a4 + b * np.maximum(0, q[ite] - a4)
        for comp in ("OPT-exact [full]", "Greedy depth-4 [full]", "OPT-IAI depth-4 [200k]", "Cost-aware RF"):
            ac = PA[f"{comp}|b={int(b)}"].to_numpy(float)
            x = fl.cluster_bootstrap_diff(c4, ac + b * np.maximum(0, q[ite] - ac), subs, n_boot=1000, seed=fs.SEED)
            sel_rows.append({"beta": b, "selected depth-4 min leaf %": 100 * r["min_leaf_frac"],
                             "depth-4 leaves": r["depth4"]["leaves"], "depth-4 val cost": r["depth4"]["val_cost"],
                             "depth-4 test cost": r["depth4"]["test_cost"], "comparator": comp,
                             "comparator test cost": float(np.mean(ac + b * np.maximum(0, q[ite] - ac))),
                             "depth4 - comparator": x["mean diff"], "95% CI low": x["ci low"], "95% CI high": x["ci high"]})
    if sel_rows:
        to_md(pd.DataFrame(sel_rows).set_index(["beta", "comparator"]), OUT / "s11_e7_selected_vs_existing", floatfmt=".4f")
    # figure: validation (hollow) and test (filled) cost per model, one panel per beta (validation-selected depth 4)
    PAv = pd.read_parquet(fs.FINAL_DATA / "policy_actions_val.parquet")
    iva = np.flatnonzero(M["val"])
    S5 = pd.read_csv(RES / "final_tree_selected.csv")
    betas = sorted({x["beta"] for x in sel_rows})
    fig, axes = plt.subplots(1, len(betas), figsize=(4.3 * len(betas) + 2.2, 3.4), squeeze=False, sharey=True)
    for j, (ax, b) in enumerate(zip(axes[0], betas)):
        r = min([x for x in runs if x["beta"] == b and x["bins"] == 32 and x["status"].startswith("OPTIMAL")],
                key=lambda x: x["depth4"]["val_cost"])
        items = []
        for pol, lab in (("OPT-exact [full]", "OPT exact, depth 3"), ("Greedy depth-4 [full]", "greedy, depth 4"),
                         ("OPT-IAI depth-4 [200k]", "IAI OPT, depth 4 (200k)"), ("Cost-aware RF", "cost-aware RF")):
            av, at = PAv[f"{pol}|b={int(b)}"].to_numpy(float), PA[f"{pol}|b={int(b)}"].to_numpy(float)
            row = S5[(S5.policy == pol) & (S5.beta == b)]
            lv = f"{int(row.leaves.iloc[0])} leaves" if len(row) else "100 trees"
            items.append((lab, lv, np.mean(av + b * np.maximum(0, q[iva] - av)), np.mean(at + b * np.maximum(0, q[ite] - at))))
        items.insert(1, ("OPT exact, depth 4 (certified)", f"{r['depth4']['leaves']} leaves", r["depth4"]["val_cost"],
                         r["depth4"]["test_cost"]))
        ys = np.arange(len(items))[::-1]
        for y, (lab, lv, v, t) in zip(ys, items):
            colr = C["blue"] if lab.startswith("OPT exact, depth 4") else (C["aqua"] if "RF" in lab else C["ink2"])
            ax.plot(v, y, marker="o", mfc="none", mec=colr, ms=8, ls="none")
            ax.plot(t, y, marker="o", color=colr, ms=8, ls="none")
            ax.annotate(lv, (t, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7, color=C["ink2"])
        if j == 0:
            ax.set_yticks(ys, [it[0] for it in items], fontsize=8.5)
        ax.set_ylim(-0.6, len(items) - 0.4)
        ax.set_title(f"beta = {b:g}", fontsize=10)
        ax.set_xlabel("mean modeled cost (hollow = validation, filled = test)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Exact depth-4 OPT (validation-selected min leaf, 32 bins, certified optimal) vs depth-3 OPT, "
                 "depth-4 greedy / IAI and cost-aware RF", fontsize=10.5, y=1.03)
    fig.savefig(FOUT / "sens_fig5_depth4.png")
    plt.close(fig)
    print(T.to_string())
    log("E7 summary done")
