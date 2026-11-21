"""Stage 5b (final project): independent certification of the exact DP with a Gurobi MIO, plus solver diagnostics.

On reproducible subsamples of the strict training set (8 coarse bins per selected feature, beta = 5,
min leaf = 1% of the sample) the same policy-tree problem is solved four ways:
  exact DP (exhaustive), MIO cold start, MIO warm-started from the DP tree, IAI OptimalTreePolicyMinimizer, and the
  objective-aligned greedy tree. We record objective, bound, gap, status and runtime.
Outputs: results/final_mio_crosscheck.{csv,md}
"""
import json
import time

import numpy as np
import pandas as pd

import final_lib as fl
import final_setup as fs
from common import RES, to_md

TIME_LIMIT = {2: 300, 3: 600}
F, M = fs.load_strict()
q = fs.q_of(F)
SEL = json.loads((fs.FINAL_DATA / "binner.json").read_text())["names"]
itr = np.flatnonzero(M["train"])
coarse = fl.Binner(8).fit(F.loc[M["train"], SEL].to_numpy(float), SEL)
Xb8 = coarse.transform(F[SEL].to_numpy(float))

rows = []
for n in (1000, 5000):
    idx = fs.sample_indices(itr, n, fs.SEED)
    P, S, B = fl.aggregate(Xb8[idx], q[idx], coarse.n_bins)
    m = max(1, int(round(0.01 * n)))
    for depth in (2, 3):
        base = {"n": n, "patterns": len(P), "depth": depth, "min_leaf": m, "beta": fl.MAIN_BETA}
        dp = fl.solve_opt_exact(P, S, B, depth, [(fl.MAIN_BETA, m)])[0]
        rows.append(base | {"method": "exact DP", "objective": dp["train_mean_cost"], "bound": dp["train_mean_cost"],
                            "gap": 0.0, "status": "OPTIMAL (exhaustive)", "runtime_s": dp["runtime_s"]})
        g = fl.fit_greedy_tree(P, S, B, fl.MAIN_BETA, depth, m)
        rows.append(base | {"method": "greedy", "objective": g["train_mean_cost"], "status": "heuristic",
                            "runtime_s": g["runtime_s"]})
        for warm in (None, dp["tree"]):
            r = fl.solve_opt_mio(P, S, coarse.n_bins, fl.MAIN_BETA, depth, m, time_limit=TIME_LIMIT[depth],
                                 threads=4, warm_tree=warm)
            rows.append(base | {"method": "Gurobi MIO" + (" (warm start = DP tree)" if warm is not None else ""),
                                "objective": r["train_total_cost"] / n, "bound": r["bound"] / n, "gap": r["gap"],
                                "status": r["status"], "runtime_s": r["runtime_s"], "mio_vars": r["n_vars"],
                                "mio_constraints": r["n_constrs"]})
            print(rows[-1], flush=True)
        if fl.iai_available():
            out = fl.run_iai_policy_trees(Xb8[idx], fl.cost_matrix(q[idx], fl.MAIN_BETA),
                                          [{"depth": depth, "minbucket": m, "seed": 1}],
                                          fs.FINAL_DATA / "iai_mio_check", threads=4)
            rows.append(base | {"method": "IAI OPT (local search)",
                                "objective": fl.tree_train_cost(out[0]["tree"], P, S, fl.MAIN_BETA) / n,
                                "status": "no certificate", "runtime_s": out[0]["runtime_s"]})

R = pd.DataFrame(rows)
R["objective minus exact"] = R.objective - R.groupby(["n", "depth"]).objective.transform(
    lambda s: s[R.loc[s.index, "method"] == "exact DP"].iloc[0])
to_md(R.set_index(["n", "depth", "method"]), RES / "final_mio_crosscheck", floatfmt=".6g")
print(R.to_string())
