"""Final-project library: modeled costs, policy evaluation, binning and policy trees.

Policy trees implemented here (all minimize the modeled allocation cost directly, except CART-MSE in the scripts):
* fit_greedy_tree   - objective-aligned greedy recursive partitioning (split = largest one-step drop in policy cost).
* solve_opt_exact   - globally optimal policy tree of depth <= 3 by exhaustive dynamic programming over a fixed
                      candidate split set (training quantile thresholds). Exhaustive => certified optimum over
                      that split set (status "OPTIMAL", gap 0).
* solve_opt_mio     - mixed-integer formulation solved with Gurobi (used to cross-check the DP and to report
                      bounds / gaps / runtimes on subsamples).
* IAI bridge        - Interpretable AI OptimalTreePolicyMinimizer (Amram, Dunn & Zhuo 2022) run through Julia;
                      local search, no optimality certificate.

Units: q = p95maxcpu / 100 in [0, 1]; actions are fractions TIERS of the VM's allocated vCPUs.
Cost:  C(a; q, beta) = a + beta * max(0, q - a).
Sufficient statistics for any beta: per region, count n and shortfall sums S_k = sum max(0, q - TIERS[k]) for the
three lower tiers (the 100% tier never falls short because q <= 1), so cost_k = n * TIERS[k] + beta * S_k.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

TIERS = np.array([0.25, 0.50, 0.75, 1.00])
BETAS = (2, 5, 10)
MAIN_BETA = 5
DAY = 86_400
TRAIN_END, VAL_END = 21 * DAY, 25 * DAY        # label-availability cutoffs for train / validation
TRACE_END = 2_591_400
_TOL = 1e-9

# Columns that must never be creation-time predictors.
FORBIDDEN_EXACT = {"y_p95", "p95maxcpu", "avgcpu", "maxcpu", "vmdeleted", "vmcategory", "lifetime_h", "deploymentsize",
                   "vmid", "subscriptionid", "deploymentid", "key_sub", "key_dep", "split", "vmcreated"}
FORBIDDEN_PREFIXES = ("post_", "LEAKY_")


def forbidden_features(cols):
    return [c for c in cols if c in FORBIDDEN_EXACT or c.startswith(FORBIDDEN_PREFIXES)]


# ---------------------------------------------------------------- units, costs, oracle, evaluation
def p95_to_fraction(p95_pct):
    """Convert p95maxcpu (percent, 0-100) to a fraction; refuses values outside the percent range."""
    p = np.asarray(p95_pct, dtype=float)
    if np.nanmin(p) < 0 or np.nanmax(p) > 100 + 1e-9:
        raise ValueError("p95maxcpu must be in percent [0, 100]")
    return p / 100.0


def check_fraction(q):
    q = np.asarray(q, dtype=float)
    if np.nanmin(q) < 0 or np.nanmax(q) > 1 + 1e-12:
        raise ValueError("utilization fractions must lie in [0, 1]")
    return q


def cost_matrix(q, beta):
    """n x 4 modeled costs C_ik = TIERS[k] + beta * max(0, q_i - TIERS[k])."""
    q = check_fraction(q)
    return TIERS[None, :] + beta * np.maximum(0.0, q[:, None] - TIERS[None, :])


def shortfall_stats(q):
    """n x 4 sufficient statistics [1, max(0,q-.25), max(0,q-.5), max(0,q-.75)]."""
    q = check_fraction(q)
    return np.column_stack([np.ones_like(q)] + [np.maximum(0.0, q - t) for t in TIERS[:3]])


def stats_to_costs(st, beta):
    """(..., 4) sufficient statistics -> (..., 4) total costs of the four actions."""
    n = st[..., 0]
    return np.stack([n * TIERS[k] + beta * st[..., k + 1] for k in range(3)] + [n * TIERS[3]], axis=-1)


def oracle_actions(q, beta):
    """Hindsight cost-minimizing tier per VM (ties -> smaller tier)."""
    return TIERS[np.argmin(cost_matrix(q, beta), axis=1)]


def evaluate(a, q, betas=BETAS, w=None):
    """Policy metrics. a, q are fractions; w optional evaluation weights (e.g. vCPUs or vCPU-hours)."""
    a, q = np.asarray(a, float), check_fraction(q)
    w = np.ones_like(q) if w is None else np.asarray(w, float)
    sf, ws = np.maximum(0, q - a), np.maximum(0, a - q)
    out = {f"cost b={b}": np.average(a + b * sf, weights=w) for b in betas}
    out |= {"avg reserved": np.average(a, weights=w), "% under-provisioned": 100 * np.average(q > a + 1e-12, weights=w),
            "mean shortfall": np.average(sf, weights=w), "mean unused": np.average(ws, weights=w)}
    for t in TIERS:
        out[f"% at {int(round(t * 100))}%"] = 100 * np.average(np.isclose(a, t), weights=w)
    return out


def route(pred_pct, delta=0.0):
    """Smallest tier >= (prediction + delta); predictions in percent."""
    idx = np.searchsorted(TIERS, (np.asarray(pred_pct, float) + delta) / 100 - 1e-12)
    return TIERS[np.clip(idx, 0, len(TIERS) - 1)]


def cluster_bootstrap_diff(cost_a, cost_b, clusters, n_boot=1000, seed=0):
    """Paired cluster bootstrap of mean(cost_a - cost_b), resampling whole clusters (subscriptions)."""
    diff = np.asarray(cost_a, float) - np.asarray(cost_b, float)
    _, inv = np.unique(clusters, return_inverse=True)
    sd, nd = np.bincount(inv, diff), np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    G = len(sd)
    reps = np.empty(n_boot)
    for r in range(n_boot):
        cnt = np.bincount(rng.integers(0, G, G), minlength=G)
        reps[r] = (cnt @ sd) / (cnt @ nd)
    return {"mean diff": diff.mean(), "ci low": np.percentile(reps, 2.5), "ci high": np.percentile(reps, 97.5),
            "share reps < 0": np.mean(reps < 0)}


# ---------------------------------------------------------------- binning (candidate split set)
class Binner:
    """Per-feature thresholds from TRAINING data. bin j  <=>  thr[j-1] < x <= thr[j]; split 'bin <= t' <=> x <= thr[t]."""

    def __init__(self, max_bins=32):
        self.max_bins = max_bins

    def fit(self, X, names):
        self.names, self.thresholds = list(names), []
        for j in range(X.shape[1]):
            v = np.asarray(X[:, j], float)
            u = np.unique(v)
            if len(u) <= self.max_bins:
                thr = u[:-1]
            else:
                thr = np.unique(np.quantile(v, np.arange(1, self.max_bins) / self.max_bins, method="higher"))
                thr = thr[thr < u[-1]]
            self.thresholds.append(thr)
        self.n_bins = np.array([len(t) + 1 for t in self.thresholds])
        return self

    def transform(self, X):
        out = np.empty(X.shape, dtype=np.int16)
        for j, thr in enumerate(self.thresholds):
            out[:, j] = np.searchsorted(thr, np.asarray(X[:, j], float), side="left")
        return out

    def threshold_value(self, j, t):
        return float(self.thresholds[j][t])


# ---------------------------------------------------------------- trees: representation and prediction
def leaf_node(action):
    return {"leaf": True, "action": int(action)}


def split_node(feature, t, left, right):
    return {"leaf": False, "feature": int(feature), "t": int(t), "left": left, "right": right}


def predict_tree(tree, Xb):
    """Action indices (0..3) for binned rows Xb."""
    out = np.empty(len(Xb), dtype=np.int8)

    def rec(node, idx):
        if node["leaf"]:
            out[idx] = node["action"]
            return
        go_left = Xb[idx, node["feature"]] <= node["t"]
        rec(node["left"], idx[go_left])
        rec(node["right"], idx[~go_left])

    rec(tree, np.arange(len(Xb)))
    return out


def tree_depth(tree):
    return 0 if tree["leaf"] else 1 + max(tree_depth(tree["left"]), tree_depth(tree["right"]))


def tree_leaves(tree):
    return 1 if tree["leaf"] else tree_leaves(tree["left"]) + tree_leaves(tree["right"])


def tree_features(tree):
    return set() if tree["leaf"] else {tree["feature"]} | tree_features(tree["left"]) | tree_features(tree["right"])


def tree_signature(tree):
    """Canonical string (feature, bin-threshold, actions) used to compare trees across seeds."""
    if tree["leaf"]:
        return f"L{tree['action']}"
    return f"({tree['feature']}<={tree['t']}?{tree_signature(tree['left'])}:{tree_signature(tree['right'])})"


def tree_to_text(tree, binner, indent=""):
    if tree["leaf"]:
        return f"{indent}reserve {int(TIERS[tree['action']] * 100)}%\n"
    name, thr = binner.names[tree["feature"]], binner.threshold_value(tree["feature"], tree["t"])
    return (f"{indent}if {name} <= {thr:.4g}:\n" + tree_to_text(tree["left"], binner, indent + "    ")
            + f"{indent}else:  # {name} > {thr:.4g}\n" + tree_to_text(tree["right"], binner, indent + "    "))


def tree_to_json(tree, binner):
    def rec(n):
        if n["leaf"]:
            return {"action": float(TIERS[n["action"]])}
        return {"feature": binner.names[n["feature"]], "threshold_le": binner.threshold_value(n["feature"], n["t"]),
                "bin_t": n["t"], "left": rec(n["left"]), "right": rec(n["right"])}
    return rec(tree)


def tree_from_json(js, binner):
    if "action" in js:
        return leaf_node(int(np.argmin(np.abs(TIERS - js["action"]))))
    return split_node(binner.names.index(js["feature"]), js["bin_t"], tree_from_json(js["left"], binner),
                      tree_from_json(js["right"], binner))


# ---------------------------------------------------------------- aggregation into unique patterns
def aggregate(Xb, q, n_bins):
    """Collapse identical binned rows: returns patterns (U, p) and summed sufficient statistics (U, 4)."""
    B = int(max(n_bins))
    key = np.zeros(len(Xb), dtype=np.int64)
    for j in range(Xb.shape[1]):
        key = key * B + Xb[:, j].astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    st = shortfall_stats(q)
    S = np.column_stack([np.bincount(inv, weights=st[:, c], minlength=len(uniq)) for c in range(4)])
    P = np.empty((len(uniq), Xb.shape[1]), dtype=np.int64)
    k = uniq.copy()
    for j in range(Xb.shape[1] - 1, -1, -1):
        P[:, j] = k % B
        k //= B
    return P, S, B


# ---------------------------------------------------------------- core vectorized pieces (shared by greedy / DP)
# `pen` is an additive per-leaf penalty in TOTAL-cost units (lambda * n_train). Because the penalized objective is
# still a sum over leaves, the exhaustive recursion stays exact; pen = 0.0 reproduces the original numbers exactly.
def _leaf(st, beta, m, pen=0.0):
    costs = stats_to_costs(st, beta)
    c = costs.min(-1) + pen
    return np.where(st[..., 0] >= m, c, np.inf), costs.argmin(-1)


def _best_stump(H, beta, m, pen=0.0):
    """H (..., B, 4): histogram along the split axis. Best cost over thresholds and the arg threshold."""
    cs = np.cumsum(H, axis=-2)
    L = cs[..., :-1, :]
    R = cs[..., -1:, :] - L
    tot = _leaf(L, beta, m, pen)[0] + _leaf(R, beta, m, pen)[0]
    t = np.argmin(tot, axis=-1)
    return np.take_along_axis(tot, t[..., None], -1)[..., 0], t


def _hist1(P, S, idx, f, B):
    return np.column_stack([np.bincount(P[idx, f], weights=S[idx, c], minlength=B) for c in range(4)])


def _better(c_new, c_old):
    return c_new < c_old - _TOL * max(1.0, abs(c_old))


def _cfg(c):
    """Internal config: (beta, min_leaf) or (beta, min_leaf, pen_total)."""
    return c[0], c[1], (c[2] if len(c) > 2 else 0.0)


def _solve_d1(P, S, idx, beta, m, B, pen=0.0):
    tot = S[idx].sum(0)
    lc, la = _leaf(tot, beta, m, pen)
    best_c, best_tree = float(lc), leaf_node(la)
    for f in range(P.shape[1]):
        c, t = _best_stump(_hist1(P, S, idx, f, B), beta, m, pen)
        if _better(c, best_c):
            left, right = idx[P[idx, f] <= t], idx[P[idx, f] > t]
            best_c = float(c)
            best_tree = split_node(f, t, leaf_node(_leaf(S[left].sum(0), beta, m)[1]),
                                   leaf_node(_leaf(S[right].sum(0), beta, m)[1]))
    return best_c, best_tree


def _d1_region(reg, beta, m, pen=0.0):
    """Best depth<=1 cost of regions. reg: (p_c, ..., B_c, 4), axis 0 = candidate split feature c."""
    lc = _leaf(reg[0].sum(-2), beta, m, pen)[0]
    sc = _best_stump(reg, beta, m, pen)[0].min(0)
    return np.minimum(lc, sc)


def _solve_d2(P, S, idx, beta, m, B, pen=0.0):
    best_c, best_tree = _solve_d1(P, S, idx, beta, m, B, pen)
    Pi, Si, p = P[idx], S[idx], P.shape[1]
    best_root = None
    for a in range(p):
        H2 = np.empty((p, B, B, 4))
        for c in range(p):
            comb = Pi[:, a] * B + Pi[:, c]
            for ch in range(4):
                H2[c, :, :, ch] = np.bincount(comb, weights=Si[:, ch], minlength=B * B).reshape(B, B)
        Pa = np.cumsum(H2, axis=1)
        Lr = Pa[:, :-1]
        Rr = Pa[:, -1:] - Lr
        root = _d1_region(Lr, beta, m, pen) + _d1_region(Rr, beta, m, pen)
        t0 = int(np.argmin(root))
        if _better(root[t0], best_c):
            best_c, best_root = float(root[t0]), (a, t0)
    if best_root is not None:
        a, t0 = best_root
        cl, tl = _solve_d1(P, S, idx[Pi[:, a] <= t0], beta, m, B, pen)
        cr, tr = _solve_d1(P, S, idx[Pi[:, a] > t0], beta, m, B, pen)
        assert np.isclose(cl + cr, best_c, rtol=1e-7), (cl + cr, best_c)
        best_tree = split_node(a, t0, tl, tr)
    return best_c, best_tree


def _solve_d3_multi(P, S, idx, B, configs):
    """Exact depth<=3 search for several (beta, min_leaf[, pen]) configs sharing one pass over the 3-D histograms.

    For root split (a, t0) and child split (b, t1) the four grandchild regions are rectangles in (a, b); 2-D prefix
    sums over the (a, b) bins of the joint histogram H[c, a_bin, b_bin, c_bin] give every region for every (t0, t1)
    at once, and a cumulative sum over c_bin gives every grandchild stump. Exhaustive => exact.
    """
    configs = [_cfg(c) for c in configs]
    Pi, Si, p = P[idx], S[idx], P.shape[1]
    K = len(configs)
    bestL = np.full((K, p, B - 1), np.inf)
    bestR = np.full((K, p, B - 1), np.inf)
    for a in range(p):
        for b in range(p):
            H3 = np.empty((p, B, B, B, 4))
            base = (Pi[:, a] * B + Pi[:, b]) * B
            for c in range(p):
                comb = base + Pi[:, c]
                for ch in range(4):
                    H3[c, ..., ch] = np.bincount(comb, weights=Si[:, ch], minlength=B ** 3).reshape(B, B, B)
            Pab = np.cumsum(np.cumsum(H3, axis=1), axis=2)
            A0 = Pab[:, :-1, :-1]           # a <= t0, b <= t1
            AL = Pab[:, :-1, -1:]           # a <= t0, any b
            AB = Pab[:, -1:, :-1]           # any a,  b <= t1
            AA = Pab[:, -1:, -1:]           # everything
            LL, LR, RL, RR = A0, AL - A0, AB - A0, AA - AL - AB + A0
            for k, (beta, m, pen) in enumerate(configs):
                bestL[k, a] = np.minimum(bestL[k, a], (_d1_region(LL, beta, m, pen)
                                                       + _d1_region(LR, beta, m, pen)).min(axis=1))
                bestR[k, a] = np.minimum(bestR[k, a], (_d1_region(RL, beta, m, pen)
                                                       + _d1_region(RR, beta, m, pen)).min(axis=1))
    results = []
    for k, (beta, m, pen) in enumerate(configs):
        best_c, best_tree = _solve_d2(P, S, idx, beta, m, B, pen)
        best_root = None
        for a in range(p):
            cs = np.cumsum(_hist1(P, S, idx, a, B), axis=0)    # child = leaf option
            root = (np.minimum(bestL[k, a], _leaf(cs[:-1], beta, m, pen)[0])
                    + np.minimum(bestR[k, a], _leaf(cs[-1:] - cs[:-1], beta, m, pen)[0]))
            t0 = int(np.argmin(root))
            if _better(root[t0], best_c):
                best_c, best_root = float(root[t0]), (a, t0)
        if best_root is not None:
            a, t0 = best_root
            cl, tl = _solve_d2(P, S, idx[Pi[:, a] <= t0], beta, m, B, pen)
            cr, tr = _solve_d2(P, S, idx[Pi[:, a] > t0], beta, m, B, pen)
            assert np.isclose(cl + cr, best_c, rtol=1e-7), (cl + cr, best_c)
            best_tree = split_node(a, t0, tl, tr)
        results.append((best_c, best_tree))
    return results


def _solve_d4_multi(P, S, idx, B, configs, deadline=None, progress=None):
    """Exact depth<=4 search: every depth<=4 tree is either a depth<=3 tree or a root split whose two children are
    depth<=3 trees, and the objective is separable across the children, so solving both children exactly for every
    root split is exact. Root splits that leave a child with fewer than min_leaf rows (for every config) are
    infeasible and skipped. With a deadline the search may stop early: the result is then the best tree over all
    depth<=3 trees and the evaluated root splits (a valid incumbent, NOT certified)."""
    configs = [_cfg(c) for c in configs]
    best = [list(x) for x in _solve_d3_multi(P, S, idx, B, configs)]
    Pi = P[idx]
    m_min = min(c[1] for c in configs)
    b0, m0, p0 = configs[0]
    cand = []
    for a in range(P.shape[1]):
        H = _hist1(P, S, idx, a, B)
        cs = np.cumsum(H, axis=0)
        for t0 in range(B - 1):
            if cs[t0, 0] >= m_min and cs[-1, 0] - cs[t0, 0] >= m_min:
                stump = float(_leaf(cs[t0], b0, m0, p0)[0] + _leaf(cs[-1] - cs[t0], b0, m0, p0)[0])
                cand.append((stump, a, t0))
    cand.sort()                         # promising root splits first, so a partial run keeps a good incumbent
    done = 0
    for _, a, t0 in cand:
        if deadline is not None and time.perf_counter() > deadline:
            break
        Ls = _solve_d3_multi(P, S, idx[Pi[:, a] <= t0], B, configs)
        Rs = _solve_d3_multi(P, S, idx[Pi[:, a] > t0], B, configs)
        for k in range(len(configs)):
            c = Ls[k][0] + Rs[k][0]
            if _better(c, best[k][0]):
                best[k] = [c, split_node(a, t0, Ls[k][1], Rs[k][1])]
        done += 1
        if progress is not None:
            progress(done, len(cand))
    return [tuple(x) for x in best], {"root_candidates": len(cand), "evaluated": done, "complete": done == len(cand)}


def solve_opt_exact(P, S, B, depth, configs, leaf_penalty=0.0, time_limit=None, progress=None):
    """Globally optimal policy trees (depth <= 4) over the candidate split set, by exhaustive DP on patterns.

    configs: list of (beta, min_leaf) or (beta, min_leaf, lambda). lambda (default `leaf_penalty`) is a penalty per
    leaf in MEAN-cost units: the solver minimizes mean training cost + lambda * (#leaves). Returns one dict per config;
    "train_total_cost" is always the UNPENALIZED cost of the returned tree, "penalized_objective_total" includes the
    penalty. Depth 4 accepts a wall-clock `time_limit` (seconds); if it stops early the status says TIME_LIMIT and
    the tree is the best found, not a certified optimum. Runtime is the call's wall time divided across configs.
    """
    if depth not in (0, 1, 2, 3, 4):
        raise ValueError("exact DP implemented for depth <= 4")
    t0 = time.perf_counter()
    n = S[:, 0].sum()
    lams = [c[2] if len(c) > 2 else leaf_penalty for c in configs]
    internal = [(c[0], c[1], lam * n) for c, lam in zip(configs, lams)]
    idx = np.arange(len(P))
    info = {"complete": True}
    if depth == 4:
        deadline = None if time_limit is None else t0 + time_limit
        sols, info = _solve_d4_multi(P, S, idx, B, internal, deadline, progress)
    elif depth == 3:
        sols = _solve_d3_multi(P, S, idx, B, internal)
    elif depth == 0:
        sols = [(float(_leaf(S.sum(0), b, m, pen)[0]), leaf_node(_leaf(S.sum(0), b, m)[1])) for b, m, pen in internal]
    else:
        sols = [{1: _solve_d1, 2: _solve_d2}[depth](P, S, idx, b, m, B, pen) for b, m, pen in internal]
    rt = (time.perf_counter() - t0) / len(configs)
    out = []
    for (b, m, pen), lam, (c, tr) in zip(internal, lams, sols):
        L = tree_leaves(tr)
        cost = c - pen * L                       # unpenalized training cost (identical to c when lambda = 0)
        status = ("OPTIMAL (exhaustive DP over candidate splits)" if info["complete"] else
                  f"TIME_LIMIT: best of all depth<=3 trees and {info['evaluated']}/{info['root_candidates']} "
                  "depth-4 root splits (not certified)")
        out.append({"beta": b, "min_leaf": m, "depth": depth, "tree": tr, "train_total_cost": cost,
                    "train_mean_cost": cost / n, "leaf_penalty": lam, "leaves": L, "penalized_objective_total": c,
                    "penalized_objective_mean": c / n, "runtime_s": rt, "status": status,
                    "bound": c if info["complete"] else np.nan, "gap": 0.0 if info["complete"] else np.nan}
                   | ({k: v for k, v in info.items() if k != "complete"} if depth == 4 else {}))
    return out


def fit_greedy_tree(P, S, B, beta, depth, min_leaf):
    """Objective-aligned greedy tree: each node takes the single split that most reduces modeled cost."""
    t0 = time.perf_counter()

    def rec(idx, d):
        lc, la = _leaf(S[idx].sum(0), beta, min_leaf)
        if d == 0:
            return float(lc), leaf_node(la)
        best_c, best = float(lc), None
        for f in range(P.shape[1]):
            c, t = _best_stump(_hist1(P, S, idx, f, B), beta, min_leaf)
            if _better(c, best_c):
                best_c, best = float(c), (f, int(t))
        if best is None:
            return float(lc), leaf_node(la)
        f, t = best
        cl, tl = rec(idx[P[idx, f] <= t], d - 1)
        cr, tr = rec(idx[P[idx, f] > t], d - 1)
        return cl + cr, split_node(f, t, tl, tr)

    cost, tree = rec(np.arange(len(P)), depth)
    return {"tree": tree, "train_total_cost": cost, "train_mean_cost": cost / S[:, 0].sum(),
            "runtime_s": time.perf_counter() - t0, "status": "greedy (no optimality guarantee)"}


def tree_train_cost(tree, P, S, beta):
    """Total training cost of a tree evaluated on aggregated patterns (leaf actions as stored in the tree)."""
    act = predict_tree(tree, P)
    costs = stats_to_costs(S, beta)
    return float(costs[np.arange(len(P)), act].sum())


# ---------------------------------------------------------------- MIO formulation (Gurobi)
def solve_opt_mio(P, S, n_bins, beta, depth, min_leaf, time_limit=600, threads=8, warm_tree=None, verbose=False):
    """Full binary tree of the given depth; each internal node picks one candidate split (feature j, bin t) or a
    dummy 'send everything left' split, so shallower trees are representable (same policy class as the DP).
    Leaves may be empty; non-empty leaves need >= min_leaf training VMs. Patterns (aggregated rows) are routed.
      min  sum_{u,l,k} c_uk y_ulk
      s.t. sum_s d_ns = 1;  g_un = sum_s a_us d_ns;  z_ul <= g_un (left ancestors), z_ul <= 1 - g_un (right);
           sum_l z_ul = 1;  sum_k w_lk = 1;  sum_k y_ulk = z_ul;  y_ulk <= w_lk;
           z_ul <= e_l;  sum_u n_u z_ul >= min_leaf * e_l.
    """
    import gurobipy as gp
    from gurobipy import GRB

    U, p = P.shape
    C = stats_to_costs(S, beta)
    cnt = S[:, 0]
    splits = [(-1, -1)] + [(j, t) for j in range(p) for t in range(int(n_bins[j]) - 1)]
    A = np.ones((U, len(splits)))
    for s, (j, t) in enumerate(splits[1:], start=1):
        A[:, s] = (P[:, j] <= t).astype(float)
    n_int, leaves = 2 ** depth - 1, list(range(2 ** depth - 1, 2 ** (depth + 1) - 1))  # 0-indexed heap

    m = gp.Model("opt_policy_tree")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit, m.Params.Threads, m.Params.Seed = time_limit, threads, 0
    d = m.addVars(n_int, len(splits), vtype=GRB.BINARY, name="d")
    g = m.addVars(U, n_int, lb=0, ub=1, name="g")
    z = m.addVars(U, len(leaves), lb=0, ub=1, name="z")
    w = m.addVars(len(leaves), 4, vtype=GRB.BINARY, name="w")
    e = m.addVars(len(leaves), vtype=GRB.BINARY, name="e")
    y = m.addVars(U, len(leaves), 4, lb=0, ub=1, name="y")
    m.addConstrs(d.sum(n, "*") == 1 for n in range(n_int))
    for n in range(n_int):
        for u in range(U):
            nz = np.flatnonzero(A[u])
            m.addConstr(g[u, n] == gp.quicksum(d[n, s] for s in nz))
    for li, leaf in enumerate(leaves):
        node = leaf
        while node > 0:
            parent = (node - 1) // 2
            is_left = node == 2 * parent + 1
            for u in range(U):
                m.addConstr(z[u, li] <= (g[u, parent] if is_left else 1 - g[u, parent]))
            node = parent
        m.addConstr(w.sum(li, "*") == 1)
        m.addConstrs(z[u, li] <= e[li] for u in range(U))
        m.addConstr(gp.quicksum(cnt[u] * z[u, li] for u in range(U)) >= min_leaf * e[li])
    m.addConstrs(z.sum(u, "*") == 1 for u in range(U))
    m.addConstrs(y.sum(u, li, "*") == z[u, li] for u in range(U) for li in range(len(leaves)))
    m.addConstrs(y[u, li, k] <= w[li, k] for u in range(U) for li in range(len(leaves)) for k in range(4))
    m.setObjective(gp.quicksum(C[u, k] * y[u, li, k] for u in range(U) for li in range(len(leaves)) for k in range(4)),
                   GRB.MINIMIZE)

    if warm_tree is not None:   # pad the tree to full depth with dummy splits
        def assign(node_tree, heap):
            if heap >= n_int:
                w[heap - n_int, node_tree["action"]].Start = 1
                return
            if node_tree["leaf"]:
                d[heap, 0].Start = 1
                assign(node_tree, 2 * heap + 1)
                for lf in _leaves_under(2 * heap + 2, n_int):
                    w[lf - n_int, node_tree["action"]].Start = 1
                for nd in _internal_under(2 * heap + 2, n_int):
                    d[nd, 0].Start = 1
                return
            d[heap, splits.index((node_tree["feature"], node_tree["t"]))].Start = 1
            assign(node_tree["left"], 2 * heap + 1)
            assign(node_tree["right"], 2 * heap + 2)
        assign(warm_tree, 0)

    t0 = time.perf_counter()
    m.optimize()
    rt = time.perf_counter() - t0
    status = {GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIME_LIMIT"}.get(m.Status, str(m.Status))
    res = {"status": status, "runtime_s": rt, "n_vars": m.NumVars, "n_constrs": m.NumConstrs,
           "n_patterns": U, "n_split_candidates": len(splits) - 1}
    if m.SolCount > 0:
        res |= {"train_total_cost": m.ObjVal, "bound": m.ObjBound, "gap": m.MIPGap,
                "tree": _mio_tree(d, w, e, splits, depth, n_int)}
    else:
        res |= {"train_total_cost": np.nan, "bound": m.ObjBound, "gap": np.nan, "tree": None}
    return res


def _leaves_under(heap, n_int):
    if heap >= n_int:
        return [heap]
    return _leaves_under(2 * heap + 1, n_int) + _leaves_under(2 * heap + 2, n_int)


def _internal_under(heap, n_int):
    if heap >= n_int:
        return []
    return [heap] + _internal_under(2 * heap + 1, n_int) + _internal_under(2 * heap + 2, n_int)


def _mio_tree(d, w, e, splits, depth, n_int):
    def rec(heap):
        if heap >= n_int:
            return leaf_node(int(np.argmax([w[heap - n_int, k].X for k in range(4)])))
        s = int(np.argmax([d[heap, s].X for s in range(len(splits))]))
        if s == 0:  # dummy split: everything goes left
            return rec(2 * heap + 1)
        j, t = splits[s]
        return split_node(j, t, rec(2 * heap + 1), rec(2 * heap + 2))
    return rec(0)


# ---------------------------------------------------------------- Interpretable AI bridge (Julia)
def _default_julia():
    return os.environ.get("IAI_JULIA", str(Path.home() / ".julia/juliaup/julia-1.11.9+0.aarch64.apple.darwin14/"
                                           "Julia-1.11.app/Contents/Resources/julia/bin/julia"))


def _default_sysimage():
    return os.environ.get("IAI_SYSIMAGE", str(Path.home() / "Library/Application Support/InterpretableAI/sysimage/"
                                              "v3.2.2/sys.dylib"))


def iai_available():
    return Path(_default_julia()).exists() and Path(_default_sysimage()).exists() and (Path.home() / "iai.lic").exists()


def run_iai_policy_trees(Xb_train, C_train, configs, workdir, threads=10):
    """Fit IAI OptimalTreePolicyMinimizer (cp = 0) for each config dict(depth, minbucket, seed).

    Features are passed as the integer bin indices, so IAI's candidate thresholds coincide with the DP's.
    Returns a list of dicts with the tree (our representation), train cost and runtime.
    """
    import hashlib
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    Xf, Cf = Xb_train.astype(np.float64).T.copy(), C_train.astype(np.float64).T.copy()   # column-major for Julia
    cfg_txt = "".join(f"{c['depth']} {c['minbucket']} {c['seed']}\n" for c in configs)
    script = Path(__file__).with_name("iai_opt.jl")
    digest = hashlib.sha256(Xf.tobytes() + Cf.tobytes() + cfg_txt.encode() + script.read_bytes()).hexdigest()
    stamp = workdir / "done.sha256"
    if not (stamp.exists() and stamp.read_text() == digest):   # identical inputs already fitted -> reuse (seeded)
        stamp.unlink(missing_ok=True)
        Xf.tofile(workdir / "X.bin")
        Cf.tofile(workdir / "C.bin")
        (workdir / "dims.txt").write_text(f"{Xb_train.shape[0]} {Xb_train.shape[1]}\n")
        (workdir / "configs.txt").write_text(cfg_txt)
        proc = subprocess.run([_default_julia(), f"--sysimage={_default_sysimage()}", "-t", str(threads), str(script),
                               str(workdir)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"IAI run failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
        stamp.write_text(digest)
    out = []
    for i, cfg in enumerate(configs):
        js = json.loads((workdir / f"tree_{i}.json").read_text())
        out.append({**cfg, "tree": _iai_tree(js), "runtime_s": js["runtime_s"], "iai_version": js.get("version", "")})
    return out


def _iai_tree(js):
    nodes = {n["id"]: n for n in js["nodes"]}

    def rec(i):
        n = nodes[i]
        if n["leaf"]:
            return leaf_node(n["action"] - 1)
        # IAI sends x < threshold to the lower child; with integer bins this is bin <= ceil(threshold) - 1
        return split_node(n["feature"] - 1, int(np.ceil(n["threshold"])) - 1, rec(n["lower"]), rec(n["upper"]))
    return rec(1)
