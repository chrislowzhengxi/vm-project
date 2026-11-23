"""Metric helpers for stage 12 (extended evaluation beyond modeled cost). Pure numpy, no data access.

Conventions: utilization Y (realized P95) and reservation r are fractions of the VM's CPU capacity; shortfall
sf = max(0, Y - r); a VM is "exceeded" when sf > EPS (same 1e-12 tolerance as final_lib.evaluate). Optional
weights w are evaluation weights only (e.g. 1 per VM, or 1 / n_sub for equal-subscription weighting).
"""
import numpy as np

import final_lib as fl

EPS = 1e-12
SEVERE = (0.05, 0.10, 0.20)          # shortfall thresholds: 5, 10, 20 percentage points of VM capacity
QUANTS = (0.90, 0.95, 0.99)


# ---------------------------------------------------------------- basic statistics
def wmean(x, w=None):
    x = np.asarray(x, float)
    return float(x.mean()) if w is None else float(np.dot(np.asarray(w, float), x) / np.sum(w))


def wquantile(x, p, w=None):
    """Weighted quantile, inverted-CDF definition (smallest x with F(x) >= p); equals
    np.quantile(x, p, method='inverted_cdf') for unit weights. nan for empty input."""
    x = np.asarray(x, float)
    p = np.atleast_1d(np.asarray(p, float))
    if x.size == 0:
        return np.full(p.shape, np.nan)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    o = np.argsort(x, kind="stable")
    xs, cw = x[o], np.cumsum(w[o])
    cw = cw / cw[-1]
    idx = np.searchsorted(cw, p - 1e-12, side="left")
    return xs[np.clip(idx, 0, len(xs) - 1)]


# ---------------------------------------------------------------- operational metrics of one action vector
def policy_metrics(a, q, beta, w=None, thresholds=SEVERE, quants=QUANTS):
    """Reservation, exceedance, shortfall-severity and cost-decomposition metrics (fractions, not percent).

    cost = E[r] + beta * E[sf]  (reservation cost + shortfall penalty). Conditional statistics are over exceeded VMs
    only and are nan when no VM is exceeded. 'max shortfall' is an extreme descriptive statistic (one VM).
    """
    a, q = np.asarray(a, float), fl.check_fraction(q)
    w = np.ones_like(q) if w is None else np.asarray(w, float)
    sf = np.maximum(0.0, q - a)
    exc = sf > EPS
    out = {"E[r]": wmean(a, w), "E[sf]": wmean(sf, w), "P(Y>r)": wmean(exc, w), "E[unused]": wmean(np.maximum(0, a - q), w)}
    out["reservation cost"] = out["E[r]"]
    out["shortfall penalty"] = beta * out["E[sf]"]
    out["cost"] = out["reservation cost"] + out["shortfall penalty"]
    out["E[sf | Y>r]"] = out["E[sf]"] / out["P(Y>r)"] if out["P(Y>r)"] > 0 else np.nan
    for p, v in zip(quants, wquantile(sf, quants, w)):
        out[f"sf P{int(round(100 * p))}"] = float(v)
    cq = wquantile(sf[exc], (0.5,) + tuple(quants), w[exc]) if exc.any() else np.full(len(quants) + 1, np.nan)
    for p, v in zip((0.5,) + tuple(quants), cq):
        out[f"sf P{int(round(100 * p))} | Y>r"] = float(v)
    out["max sf"] = float(sf.max()) if sf.size else np.nan
    for t in thresholds:
        out[f"P(sf>{int(round(100 * t))}pp)"] = wmean(sf > t + EPS, w)
    for t in fl.TIERS:
        out[f"share {int(round(100 * t))}%"] = wmean(np.isclose(a, t), w)
    return out


def oracle_regret(c_pol, c_fixed, c_oracle):
    """Regret vs the hindsight oracle and the share of the attainable (fixed -> oracle) saving captured."""
    gap = c_fixed - c_oracle
    return {"regret": c_pol - c_oracle, "regret % of oracle cost": 100 * (c_pol - c_oracle) / c_oracle,
            "% saving vs fixed": 100 * (c_fixed - c_pol) / c_fixed,
            "% of attainable saving captured": 100 * (c_fixed - c_pol) / gap if abs(gap) > EPS else np.nan}


# ---------------------------------------------------------------- "reserving less" vs "targeting"
def tier_shortfalls(q, w=None):
    """E[(Y - t)+] for each tier t (a fixed tier's expected shortfall)."""
    q = fl.check_fraction(q)
    return np.array([wmean(np.maximum(0, q - t), w) for t in fl.TIERS])


def untargeted_shortfall(a, q, w=None):
    """Expected shortfall if the policy's own tier mix were assigned to VMs at random (independently of the VM):
    sum_k share_k * E[(Y - t_k)+]. Equals the average over all permutations of the action vector."""
    a = np.asarray(a, float)
    shares = np.array([wmean(np.isclose(a, t), w) for t in fl.TIERS])
    return float(shares @ tier_shortfalls(q, w))


def lower_hull(x, y):
    """Lower convex hull of points (x, y) sorted by x: the frontier of randomized mixtures (min y for each x)."""
    pts = sorted(set(zip(np.asarray(x, float), np.asarray(y, float))))
    hull = []
    for p in pts:
        while len(hull) >= 2 and ((hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1])
                                  - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0])) <= 0:
            hull.pop()
        if hull and np.isclose(hull[-1][0], p[0]):
            continue
        hull.append(p)
    return np.array(hull)


def mixture_frontier(x, y, x0):
    """Smallest y attainable by randomizing over the given configurations with risk at most x0 (x and y are both
    'lower is better', e.g. x = expected shortfall, y = mean reservation). nan if x0 is below every configuration."""
    h = lower_hull(x, y)
    if x0 < h[0, 0] - EPS:
        return np.nan
    xs = np.clip(x0, h[0, 0], h[-1, 0])
    v = float(np.interp(xs, h[:, 0], h[:, 1]))
    return min(v, float(h[h[:, 0] <= x0 + EPS, 1].min()))


def fixed_mixture_reservation(q, sf_target, w=None):
    """Lowest mean reservation of a randomized fixed-tier rule with expected shortfall <= sf_target."""
    return mixture_frontier(tier_shortfalls(q, w), fl.TIERS, sf_target)


def fixed_mixture_shortfall(q, r_mean, w=None):
    """Expected shortfall of the best randomized fixed-tier rule with mean reservation r_mean (0.25 <= r_mean <= 1)."""
    return float(np.interp(r_mean, fl.TIERS, tier_shortfalls(q, w)))


# ---------------------------------------------------------------- Pareto dominance
def pareto_dominated(P, tol=1e-12):
    """P: (n, d) array, every column 'lower is better'. Row i is dominated if some row j is <= in all columns and
    strictly < (by more than tol) in at least one."""
    P = np.asarray(P, float)
    out = np.zeros(len(P), bool)
    for i in range(len(P)):
        le = np.all(P <= P[i] + tol, axis=1)
        lt = np.any(P < P[i] - tol, axis=1)
        out[i] = bool(np.any(le & lt))
    return out


# ---------------------------------------------------------------- groups (subscriptions), concentration
def group_sums(groups, **cols):
    """Per-group sums of each column (plus 'n'). Returns (unique groups, dict of arrays)."""
    g, inv = np.unique(np.asarray(groups), return_inverse=True)
    out = {"n": np.bincount(inv).astype(float)}
    for k, v in cols.items():
        out[k] = np.bincount(inv, weights=np.asarray(v, float), minlength=len(g))
    return g, out


def equal_group_weights(groups):
    """Per-row weights 1 / (rows in group): every group counts once (equal-subscription weighting)."""
    _, inv, cnt = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def top_share(contrib, frac=None, k=None):
    """Share of sum(|contrib|) coming from the largest-|contrib| units (top fraction `frac` or top `k` units)."""
    c = np.sort(np.abs(np.asarray(contrib, float)))[::-1]
    if c.sum() <= 0:
        return np.nan
    k = int(np.ceil(frac * len(c))) if k is None else int(k)
    return float(c[:k].sum() / c.sum())


# ---------------------------------------------------------------- disagreement between two policies
def disagreement(a1, a2, q, beta, groups=None, top_frac=0.10, top_groups=10):
    """How two action vectors differ and where the cost difference (policy 1 - policy 2) comes from.

    Every VM where the actions agree contributes exactly zero to the cost difference, so the whole difference lives
    in the disagreement set D. Reports its size, direction, reservation / cost / shortfall differences inside D and
    the concentration of the gross absolute difference.
    """
    a1, a2, q = np.asarray(a1, float), np.asarray(a2, float), fl.check_fraction(q)
    c1, c2 = a1 + beta * np.maximum(0, q - a1), a2 + beta * np.maximum(0, q - a2)
    d = ~np.isclose(a1, a2)
    n = len(q)
    dc = c1 - c2
    out = {"% VMs disagree": 100 * d.mean(), "% policy 1 higher": 100 * np.mean(a1 > a2 + EPS),
           "% policy 1 lower": 100 * np.mean(a1 < a2 - EPS),
           "net cost diff per VM (1-2)": float(dc.sum() / n), "gross |cost diff| per VM": float(np.abs(dc).sum() / n)}
    if d.any():
        out |= {"mean r diff in D (pp)": 100 * float((a1 - a2)[d].mean()),
                "mean |r diff| in D (pp)": 100 * float(np.abs(a1 - a2)[d].mean()),
                "mean cost diff in D": float(dc[d].mean()),
                "P(Y>r) in D, policy 1 %": 100 * float(np.mean(q[d] > a1[d] + EPS)),
                "P(Y>r) in D, policy 2 %": 100 * float(np.mean(q[d] > a2[d] + EPS)),
                "E[sf] in D, policy 1 (pp)": 100 * float(np.maximum(0, q[d] - a1[d]).mean()),
                "E[sf] in D, policy 2 (pp)": 100 * float(np.maximum(0, q[d] - a2[d]).mean()),
                f"share of gross |diff| in top {int(100 * top_frac)}% of D": top_share(dc[d], frac=top_frac),
                "% of D where policy 1 cheaper": 100 * float(np.mean(dc[d] < -EPS))}
        if groups is not None:
            _, s = group_sums(np.asarray(groups)[d], dc=dc[d])
            out[f"subscriptions in D"] = len(s["n"])
            out[f"share of gross |sub diff| in top {top_groups} subs"] = top_share(s["dc"], k=top_groups)
    return out


# ---------------------------------------------------------------- paired cluster bootstrap for several metrics
def cluster_bootstrap_multi(diffs, clusters, n_boot=1000, seed=0):
    """Paired subscription-cluster bootstrap of the mean of several per-VM difference vectors with ONE shared set of
    resamples. Uses exactly the random draws of final_lib.cluster_bootstrap_diff (same seed -> identical CI for a
    single metric). Percentile 95% CIs; unadjusted for multiplicity; conditional on the trained models."""
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    nd = np.bincount(inv).astype(float)
    G = len(nd)
    keys = list(diffs)
    S = np.column_stack([np.bincount(inv, weights=np.asarray(diffs[k], float), minlength=G) for k in keys])
    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, len(keys)))
    for r in range(n_boot):
        cnt = np.bincount(rng.integers(0, G, G), minlength=G)
        reps[r] = (cnt @ S) / (cnt @ nd)
    out = {}
    for j, k in enumerate(keys):
        out[k] = {"mean diff": float(np.mean(diffs[k])), "ci low": float(np.percentile(reps[:, j], 2.5)),
                  "ci high": float(np.percentile(reps[:, j], 97.5)), "share reps < 0": float(np.mean(reps[:, j] < 0))}
    return out
