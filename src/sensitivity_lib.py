"""Beta-sensitivity and complexity helpers (stages 10-11).

Fixed policies. For a fixed action vector r, the mean modeled cost is linear in beta:
    E[C(r, Y; beta)] = E[r] + beta * E[(Y - r)+]  =  intercept + beta * slope.
Two fixed policies A, B cost the same at beta* = (E[r_B] - E[r_A]) / (E[(Y-r_A)+] - E[(Y-r_B)+]).

Optimal trees. For a fixed candidate tree T (structure AND leaf actions) the training cost is A_T + theta * B_T, so the
optimal training value V(theta) = min_T (A_T + theta * B_T) is a minimum of lines: concave and piecewise linear. Its
breakpoints are exactly where the optimal tree changes. `parametric_path` recovers all of them with an exact solver
(Eisner-Severance style): solve at the interval ends; if the two optimal lines differ, solve at their intersection; if
nothing beats them there they are adjacent on the envelope, otherwise recurse on both halves. Along the path the slope
B is non-increasing and the intercept A non-decreasing in theta (concavity). theta = beta (A = reserved capacity,
B = shortfall) or theta = lambda, the per-leaf penalty (A = unpenalized cost, B = n * #leaves).
"""
import time

import numpy as np

import final_lib as fl


# ---------------------------------------------------------------- fixed policies: linear in beta
def line_coeffs(a, q, w=None):
    """(intercept, slope) of a fixed policy's mean cost in beta: E[r], E[(Y - r)+] (optionally weighted)."""
    a, q = np.asarray(a, float), fl.check_fraction(q)
    w = np.ones_like(q) if w is None else np.asarray(w, float)
    return float(np.average(a, weights=w)), float(np.average(np.maximum(0, q - a), weights=w))


def cost_at(coef, beta):
    return coef[0] + np.asarray(beta, float) * coef[1]


def break_even(cA, cB):
    """beta where policies A and B have equal cost (nan if parallel)."""
    d = cA[1] - cB[1]
    return np.nan if abs(d) < 1e-15 else (cB[0] - cA[0]) / d


def lower_envelope(lines, lo, hi):
    """lines: dict label -> (intercept, slope). Returns [(start, end, label)] of argmin over beta in [lo, hi]."""
    labs = list(lines)
    pts = {lo, hi}
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            b = break_even(lines[labs[i]], lines[labs[j]])
            if np.isfinite(b) and lo < b < hi:
                pts.add(float(b))
    pts = sorted(pts)
    segs = []
    for s, e in zip(pts[:-1], pts[1:]):
        mid = 0.5 * (s + e)
        lab = min(labs, key=lambda k: cost_at(lines[k], mid))
        if segs and segs[-1][2] == lab:
            segs[-1] = (segs[-1][0], e, lab)
        else:
            segs.append((s, e, lab))
    return segs


# ---------------------------------------------------------------- trees on aggregated training patterns
def tree_line(tree, P, S):
    """Totals (A, B) of a fixed tree on patterns: A = sum n * tier, B = sum shortfall at the chosen tier."""
    act = fl.predict_tree(tree, P)
    A = float((S[:, 0] * fl.TIERS[act]).sum())
    sf = np.where(act < 3, S[np.arange(len(P)), np.minimum(act, 2) + 1], 0.0)
    return A, float(sf.sum())


def leaf_table(tree, P, S, names=None, thresholds=None):
    """One row per leaf: path, n and shortfall sums (training statistics) and the stored action."""
    rows = []

    def rec(node, mask, path):
        if node["leaf"]:
            st = S[mask].sum(0)
            rows.append({"path": " & ".join(path) or "(root)", "action": float(fl.TIERS[node["action"]]),
                         "n": st[0], "S25": st[1], "S50": st[2], "S75": st[3]})
            return
        f, t = node["feature"], node["t"]
        nm = names[f] if names else f"x{f}"
        thr = f"{thresholds[f][t]:.4g}" if thresholds else f"bin {t}"
        go = mask & (P[:, f] <= t)
        rec(node["left"], go, path + [f"{nm} <= {thr}"])
        rec(node["right"], mask & ~go, path + [f"{nm} > {thr}"])

    rec(tree, np.ones(len(P), bool), [])
    return rows


def leaf_action_segments(n, S25, S50, S75, lo, hi):
    """beta intervals on which each tier is the cheapest action for a leaf with these training statistics."""
    lines = {float(t): (float(t), s / n) for t, s in zip(fl.TIERS, (S25, S50, S75, 0.0))}   # per-VM cost in the leaf
    return lower_envelope(lines, lo, hi)


# ---------------------------------------------------------------- exact parametric breakpoint search
def parametric_path(solve, lo, hi, max_solves=200, time_limit=None, rtol=1e-9):
    """All breakpoints of V(theta) = min_T (A_T + theta * B_T) on [lo, hi] with an exact solver.

    solve(theta) -> dict with totals "A", "B" and any payload (tree, signature...). Returns segments
    [(start, end, record)], the breakpoints, the number of solves, and any intervals left unresolved because the
    solve / time budget ran out (then coverage is NOT exhaustive and the result says so).
    """
    t0 = time.perf_counter()
    n_solves = 0

    def call(th):
        nonlocal n_solves
        n_solves += 1
        r = solve(th)
        r["theta_solved"] = th
        return r

    def same(L, R):
        return (abs(L["A"] - R["A"]) <= rtol * max(1.0, abs(L["A"]))
                and abs(L["B"] - R["B"]) <= rtol * max(1.0, abs(L["A"]), abs(L["B"])))

    lo_r, hi_r = call(lo), call(hi)
    bps, unresolved = [], []
    stack = [(lo, lo_r, hi, hi_r)]
    while stack:
        a, L, b, R = stack.pop()
        if same(L, R):
            continue
        if L["B"] - R["B"] <= rtol * max(1.0, abs(L["B"])):     # parallel lines: cannot both be optimal unless same
            unresolved.append((a, b, "parallel lines"))
            continue
        th = (R["A"] - L["A"]) / (L["B"] - R["B"])
        th = min(max(th, a), b)
        if n_solves >= max_solves or (time_limit is not None and time.perf_counter() - t0 > time_limit):
            unresolved.append((a, b, "budget exhausted"))
            continue
        M = call(th)
        vM, vL = M["A"] + th * M["B"], L["A"] + th * L["B"]
        if vM >= vL - rtol * max(1.0, abs(vL)):
            bps.append((th, L, R))
        else:
            stack.append((th, M, b, R))
            stack.append((a, L, th, M))
    bps.sort(key=lambda x: x[0])
    segs, cur, start = [], lo_r, lo
    for th, L, R in bps:
        segs.append((start, th, L))
        start, cur = th, R
    segs.append((start, hi, cur))
    return {"segments": segs, "breakpoints": [b[0] for b in bps], "n_solves": n_solves,
            "unresolved": unresolved, "complete": not unresolved, "runtime_s": time.perf_counter() - t0}
