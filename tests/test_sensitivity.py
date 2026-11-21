"""Tests for the beta-sensitivity and OPT-complexity extensions. Run:  python -m pytest -q tests/test_sensitivity.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import final_lib as fl  # noqa: E402
import sensitivity_lib as sl  # noqa: E402

PROC = ROOT / "data" / "processed"


def _toy(n=400, p=2, B=4, seed=0):
    rng = np.random.default_rng(seed)
    Xb = rng.integers(0, B, size=(n, p)).astype(np.int16)
    q = np.clip(0.18 * Xb[:, 0] + 0.08 * (Xb[:, 1] > 1) + rng.uniform(0, 0.45, n), 0, 1)
    P, S, BB = fl.aggregate(Xb, q, np.full(p, B))
    return Xb, q, P, S, BB


def _brute(P, S, B, beta, m, depth, pen=0.0, mask=None):
    """Definition of the optimum: every tree of depth <= `depth`, each leaf paying `pen` (total units)."""
    mask = np.ones(len(P), bool) if mask is None else mask
    st = S[mask].sum(0)
    best = fl.stats_to_costs(st, beta).min() + pen if st[0] >= m else np.inf
    if depth == 0:
        return best
    for j in range(P.shape[1]):
        for t in range(B - 1):
            left = mask & (P[:, j] <= t)
            best = min(best, _brute(P, S, B, beta, m, depth - 1, pen, left)
                       + _brute(P, S, B, beta, m, depth - 1, pen, mask & ~left))
    return best


# ---------------------------------------------------------------- fixed-policy algebra
def test_fixed_policy_cost_is_linear_in_beta():
    rng = np.random.default_rng(1)
    q = rng.uniform(0, 1, 3000)
    a = fl.TIERS[rng.integers(0, 4, 3000)]
    coef = sl.line_coeffs(a, q)
    for beta in (1, 1.5, 2, 3.7, 5, 10, 20):
        assert np.isclose(np.mean(a + beta * np.maximum(0, q - a)), sl.cost_at(coef, beta))
    w = rng.uniform(1, 5, 3000)
    cw = sl.line_coeffs(a, q, w)
    assert np.isclose(np.average(a + 4 * np.maximum(0, q - a), weights=w), sl.cost_at(cw, 4))


def test_break_even_and_lower_envelope():
    rng = np.random.default_rng(2)
    q = rng.uniform(0, 1, 5000)
    lines = {float(t): sl.line_coeffs(np.full_like(q, t), q) for t in fl.TIERS}
    be = sl.break_even(lines[0.75], lines[1.0])
    assert np.isclose(sl.cost_at(lines[0.75], be), sl.cost_at(lines[1.0], be))
    segs = sl.lower_envelope(lines, 1.0, 20.0)
    for beta in np.linspace(1.01, 19.99, 300):
        lab = [s[2] for s in segs if s[0] <= beta <= s[1]][0]
        assert np.isclose(sl.cost_at(lines[lab], beta), min(sl.cost_at(c, beta) for c in lines.values()))
    starts = [s[2] for s in segs]
    assert starts == sorted(starts)                 # best fixed tier never decreases as beta increases


# ---------------------------------------------------------------- exact path search
def _beta_solver(P, S, B, depth, m):
    def solve(beta):
        r = fl.solve_opt_exact(P, S, B, depth, [(beta, m)])[0]
        A, Bt = sl.tree_line(r["tree"], P, S)
        assert np.isclose(A + beta * Bt, r["train_total_cost"])
        return {"A": A, "B": Bt, "tree": r["tree"]}
    return solve


@pytest.mark.parametrize("depth,m", [(2, 5), (3, 15)])
def test_beta_path_matches_dense_grid_and_is_monotone(depth, m):
    _, _, P, S, B = _toy(seed=3)
    solve = _beta_solver(P, S, B, depth, m)
    path = sl.parametric_path(solve, 1.0, 20.0)
    assert path["complete"]
    segs = path["segments"]
    for beta in np.linspace(1.0, 20.0, 120):
        v_true = fl.solve_opt_exact(P, S, B, depth, [(beta, m)])[0]["train_total_cost"]
        v_env = min(r["A"] + beta * r["B"] for _, _, r in segs)
        assert np.isclose(v_true, v_env, rtol=1e-9)
    A = [r["A"] for _, _, r in segs]
    Bs = [r["B"] for _, _, r in segs]
    assert all(np.diff(A) >= -1e-9) and all(np.diff(Bs) <= 1e-9)          # reserve more / fall short less


def test_leaf_penalty_path_matches_dense_grid():
    _, _, P, S, B = _toy(seed=4)
    n = S[:, 0].sum()

    def solve(lam):
        r = fl.solve_opt_exact(P, S, B, 3, [(5, 10, lam)])[0]
        return {"A": r["train_total_cost"], "B": n * r["leaves"], "tree": r["tree"]}
    path = sl.parametric_path(solve, 0.0, 0.2)
    assert path["complete"]
    for lam in np.linspace(0, 0.2, 60):
        v = fl.solve_opt_exact(P, S, B, 3, [(5, 10, lam)])[0]["penalized_objective_total"]
        assert np.isclose(v, min(r["A"] + lam * r["B"] for _, _, r in path["segments"]), rtol=1e-9)
    leaves = [r["B"] / n for _, _, r in path["segments"]]
    assert leaves == sorted(leaves, reverse=True)                           # more penalty -> fewer leaves


# ---------------------------------------------------------------- penalty and depth-4 exactness (brute force)
@pytest.mark.parametrize("depth,seed,m,lam", [(2, 5, 1, 0.01), (2, 6, 20, 0.003), (3, 7, 10, 0.002), (3, 8, 1, 0.02)])
def test_leaf_penalty_is_exact(depth, seed, m, lam):
    _, _, P, S, B = _toy(n=250, B=4, seed=seed)
    n = S[:, 0].sum()
    for beta in (2, 5, 10):
        r = fl.solve_opt_exact(P, S, B, depth, [(beta, m, lam)])[0]
        assert np.isclose(r["penalized_objective_total"], _brute(P, S, B, beta, m, depth, lam * n))
        assert np.isclose(r["train_total_cost"], fl.tree_train_cost(r["tree"], P, S, beta))
        assert np.isclose(r["penalized_objective_total"], r["train_total_cost"] + lam * n * r["leaves"])


@pytest.mark.parametrize("seed,m", [(9, 1), (10, 12)])
def test_depth4_is_exact(seed, m):
    _, _, P, S, B = _toy(n=220, B=3, seed=seed)
    for beta in (2, 5, 10):
        r = fl.solve_opt_exact(P, S, B, 4, [(beta, m)])[0]
        assert r["status"].startswith("OPTIMAL")
        assert np.isclose(r["train_total_cost"], _brute(P, S, B, beta, m, 4))
        assert fl.tree_depth(r["tree"]) <= 4
        assert r["train_total_cost"] <= fl.solve_opt_exact(P, S, B, 3, [(beta, m)])[0]["train_total_cost"] + 1e-9


def test_depth4_time_limit_returns_valid_incumbent():
    _, _, P, S, B = _toy(n=300, B=4, seed=11)
    r = fl.solve_opt_exact(P, S, B, 4, [(5, 5)], time_limit=0.0)[0]
    assert r["status"].startswith("TIME_LIMIT") and np.isnan(r["gap"])
    d3 = fl.solve_opt_exact(P, S, B, 3, [(5, 5)])[0]["train_total_cost"]
    assert r["train_total_cost"] <= d3 + 1e-9                                # never worse than the depth-3 optimum


def test_zero_penalty_preserves_existing_behavior():
    _, _, P, S, B = _toy(n=400, B=4, seed=12)
    for depth in (1, 2, 3):
        a = fl.solve_opt_exact(P, S, B, depth, [(5, 10), (2, 30)])
        b = fl.solve_opt_exact(P, S, B, depth, [(5, 10, 0.0), (2, 30, 0.0)], leaf_penalty=0.0)
        for x, y in zip(a, b):
            assert x["train_total_cost"] == y["train_total_cost"]
            assert fl.tree_signature(x["tree"]) == fl.tree_signature(y["tree"])


needs_data = pytest.mark.skipif(not (PROC / "final" / "Xb_opt.npy").exists(), reason="final-stage data not built")


@needs_data
def test_saved_beta5_opt_tree_is_reproduced_exactly():
    import pandas as pd
    import final_setup as fs
    F, M = fs.load_strict()
    q = fs.q_of(F)
    Xb = np.load(fs.FINAL_DATA / "Xb_opt.npy")
    js = json.loads((fs.FINAL_DATA / "binner.json").read_text())
    nb = np.array([len(t) + 1 for t in js["thresholds"]])
    itr = np.flatnonzero(M["train"])
    P, S, B = fl.aggregate(Xb[itr], q[itr], nb)
    sel = pd.read_csv(fs.RES / "final_tree_selected.csv")
    r5 = sel[(sel.policy == "OPT-exact [full]") & (sel.beta == 5)].iloc[0]
    r = fl.solve_opt_exact(P, S, B, int(r5.max_depth), [(5, int(r5.min_leaf))])[0]
    binner = type("Bn", (), {"names": js["names"]})()
    saved = fl.tree_from_json(json.loads((fs.MODELS / "OPT-exact_full_b5.json").read_text()), binner)
    assert fl.tree_signature(r["tree"]) == fl.tree_signature(saved)
    assert np.isclose(r["train_mean_cost"], r5.train_cost, rtol=1e-12)


@needs_data
def test_chronology_and_leakage_safeguards():
    import final_setup as fs
    F, M = fs.load_strict()
    assert (F.loc[M["train"], "post_vmdeleted"] < fl.TRAIN_END).all()
    assert (F.loc[M["val"], "post_vmdeleted"] < fl.VAL_END).all()
    assert (F.loc[M["test"], "vmcreated"] >= fl.VAL_END).all()
    js = json.loads((fs.FINAL_DATA / "binner.json").read_text())
    assert fl.forbidden_features(js["names"]) == []
    assert fl.forbidden_features(fs.OPT_CANDIDATES) == []
