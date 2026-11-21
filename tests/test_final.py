"""Tests for the final-project pipeline. Run:  python -m pytest -q tests/

Synthetic tests always run; data tests are skipped when data/processed/*.parquet are absent; the MIO and IAI tests
are skipped when gurobipy / the IAI system image are unavailable.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import final_lib as fl  # noqa: E402

PROC = ROOT / "data" / "processed"


# ---------------------------------------------------------------- costs, units, oracle
def test_cost_matrix_hand_values():
    C = fl.cost_matrix(np.array([0.10, 0.60, 1.00]), beta=5)
    expected = np.array([[0.25, 0.50, 0.75, 1.00],
                         [0.25 + 5 * 0.35, 0.50 + 5 * 0.10, 0.75, 1.00],
                         [0.25 + 5 * 0.75, 0.50 + 5 * 0.50, 0.75 + 5 * 0.25, 1.00]])
    assert np.allclose(C, expected)


def test_sufficient_statistics_reproduce_costs():
    q = np.random.default_rng(0).uniform(0, 1, 500)
    for beta in (1.5, 2, 5, 10):
        assert np.allclose(fl.stats_to_costs(fl.shortfall_stats(q).sum(0), beta), fl.cost_matrix(q, beta).sum(0))


def test_oracle_is_argmin_and_large_beta_is_smallest_sufficient_tier():
    q = np.random.default_rng(1).uniform(0, 1, 2000)
    for beta in (2, 5, 10):
        a = fl.oracle_actions(q, beta)
        C = fl.cost_matrix(q, beta)
        assert np.allclose(C[np.arange(len(q)), np.searchsorted(fl.TIERS, a)], C.min(1))
    smallest_sufficient = fl.TIERS[np.searchsorted(fl.TIERS, q - 1e-12)]
    assert np.array_equal(fl.oracle_actions(q, 1e6), smallest_sufficient)


def test_units_percent_vs_fraction():
    assert np.allclose(fl.p95_to_fraction([0, 50, 100]), [0, 0.5, 1.0])
    with pytest.raises(ValueError):
        fl.p95_to_fraction([150.0])
    with pytest.raises(ValueError):
        fl.cost_matrix(np.array([50.0]), 5)          # percent passed where a fraction is required


def test_route_and_evaluate():
    assert np.allclose(fl.route(np.array([0, 25, 25.01, 60, 99, 140])), [0.25, 0.25, 0.5, 0.75, 1, 1])
    r = fl.evaluate(np.array([0.5, 0.5]), np.array([0.4, 0.7]), betas=(5,))
    assert np.isclose(r["cost b=5"], 0.5 + 5 * 0.1) and np.isclose(r["% under-provisioned"], 50)
    assert np.isclose(r["mean shortfall"], 0.1) and np.isclose(r["mean unused"], 0.05)


# ---------------------------------------------------------------- policy trees on synthetic data
def _toy(n=400, p=3, B=5, seed=0):
    rng = np.random.default_rng(seed)
    Xb = rng.integers(0, B, size=(n, p)).astype(np.int16)
    q = np.clip(0.15 * Xb[:, 0] + 0.1 * (Xb[:, 1] > 2) + rng.uniform(0, 0.4, n), 0, 1)
    P, S, BB = fl.aggregate(Xb, q, np.full(p, B))
    return Xb, q, P, S, BB


def _brute_force_depth2(P, S, B, beta, m):
    """Enumerate every tree of depth <= 2 (each node: leaf or any candidate split)."""
    p = P.shape[1]
    cands = [(j, t) for j in range(p) for t in range(B - 1)]

    def leaf_cost(mask):
        st = S[mask].sum(0)
        return fl.stats_to_costs(st, beta).min() if st[0] >= m else np.inf

    def best_d1(mask):
        best = leaf_cost(mask)
        for j, t in cands:
            left = mask & (P[:, j] <= t)
            best = min(best, leaf_cost(left) + leaf_cost(mask & ~left))
        return best

    allm = np.ones(len(P), bool)
    best = best_d1(allm)
    for j, t in cands:
        left = P[:, j] <= t
        best = min(best, best_d1(left) + best_d1(~left))
    return best


@pytest.mark.parametrize("seed,m", [(0, 1), (1, 20), (2, 60)])
def test_exact_dp_matches_brute_force_depth2(seed, m):
    _, _, P, S, B = _toy(seed=seed)
    for beta in (2, 5, 10):
        res = fl.solve_opt_exact(P, S, B, 2, [(beta, m)])[0]
        assert np.isclose(res["train_total_cost"], _brute_force_depth2(P, S, B, beta, m))
        assert np.isclose(fl.tree_train_cost(res["tree"], P, S, beta), res["train_total_cost"])


def _brute_force(P, S, B, beta, m, depth, mask=None):
    """Recursive enumeration of every tree of depth <= `depth` (definition of the optimum, no shortcuts)."""
    mask = np.ones(len(P), bool) if mask is None else mask
    st = S[mask].sum(0)
    best = fl.stats_to_costs(st, beta).min() if st[0] >= m else np.inf
    if depth == 0:
        return best
    for j in range(P.shape[1]):
        for t in range(B - 1):
            left = mask & (P[:, j] <= t)
            best = min(best, _brute_force(P, S, B, beta, m, depth - 1, left)
                       + _brute_force(P, S, B, beta, m, depth - 1, mask & ~left))
    return best


@pytest.mark.parametrize("seed,m", [(10, 1), (11, 15), (12, 40)])
def test_exact_dp_matches_brute_force_depth3(seed, m):
    _, _, P, S, B = _toy(n=300, p=2, B=4, seed=seed)
    for beta in (2, 5, 10):
        res = fl.solve_opt_exact(P, S, B, 3, [(beta, m)])[0]
        assert np.isclose(res["train_total_cost"], _brute_force(P, S, B, beta, m, 3))


def test_exact_dp_depth3_dominates_and_is_consistent():
    _, _, P, S, B = _toy(n=600, p=3, seed=3)
    for beta in (2, 5):
        d2 = fl.solve_opt_exact(P, S, B, 2, [(beta, 10)])[0]["train_total_cost"]
        r3 = fl.solve_opt_exact(P, S, B, 3, [(beta, 10)])[0]
        g3 = fl.fit_greedy_tree(P, S, B, beta, 3, 10)
        assert r3["train_total_cost"] <= d2 + 1e-9
        assert r3["train_total_cost"] <= g3["train_total_cost"] + 1e-9      # exact never worse than greedy
        assert np.isclose(fl.tree_train_cost(r3["tree"], P, S, beta), r3["train_total_cost"])
        assert fl.tree_depth(r3["tree"]) <= 3


def test_depth3_multi_config_equals_single_config():
    _, _, P, S, B = _toy(n=500, p=3, seed=4)
    cfgs = [(2, 5), (5, 30), (10, 80)]
    multi = fl.solve_opt_exact(P, S, B, 3, cfgs)
    for cfg, r in zip(cfgs, multi):
        assert np.isclose(fl.solve_opt_exact(P, S, B, 3, [cfg])[0]["train_total_cost"], r["train_total_cost"])


def test_greedy_leaf_actions_minimize_leaf_cost_and_respect_min_leaf():
    Xb, q, P, S, B = _toy(n=800, seed=5)
    for beta in (2, 5, 10):
        tree = fl.fit_greedy_tree(P, S, B, beta, 3, 40)["tree"]
        act = fl.predict_tree(tree, Xb)
        C = fl.cost_matrix(q, beta)
        paths = _leaf_ids(tree, Xb)
        for lid in np.unique(paths):
            rows = paths == lid
            assert rows.sum() >= 40
            assert np.isclose(C[rows].sum(0)[act[rows][0]], C[rows].sum(0).min())
        assert set(np.unique(fl.TIERS[act])) <= set(fl.TIERS)


def _leaf_ids(tree, Xb):
    out = np.zeros(len(Xb), dtype=np.int64)

    def rec(node, idx, code):
        if node["leaf"]:
            out[idx] = code
            return
        left = Xb[idx, node["feature"]] <= node["t"]
        rec(node["left"], idx[left], 2 * code + 1)
        rec(node["right"], idx[~left], 2 * code + 2)
    rec(tree, np.arange(len(Xb)), 0)
    return out


def test_tree_predictions_are_valid_actions_and_json_roundtrip():
    Xb, q, P, S, B = _toy(seed=6)
    binner = fl.Binner(max_bins=5).fit(Xb.astype(float), ["f0", "f1", "f2"])
    tree = fl.solve_opt_exact(P, S, B, 2, [(5, 5)])[0]["tree"]
    pred = fl.predict_tree(tree, Xb)
    assert set(np.unique(pred)) <= {0, 1, 2, 3}
    back = fl.tree_from_json(fl.tree_to_json(tree, binner), binner)
    assert np.array_equal(fl.predict_tree(back, Xb), pred)


def test_binner_semantics():
    x = np.array([[1.0], [2.0], [2.0], [3.0], [10.0]])
    b = fl.Binner(max_bins=32).fit(x, ["x"])
    assert np.array_equal(b.transform(x)[:, 0], [0, 1, 1, 2, 3])
    for t in range(len(b.thresholds[0])):     # split 'bin <= t' <=> 'x <= threshold_value'
        assert np.array_equal(b.transform(x)[:, 0] <= t, x[:, 0] <= b.threshold_value(0, t))


def test_cluster_bootstrap_detects_sign():
    rng = np.random.default_rng(0)
    cl = np.repeat(np.arange(100), 20)
    a = rng.normal(1.0, 0.1, 2000)
    r = fl.cluster_bootstrap_diff(a, a + 0.05, cl, n_boot=300)
    assert r["ci high"] < 0 and np.isclose(r["mean diff"], -0.05)


try:
    import gurobipy  # noqa: F401
    HAVE_GUROBI = True
except ImportError:
    HAVE_GUROBI = False


@pytest.mark.skipif(not HAVE_GUROBI, reason="gurobipy missing")
def test_mio_matches_exact_dp_on_small_problem():
    _, _, P, S, B = _toy(n=150, p=2, B=4, seed=7)
    for depth in (1, 2):
        dp = fl.solve_opt_exact(P, S, B, depth, [(5, 10)])[0]
        mio = fl.solve_opt_mio(P, S, np.full(2, B), 5, depth, 10, time_limit=120, threads=4)
        assert mio["status"] == "OPTIMAL"
        assert np.isclose(mio["train_total_cost"], dp["train_total_cost"], rtol=1e-6)
        assert np.isclose(fl.tree_train_cost(mio["tree"], P, S, 5), mio["train_total_cost"], rtol=1e-6)


@pytest.mark.skipif(not fl.iai_available(), reason="IAI system image / license not available")
def test_iai_bridge_returns_valid_tree(tmp_path):
    Xb, q, P, S, B = _toy(n=600, seed=8)
    out = fl.run_iai_policy_trees(Xb, fl.cost_matrix(q, 5), [{"depth": 2, "minbucket": 20, "seed": 1}], tmp_path, 2)
    tree = out[0]["tree"]
    assert fl.tree_depth(tree) <= 2
    exact = fl.solve_opt_exact(P, S, B, 2, [(5, 20)])[0]["train_total_cost"]
    assert fl.tree_train_cost(tree, P, S, 5) >= exact - 1e-6        # local search cannot beat the exact optimum


# ---------------------------------------------------------------- leakage and chronology (real data)
needs_data = pytest.mark.skipif(not (PROC / "features.parquet").exists(), reason="processed data not built")


def test_no_forbidden_features_in_any_feature_set():
    import common
    import final_setup as fs
    for cols in (common.FEATS_A, common.FEATS_B, common.FEATS_C, fs.OPT_CANDIDATES):
        assert fl.forbidden_features(cols) == [], cols


@needs_data
def test_strict_masks_respect_label_availability_and_chronology():
    import final_setup as fs
    F, masks = fs.load_strict()
    tr, va, te = masks["train"], masks["val"], masks["test"]
    assert (F.loc[tr, "vmcreated"] < fl.TRAIN_END).all() and (F.loc[tr, "post_vmdeleted"] < fl.TRAIN_END).all()
    assert (F.loc[va, "vmcreated"] >= fl.TRAIN_END).all() and (F.loc[va, "post_vmdeleted"] < fl.VAL_END).all()
    assert (F.loc[te, "vmcreated"] >= fl.VAL_END).all()
    assert not (tr & va).any() and not (va & te).any() and not (tr & te).any()
    assert F.y_p95.max() > 1.0                         # stored in percent; converted with p95_to_fraction
    assert 0 <= fs.q_of(F).min() and fs.q_of(F).max() <= 1


@needs_data
def test_history_features_use_only_completed_prior_vms():
    import pandas as pd
    F = pd.read_parquet(PROC / "features.parquet", columns=["key_sub", "vmcreated", "sub_n_hist", "sub_hist_mean",
                                                             "sub_n_prior_created"])
    vm = pd.read_parquet(PROC / "vmtable.parquet", columns=["subscriptionid", "vmcreated", "vmdeleted", "p95maxcpu"])
    vm["sub"] = vm.subscriptionid.cat.codes
    rng = np.random.default_rng(0)
    rows = rng.choice(np.flatnonzero((F.sub_n_hist > 0).to_numpy() & (F.sub_n_hist < 2000).to_numpy()), 40, replace=False)
    groups = dict(tuple(vm.groupby("sub")))
    for i in rows:
        r = F.iloc[i]
        g = groups[r.key_sub]
        done = g[g.vmdeleted < r.vmcreated]                     # completed strictly before creation
        assert len(done) == r.sub_n_hist
        assert np.isclose(done.p95maxcpu.mean(), r.sub_hist_mean, rtol=1e-4)
        assert (g.vmcreated < r.vmcreated).sum() == r.sub_n_prior_created


def test_sampling_is_reproducible():
    import final_setup as fs
    a = fs.sample_indices(np.arange(10_000), 500, seed=3)
    b = fs.sample_indices(np.arange(10_000), 500, seed=3)
    c = fs.sample_indices(np.arange(10_000), 500, seed=4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
