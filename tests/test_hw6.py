"""Tests for the Homework 6 refinements. Run:  python -m pytest -q tests/test_hw6.py"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import final_lib as fl  # noqa: E402
import hw6_lib as h6  # noqa: E402

PROC = ROOT / "data" / "processed"


def _forest(n=600, seed=0, bootstrap=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    q = np.clip(0.4 + 0.2 * X[:, 0] + 0.15 * rng.normal(size=n), 0, 1)
    rf = RandomForestRegressor(n_estimators=15, min_samples_leaf=10, max_samples=0.5 if bootstrap else None,
                               bootstrap=bootstrap, random_state=seed).fit(X, q)
    return rf, X, q


def test_forest_weights_nonnegative_and_sum_to_one():
    rf, X, _ = _forest()
    for x in X[:20]:
        w = h6.explicit_weights(rf, X, x)
        assert (w >= 0).all() and np.isclose(w.sum(), 1.0)


@pytest.mark.parametrize("bootstrap", [True, False])
def test_explicit_weighted_cost_equals_leaf_aggregation(bootstrap):
    rf, X, q = _forest(bootstrap=bootstrap)
    tabs = h6.leaf_tables(rf, X, q)
    rng = np.random.default_rng(1)
    Xq = rng.normal(size=(25, 3))
    agg = h6.weighted_shortfall(rf, tabs, Xq)
    for beta in (2, 5, 10):
        for j, x in enumerate(Xq):
            w = h6.explicit_weights(rf, X, x)
            explicit = w @ fl.cost_matrix(q, beta)                       # sum_i w_i C(a, Y_i)
            from_leaves = np.r_[fl.TIERS[:3] + beta * agg[j], 1.0]
            assert np.allclose(explicit, from_leaves)
            assert h6.prescribe(agg[j:j + 1], beta)[0] == np.argmin(explicit)


def test_prescribe_picks_minimum_cost_tier():
    S = np.array([[0.30, 0.10, 0.00], [0.00, 0.00, 0.00], [0.60, 0.40, 0.20]])
    # beta=5: costs [0.25+1.5, 0.5+0.5, 0.75, 1] -> 75%; [0.25, ...] -> 25%; [3.25, 2.5, 1.75, 1] -> 100%
    assert list(h6.prescribe(S, 5)) == [2, 0, 3]


def test_cluster_to_action_mapping():
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 4, 500)
    q = np.clip(0.2 * labels + rng.uniform(0, 0.3, 500), 0, 1)
    for beta in (2, 5, 10):
        act, tot = h6.cluster_actions(labels, q, beta, K=5)
        C = fl.cost_matrix(q, beta)
        for k in range(4):
            assert act[k] == np.argmin(C[labels == k].sum(0))
        assert act[4] == 3                                   # empty cluster -> conservative 100%


def test_sklearn_tree_conversion_matches_predict():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(800, 4))
    y = (X[:, 0] > 0).astype(int) + 2 * (X[:, 2] > 0.5)
    sk = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    tree = h6.sklearn_classifier_to_tree(sk)
    assert np.array_equal(h6.predict_threshold_tree(tree, X), sk.predict(X))


@pytest.mark.skipif(not fl.iai_available(), reason="IAI not available")
def test_iai_oct_returns_valid_cluster_ids(tmp_path):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(1500, 3))
    y = (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0)
    out = h6.run_iai_oct(X, y, [{"depth": 2, "minbucket": 20, "seed": 1, "cp": 1e-4}], tmp_path, threads=2)
    pred = h6.predict_threshold_tree(out[0]["tree"], X)
    assert set(np.unique(pred)) <= set(np.unique(y))
    assert (pred == y).mean() > 0.9


def test_cluster_features_are_creation_time_only():
    assert fl.forbidden_features(h6.CLUSTER_RAW) == []


needs_data = pytest.mark.skipif(not (PROC / "features.parquet").exists(), reason="processed data not built")


@needs_data
def test_chronological_split_integrity():
    import final_setup as fs
    F, M = fs.load_strict()
    assert (F.loc[M["train"], "vmcreated"] < fl.TRAIN_END).all()
    assert (F.loc[M["train"], "post_vmdeleted"] < fl.TRAIN_END).all()
    assert (F.loc[M["val"], "vmcreated"].between(fl.TRAIN_END, fl.VAL_END - 1)).all()
    assert (F.loc[M["val"], "post_vmdeleted"] < fl.VAL_END).all()
    assert (F.loc[M["test"], "vmcreated"] >= fl.VAL_END).all()
    assert M["test"].sum() == 426_846 and M["train"].sum() == 1_703_020 and M["val"].sum() == 275_143
    Xc, names = h6.cluster_matrix(F.iloc[:1000])
    assert np.isfinite(Xc).all() and len(names) == Xc.shape[1]


def test_reproducibility():
    a, X, q = _forest(seed=5)
    b, _, _ = _forest(seed=5)
    Xq = np.random.default_rng(6).normal(size=(50, 3))
    assert np.array_equal(h6.weighted_shortfall(a, h6.leaf_tables(a, X, q), Xq),
                          h6.weighted_shortfall(b, h6.leaf_tables(b, X, q), Xq))
    from sklearn.cluster import KMeans
    l1 = KMeans(4, n_init=3, random_state=0).fit_predict(X)
    l2 = KMeans(4, n_init=3, random_state=0).fit_predict(X)
    assert np.array_equal(l1, l2)
