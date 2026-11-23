"""Unit tests for the stage-12 metric helpers (src/evaluation_metrics_lib.py)."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluation_metrics_lib as em  # noqa: E402
import final_lib as fl  # noqa: E402

RNG = np.random.default_rng(0)


def random_case(n=500, seed=1):
    r = np.random.default_rng(seed)
    q = np.clip(r.beta(0.8, 1.2, n), 0, 1)
    a = fl.TIERS[r.integers(0, 4, n)]
    return a, q


# ---------------------------------------------------------------- operational metrics
def test_zero_exceedances():
    q = np.array([0.0, 0.3, 0.99, 1.0])
    m = em.policy_metrics(np.ones(4), q, beta=5)
    assert m["P(Y>r)"] == 0 and m["E[sf]"] == 0 and m["max sf"] == 0
    assert np.isnan(m["E[sf | Y>r]"]) and np.isnan(m["sf P95 | Y>r"]) and np.isnan(m["sf P50 | Y>r"])
    assert m["sf P99"] == 0 and all(m[f"P(sf>{t}pp)"] == 0 for t in (5, 10, 20))
    assert m["cost"] == pytest.approx(1.0) and m["shortfall penalty"] == 0 and m["share 100%"] == 1


def test_extreme_shortfall_and_thresholds():
    q = np.array([1.0, 0.0, 0.30])
    a = np.full(3, 0.25)                      # shortfalls 0.75, 0, 0.05 (the 5 pp one sits exactly on the threshold)
    m = em.policy_metrics(a, q, beta=5)
    assert m["max sf"] == pytest.approx(0.75)
    assert m["P(Y>r)"] == pytest.approx(2 / 3)
    assert m["E[sf]"] == pytest.approx(0.8 / 3)
    assert m["E[sf | Y>r]"] == pytest.approx(0.4)
    assert m["P(sf>5pp)"] == pytest.approx(1 / 3)          # strict '>' : exactly 5 pp does not count
    assert m["P(sf>20pp)"] == pytest.approx(1 / 3)
    assert m["cost"] == pytest.approx(0.25 + 5 * 0.8 / 3)
    assert m["sf P99 | Y>r"] == pytest.approx(0.75)


def test_metrics_agree_with_final_lib_evaluate():
    a, q = random_case()
    for b in (2, 5, 10):
        m, ev = em.policy_metrics(a, q, b), fl.evaluate(a, q, betas=(b,))
        assert m["cost"] == pytest.approx(ev[f"cost b={b}"])
        assert m["reservation cost"] + m["shortfall penalty"] == pytest.approx(m["cost"])
        assert m["E[r]"] == pytest.approx(ev["avg reserved"])
        assert 100 * m["P(Y>r)"] == pytest.approx(ev["% under-provisioned"])
        assert m["E[sf]"] == pytest.approx(ev["mean shortfall"])
        assert m["E[unused]"] == pytest.approx(ev["mean unused"])
        assert sum(m[f"share {t}%"] for t in (25, 50, 75, 100)) == pytest.approx(1)


def test_integer_weights_equal_row_duplication():
    a, q = random_case(200, seed=2)
    w = np.random.default_rng(3).integers(1, 5, len(q))
    mw = em.policy_metrics(a, q, 5, w=w)
    md = em.policy_metrics(np.repeat(a, w), np.repeat(q, w), 5)
    for k, v in md.items():
        assert (np.isnan(v) and np.isnan(mw[k])) or mw[k] == pytest.approx(v), k


def test_wquantile_matches_numpy_inverted_cdf():
    x = RNG.normal(size=257)
    for p in (0.0, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0):
        assert em.wquantile(x, p)[0] == np.quantile(x, p, method="inverted_cdf")
    assert np.isnan(em.wquantile(np.array([]), 0.5)[0])


def test_equal_subscription_weighting_is_mean_of_group_means():
    g = np.array(["a", "b", "b", "b", "c", "c"])
    x = np.array([1.0, 0.0, 0.0, 3.0, 2.0, 4.0])
    w = em.equal_group_weights(g)
    assert em.wmean(x, w) == pytest.approx(np.mean([1.0, 1.0, 3.0]))
    assert em.wmean(x) == pytest.approx(x.mean())          # VM weighting differs
    _, s = em.group_sums(g, x=x)
    assert list(s["n"]) == [1, 3, 2] and list(s["x"]) == [1, 3, 6]


# ---------------------------------------------------------------- oracle regret
def test_oracle_regret_definitions():
    r = em.oracle_regret(0.6, 1.0, 0.6)
    assert r["regret"] == 0 and r["% of attainable saving captured"] == pytest.approx(100)
    r = em.oracle_regret(1.0, 1.0, 0.6)
    assert r["% of attainable saving captured"] == 0 and r["% saving vs fixed"] == 0
    r = em.oracle_regret(0.8, 1.0, 0.6)
    assert r["regret"] == pytest.approx(0.2) and r["% of attainable saving captured"] == pytest.approx(50)
    assert r["regret % of oracle cost"] == pytest.approx(100 * 0.2 / 0.6)
    assert np.isnan(em.oracle_regret(0.9, 0.7, 0.7)["% of attainable saving captured"])


def test_oracle_regret_nonnegative():
    _, q = random_case(2000, seed=4)
    for b in (2, 5, 10):
        co = np.mean(fl.cost_matrix(q, b).min(1))
        assert co == pytest.approx(np.mean(fl.TIERS[np.argmin(fl.cost_matrix(q, b), 1)] + b * np.maximum(0, q - fl.oracle_actions(q, b))))
        for s in range(5):
            a, _ = random_case(2000, seed=10 + s)
            assert em.oracle_regret(np.mean(a + b * np.maximum(0, q - a)), 1.0, co)["regret"] >= -1e-12


# ---------------------------------------------------------------- reserving less vs targeting
def test_untargeted_shortfall_is_permutation_average():
    a = np.array([0.25, 0.5, 0.5, 0.75, 1.0])
    q = np.array([0.9, 0.1, 0.6, 0.3, 0.8])
    perm = np.mean([np.maximum(0, q - a[list(p)]).mean() for p in itertools.permutations(range(5))])
    assert em.untargeted_shortfall(a, q) == pytest.approx(perm)


def test_fixed_tier_mixtures():
    _, q = random_case(1000, seed=5)
    sf = em.tier_shortfalls(q)
    assert sf[-1] == 0 and np.all(np.diff(sf) <= 0)
    assert em.fixed_mixture_reservation(q, 0.0) == pytest.approx(1.0)
    assert em.fixed_mixture_reservation(q, sf[2]) == pytest.approx(0.75)
    assert em.fixed_mixture_reservation(q, sf[2] / 2) == pytest.approx(0.875)          # 50/50 mix of 75% and 100%
    assert em.fixed_mixture_shortfall(q, 0.875) == pytest.approx(sf[2] / 2)
    assert em.fixed_mixture_reservation(q, 10.0) == pytest.approx(0.25)


def test_lower_hull_and_mixture_frontier():
    x = np.array([0.0, 1.0, 2.0, 3.0, 1.0])
    y = np.array([10.0, 9.0, 5.0, 4.0, 12.0])
    h = em.lower_hull(x, y)
    assert [tuple(p) for p in h] == [(0.0, 10.0), (2.0, 5.0), (3.0, 4.0)]     # (1, 9) is above the chord
    assert em.mixture_frontier(x, y, 1.0) == pytest.approx(7.5)
    assert em.mixture_frontier(x, y, 5.0) == pytest.approx(4.0)
    assert np.isnan(em.mixture_frontier(x + 1, y, 0.5))


def test_pareto_dominance():
    P = np.array([[1, 1], [2, 2], [1, 2], [0, 3], [1, 1]])
    assert list(em.pareto_dominated(P)) == [False, True, True, False, False]   # ties do not dominate


# ---------------------------------------------------------------- disagreement and bootstrap
def test_disagreement_accounting():
    a1, q = random_case(400, seed=6)
    d = em.disagreement(a1, a1.copy(), q, 5)
    assert d["% VMs disagree"] == 0 and d["net cost diff per VM (1-2)"] == 0
    a2 = a1.copy()
    a2[:40] = 1.0
    d = em.disagreement(a1, a2, q, 5, groups=np.arange(400) % 7)
    c = lambda a: a + 5 * np.maximum(0, q - a)
    assert d["net cost diff per VM (1-2)"] == pytest.approx(np.mean(c(a1) - c(a2)))
    D = ~np.isclose(a1, a2)
    assert d["mean cost diff in D"] * D.sum() / len(q) == pytest.approx(d["net cost diff per VM (1-2)"])
    assert 0 < d["share of gross |diff| in top 10% of D"] <= 1


def test_top_share():
    assert em.top_share([5, -3, 1, 1], k=1) == pytest.approx(0.5)
    assert em.top_share([1, 1, 1, 1], frac=0.5) == pytest.approx(0.5)
    assert np.isnan(em.top_share([0, 0]))


def test_multi_metric_bootstrap_matches_final_lib():
    n = 3000
    g = RNG.integers(0, 150, n)
    x, y = RNG.normal(size=n), RNG.normal(size=n)
    ref = fl.cluster_bootstrap_diff(x, y, g, n_boot=300, seed=0)
    out = em.cluster_bootstrap_multi({"d": x - y, "other": x}, g, n_boot=300, seed=0)
    for k in ("mean diff", "ci low", "ci high", "share reps < 0"):
        assert out["d"][k] == pytest.approx(ref[k], abs=1e-12)
    neg = em.cluster_bootstrap_multi({"d": y - x}, g, n_boot=300, seed=0)["d"]
    assert neg["ci low"] == pytest.approx(-ref["ci high"], abs=1e-12)
