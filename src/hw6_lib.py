"""Homework 6 refinements: forest-weighted prescriptions (Ch. 13) and interpretable clustering (Ch. 20).

Forest weights (Bertsimas & Kallus style):  w_i(x) = (1/T) sum_t 1{X_i in L_t(x)} / |L_t(x)|, where L_t(x) is the set of
TRAINING observations that fall in the leaf of tree t containing x. Here |L_t(x)| counts ALL strict training rows
routed to that leaf (not only the tree's bootstrap sample), so every training VM can receive weight from every tree.
Because the modeled cost is linear in the shortfall statistics, the weighted cost needs only per-leaf sums:
    C_hat(a | x) = a + beta * sum_i w_i(x) max(0, Y_i - a)
                 = a + beta * (1/T) sum_t [ sum_{i in L_t(x)} max(0, Y_i - a) / |L_t(x)| ].
So we store, for every tree and node, [count, S25, S50, S75] and never build an n_query x n_train weight matrix.
"""
import json
import os
import subprocess
from pathlib import Path

import numpy as np

import final_lib as fl


# ---------------------------------------------------------------- forest-weighted prescriptions
def leaf_tables(forest, X_train, q_train):
    """Per tree: (n_nodes, 4) array of [count, S25, S50, S75] over all training rows routed to each node's leaf."""
    st = fl.shortfall_stats(q_train)
    tables = []
    for est in forest.estimators_:
        leaf = est.apply(X_train)
        n_nodes = est.tree_.node_count
        tables.append(np.column_stack([np.bincount(leaf, weights=st[:, c], minlength=n_nodes) for c in range(4)]))
    return tables


def weighted_shortfall(forest, tables, X):
    """(n, 3) forest-weighted expected shortfall at the 25/50/75% tiers: sum_i w_i(x) max(0, Y_i - a)."""
    out = np.zeros((len(X), 3))
    for est, tab in zip(forest.estimators_, tables):
        rows = tab[est.apply(X)]
        if (rows[:, 0] <= 0).any():
            raise ValueError("a query fell in a leaf with no training rows")
        out += rows[:, 1:] / rows[:, :1]
    return out / len(forest.estimators_)


def prescribe(Shat, beta):
    """Action index minimizing a + beta * Shat_a (the 100% tier never falls short)."""
    exp_cost = np.column_stack([fl.TIERS[:3] + beta * Shat, np.ones(len(Shat))])
    return np.argmin(exp_cost, axis=1)


def explicit_weights(forest, X_train, x):
    """Dense weight vector w(x) over training rows for ONE query (used for tests and the illustrative example)."""
    w = np.zeros(len(X_train))
    for est in forest.estimators_:
        leaf_tr = est.apply(X_train)
        lx = est.apply(x.reshape(1, -1))[0]
        members = leaf_tr == lx
        w[members] += 1.0 / members.sum()
    return w / len(forest.estimators_)


# ---------------------------------------------------------------- clustering policies
def cluster_actions(labels, q, beta, K):
    """Cheapest action per cluster from TRAINING costs; clusters with no rows default to 100%."""
    C = fl.cost_matrix(q, beta)
    tot = np.column_stack([np.bincount(labels, weights=C[:, k], minlength=K) for k in range(4)])
    act = np.argmin(tot, axis=1)
    act[np.bincount(labels, minlength=K) == 0] = 3
    return act, tot


def predict_threshold_tree(tree, X):
    """Trees with raw thresholds (IAI convention: x < threshold goes to the lower/left child). Returns leaf labels."""
    out = np.empty(len(X), dtype=np.int64)

    def rec(node, idx):
        if node["leaf"]:
            out[idx] = node["label"]
            return
        go = X[idx, node["feature"]] < node["threshold"]
        rec(node["left"], idx[go])
        rec(node["right"], idx[~go])

    rec(tree, np.arange(len(X)))
    return out


def threshold_tree_depth(tree):
    return 0 if tree["leaf"] else 1 + max(threshold_tree_depth(tree["left"]), threshold_tree_depth(tree["right"]))


def threshold_tree_leaves(tree):
    return 1 if tree["leaf"] else threshold_tree_leaves(tree["left"]) + threshold_tree_leaves(tree["right"])


def sklearn_classifier_to_tree(sk):
    """sklearn sends x <= thr left; convert to the 'x < thr' convention with the next representable float."""
    t = sk.tree_

    def rec(i):
        if t.children_left[i] == -1:
            return {"leaf": True, "label": int(sk.classes_[np.argmax(t.value[i][0])])}
        return {"leaf": False, "feature": int(t.feature[i]), "threshold": float(np.nextafter(t.threshold[i], np.inf)),
                "left": rec(t.children_left[i]), "right": rec(t.children_right[i])}
    return rec(0)


def run_iai_oct(X, y, configs, workdir, threads=10):
    """Fit IAI OptimalTreeClassifier for each config dict(depth, minbucket, seed, cp); y = integer cluster ids."""
    import hashlib
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    Xf = X.astype(np.float64).T.copy()
    yf = y.astype(np.float64)
    cfg = "".join(f"{c['depth']} {c['minbucket']} {c['seed']} {c['cp']}\n" for c in configs)
    script = Path(__file__).with_name("iai_oct.jl")
    digest = hashlib.sha256(Xf.tobytes() + yf.tobytes() + cfg.encode() + script.read_bytes()).hexdigest()
    stamp = workdir / "done.sha256"
    if not (stamp.exists() and stamp.read_text() == digest):
        stamp.unlink(missing_ok=True)
        Xf.tofile(workdir / "X.bin")
        yf.tofile(workdir / "y.bin")
        (workdir / "dims.txt").write_text(f"{X.shape[0]} {X.shape[1]}\n")
        (workdir / "configs.txt").write_text(cfg)
        proc = subprocess.run([fl._default_julia(), f"--sysimage={fl._default_sysimage()}", "-t", str(threads),
                               str(script), str(workdir)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"IAI OCT failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
        stamp.write_text(digest)
    out = []
    for i, c in enumerate(configs):
        js = json.loads((workdir / f"oct_{i}.json").read_text())
        nodes = {n["id"]: n for n in js["nodes"]}

        def rec(k):
            n = nodes[k]
            if n["leaf"]:
                return {"leaf": True, "label": int(n["label"])}
            return {"leaf": False, "feature": n["feature"] - 1, "threshold": float(n["threshold"]),
                    "left": rec(n["lower"]), "right": rec(n["upper"])}
        out.append({**c, "tree": rec(1), "runtime_s": js["runtime_s"]})
    return out


# ---------------------------------------------------------------- clustering features (creation-time only)
CLUSTER_RAW = ["cores", "memory_gb", "sub_hist_mean", "sub_hist_median", "sub_hist_frac_le75", "sub_hist_frac_le50",
               "sub_n_hist", "dep_hist_mean", "hour"]


def cluster_matrix(F, with_time=True):
    """Transformed clustering features (before standardization). Skewed sizes/counts are log-transformed, shares are
    turned into 'share of past VMs above the threshold', and hour is encoded on the circle."""
    cols = {
        "log2 vCPU": np.log2(F.cores.to_numpy(float)),
        "log2 memory GB": np.log2(F.memory_gb.to_numpy(float)),
        "sub mean past P95": F.sub_hist_mean.to_numpy(float) / 100,
        "sub median past P95": F.sub_hist_median.to_numpy(float) / 100,
        "sub share past VMs >75%": 1 - F.sub_hist_frac_le75.to_numpy(float),
        "sub share past VMs >50%": 1 - F.sub_hist_frac_le50.to_numpy(float),
        "log1p # completed sub VMs": np.log1p(F.sub_n_hist.to_numpy(float)),
        "dep mean past P95": F.dep_hist_mean.to_numpy(float) / 100,
    }
    if with_time:
        h = F.hour.to_numpy(float)
        cols["hour sin"], cols["hour cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    return np.column_stack(list(cols.values())), list(cols)
