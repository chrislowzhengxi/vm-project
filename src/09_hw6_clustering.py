"""Stage 9 (HW6 refinement 1, Ch. 20): interpretable clustering -> cluster-specific reservations.

Step A  k-means on standardized creation-time features (scaler and k-means fit on strict TRAINING rows only),
        K in {3, 5, 8, 10}, with and without creation-hour features; inertia + silhouette (20k-row sample).
Step B  each cluster gets the action with the lowest TRAINING modeled cost (per beta); K / variant chosen on the
        strict validation set at beta = 5 (clusters are beta-independent, the action map is per beta).
Step C  Optimal Classification Trees (IAI OptimalTreeClassifier, local search, no certificate) predict the k-means
        cluster from the same creation-time features (raw, interpretable units) on a reproducible 200k-row training
        sample; depth 2-4 chosen on validation allocation cost. A new VM follows the OCT to a predicted cluster and
        receives that cluster's (fixed, training-derived) reservation. A greedy CART classifier of equal depth and
        sample is reported as a reference.
Outputs: results/hw6/hw6_cluster_*.{csv,md}, figures/hw6/hw6_fig_cluster_*.png, data/processed/hw6/cluster_actions_*.parquet
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import final_lib as fl
import final_setup as fs
import hw6_lib as h6
from common import C, FIG, PROC, RES, plt, to_md

OUT, FOUT, DOUT = RES / "hw6", FIG / "hw6", PROC / "hw6"
for d in (OUT, FOUT, DOUT):
    d.mkdir(parents=True, exist_ok=True)
t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)


F, M = fs.load_strict()
q = fs.q_of(F)
itr, iva, ite = (np.flatnonzero(M[k]) for k in ("train", "val", "test"))
qtr, qv, qt = q[itr], q[iva], q[ite]
PA_t = pd.read_parquet(fs.FINAL_DATA / "policy_actions_test.parquet")
KS = (3, 5, 8, 10)


def cost(a, qq, b, w=None):
    return float(np.average(a + b * np.maximum(0, qq - a), weights=w))


# ---------------------------------------------------------------- feature distributions before / after transforms
raw = F.iloc[itr][h6.CLUSTER_RAW]
Xall, names = h6.cluster_matrix(F, with_time=True)
after = pd.DataFrame(Xall[itr][:, :8], columns=names[:8]).skew().to_numpy()   # first 8 columns map 1:1 to raw features
skew = pd.DataFrame({"raw feature skewness (train)": raw.skew(),
                     "after transform (train)": list(after) + [np.nan]})   # hour is encoded as sin/cos
skew["transform"] = ["log2", "log2", "/100", "/100", "1 - share", "1 - share", "log1p", "/100", "sin/cos of 2*pi*h/24"]
to_md(skew, OUT / "hw6_cluster_feature_skew", floatfmt=".2f")

# ---------------------------------------------------------------- Step A + B: k-means policies
grid, models = [], {}
rng = np.random.default_rng(fs.SEED)
sil_idx = rng.choice(len(itr), 20_000, replace=False)
for variant in ("with time", "no time"):
    Xc, cn = h6.cluster_matrix(F, with_time=(variant == "with time"))
    scaler = StandardScaler().fit(Xc[itr])
    Z = scaler.transform(Xc)
    for K in KS:
        t0 = time.perf_counter()
        km = KMeans(n_clusters=K, n_init=5, random_state=fs.SEED).fit(Z[itr])
        t_fit = time.perf_counter() - t0
        ltr, lva, lte = km.labels_, km.predict(Z[iva]), km.predict(Z[ite])
        sil = silhouette_score(Z[itr][sil_idx], ltr[sil_idx], random_state=fs.SEED)
        row = {"variant": variant, "K": K, "inertia per VM (train)": km.inertia_ / len(itr), "silhouette (20k)": sil,
               "fit_s": t_fit, "smallest cluster share %": 100 * np.bincount(ltr).min() / len(itr)}
        for b in fl.BETAS:
            act, _ = h6.cluster_actions(ltr, qtr, b, K)
            row[f"val cost b={b}"] = cost(fl.TIERS[act[lva]], qv, b)
            row[f"test cost b={b}"] = cost(fl.TIERS[act[lte]], qt, b)
            row[f"distinct actions b={b}"] = len(set(act))
        grid.append(row)
        models[(variant, K)] = (km, scaler, cn, ltr, lva, lte)
        log(f"k-means {variant} K={K}: {t_fit:.0f}s, silhouette {sil:.3f}, val cost b=5 {row['val cost b=5']:.4f}")
KG = pd.DataFrame(grid)
to_md(KG.set_index(["variant", "K"]), OUT / "hw6_cluster_kmeans_grid", floatfmt=".4f")
best = KG.loc[KG["val cost b=5"].idxmin()]
VAR, K = best.variant, int(best.K)
km, scaler, cn, ltr, lva, lte = models[(VAR, K)]
log(f"selected on validation (beta=5): {VAR}, K={K}")
ACT = {b: h6.cluster_actions(ltr, qtr, b, K)[0] for b in fl.BETAS}

# cluster profiles (training rows; realized P95 shown for interpretation only)
Ftr = F.iloc[itr]
prof = pd.DataFrame({
    "share of train VMs %": 100 * np.bincount(ltr, minlength=K) / len(itr),
    "share of test VMs %": 100 * np.bincount(lte, minlength=K) / len(ite),
    "median vCPU": pd.Series(Ftr.cores.to_numpy()).groupby(ltr).median(),
    "median memory GB": pd.Series(Ftr.memory_gb.to_numpy()).groupby(ltr).median(),
    "mean sub mean past P95 %": pd.Series(Ftr.sub_hist_mean.to_numpy()).groupby(ltr).mean(),
    "mean sub share past VMs >75% %": 100 * (1 - pd.Series(Ftr.sub_hist_frac_le75.to_numpy()).groupby(ltr).mean()),
    "median # completed sub VMs": pd.Series(Ftr.sub_n_hist.to_numpy()).groupby(ltr).median(),
    "mean dep mean past P95 %": pd.Series(Ftr.dep_hist_mean.to_numpy()).groupby(ltr).mean(),
    "mean creation hour": pd.Series(Ftr.hour.to_numpy()).groupby(ltr).mean(),
    "realized mean P95 % (train, not a feature)": 100 * pd.Series(qtr).groupby(ltr).mean(),
} | {f"action b={b}": [f"{int(fl.TIERS[a] * 100)}%" for a in ACT[b]] for b in fl.BETAS})
prof.index.name = "cluster"
order = prof["realized mean P95 % (train, not a feature)"].sort_values().index.to_numpy()
to_md(prof.loc[order], OUT / "hw6_cluster_profiles", floatfmt=".1f")

# ---------------------------------------------------------------- Step C: OCT (and greedy CART) on cluster labels
oct_cols = [c for c in h6.CLUSTER_RAW if VAR == "with time" or c != "hour"]
Xo = F[oct_cols].to_numpy(float)
samp = fs.sample_indices(itr, 200_000, fs.SEED)
lab_samp = km.predict(scaler.transform(h6.cluster_matrix(F.iloc[samp], with_time=(VAR == "with time"))[0]))
assert np.array_equal(lab_samp, ltr[np.searchsorted(itr, samp)])
mb = int(0.005 * len(samp))
trees, trow = {}, []
if fl.iai_available():
    cfgs = [{"depth": d, "minbucket": mb, "seed": 1, "cp": 1e-6} for d in (2, 3, 4)]
    for r in h6.run_iai_oct(Xo[samp], lab_samp, cfgs, DOUT / "iai_oct", threads=10):
        trees[("OCT (IAI)", r["depth"])] = (r["tree"], r["runtime_s"])
    log("OCT fits done")
else:
    log("IAI not available: OCT skipped (CART classifier reference only)")
for d in (2, 3, 4):
    t0 = time.perf_counter()
    sk = DecisionTreeClassifier(max_depth=d, min_samples_leaf=mb, random_state=fs.SEED).fit(Xo[samp], lab_samp)
    trees[("CART classifier", d)] = (h6.sklearn_classifier_to_tree(sk), time.perf_counter() - t0)
for (meth, d), (tree, rt) in trees.items():
    pv, pt_ = h6.predict_threshold_tree(tree, Xo[iva]), h6.predict_threshold_tree(tree, Xo[ite])
    row = {"method": meth, "max depth": d, "depth": h6.threshold_tree_depth(tree), "leaves": h6.threshold_tree_leaves(tree),
           "train-sample agreement %": 100 * np.mean(h6.predict_threshold_tree(tree, Xo[samp]) == lab_samp),
           "val agreement %": 100 * np.mean(pv == lva), "test agreement %": 100 * np.mean(pt_ == lte), "fit_s": rt}
    for b in fl.BETAS:
        row[f"val cost b={b}"] = cost(fl.TIERS[ACT[b][pv]], qv, b)
        row[f"test cost b={b}"] = cost(fl.TIERS[ACT[b][pt_]], qt, b)
    trow.append(row)
TG = pd.DataFrame(trow)


def leaf_labels(tree):
    return [tree["label"]] if tree["leaf"] else leaf_labels(tree["left"]) + leaf_labels(tree["right"])


TG["clusters reachable"] = [len(set(leaf_labels(trees[(m, d)][0]))) for m, d in zip(TG.method, TG["max depth"])]
to_md(TG.set_index(["method", "max depth"]), OUT / "hw6_cluster_oct_grid", floatfmt=".4f")
sel_meth = "OCT (IAI)" if ("OCT (IAI)", 2) in trees else "CART classifier"
gsel = TG[TG.method == sel_meth]
d_sel = int(gsel.loc[gsel["val cost b=5"].idxmin(), "max depth"])
d_acc = int(gsel.loc[gsel["val agreement %"].idxmax(), "max depth"])
gc = TG[TG.method == "CART classifier"]
d_cart = int(gc.loc[gc["val cost b=5"].idxmin(), "max depth"])
oct_tree = trees[(sel_meth, d_sel)][0]
(OUT / "hw6_oct_tree.json").write_text(json.dumps({"features": oct_cols, "cluster_actions": {str(b): [float(fl.TIERS[a])
                                                   for a in ACT[b]] for b in fl.BETAS}, "tree": oct_tree}, indent=1))
log(f"selected {sel_meth} depth {d_sel} (by val cost); depth by val agreement = {d_acc}")

# ---------------------------------------------------------------- consolidated comparison
POL = {}
for b in fl.BETAS:
    POL[("k-means clusters -> action", b)] = fl.TIERS[ACT[b][lte]]
    POL[(f"{sel_meth} depth {d_sel} -> cluster -> action", b)] = fl.TIERS[ACT[b][h6.predict_threshold_tree(oct_tree, Xo[ite])]]
    POL[(f"CART classifier depth {d_cart} -> cluster -> action", b)] = fl.TIERS[ACT[b][
        h6.predict_threshold_tree(trees[("CART classifier", d_cart)][0], Xo[ite])]]
    for name, col in {"Best fixed (val)": "Best fixed (val)", "Greedy policy tree": "Greedy [full]",
                      "OPT-exact": "OPT-exact [full]", "Cost-aware RF": "Cost-aware RF",
                      "Oracle (hindsight)": "Oracle (hindsight)"}.items():
        POL[(name + " (existing)", b)] = PA_t[f"{col}|b={b}"].to_numpy(float)
rows = []
for b in fl.BETAS:
    bf, orc = cost(POL[("Best fixed (val) (existing)", b)], qt, b), cost(POL[("Oracle (hindsight) (existing)", b)], qt, b)
    for (name, bb), a in POL.items():
        if bb == b:
            ev = fl.evaluate(a, qt, betas=(b,))
            c_ = ev.pop(f"cost b={b}")
            rows.append({"beta": b, "policy": name, "test cost": c_, "% reduction vs best fixed": 100 * (bf - c_) / bf,
                         "% of gap closed": 100 * (bf - c_) / (bf - orc)} | ev)
T = pd.DataFrame(rows)
T.to_csv(OUT / "hw6_cluster_results.csv", index=False)
for b in fl.BETAS:
    to_md(T[T.beta == b].drop(columns="beta").set_index("policy"), OUT / f"hw6_cluster_results_b{b}", floatfmt=".4f")
pd.DataFrame({f"{n}|b={b}": a.astype(np.float32) for (n, b), a in POL.items() if "existing" not in n},
             index=ite).to_parquet(DOUT / "cluster_actions_test.parquet")

subs = F.key_sub.to_numpy()[ite]
KMN, OCTN = "k-means clusters -> action", f"{sel_meth} depth {d_sel} -> cluster -> action"
bt = []
for b in fl.BETAS:
    for a_, b_ in [(KMN, "Best fixed (val) (existing)"), (KMN, "OPT-exact (existing)"), (OCTN, "Best fixed (val) (existing)"),
                   (OCTN, "OPT-exact (existing)"), (OCTN, KMN), (OCTN, "Cost-aware RF (existing)")]:
        ca = POL[(a_, b)] + b * np.maximum(0, qt - POL[(a_, b)])
        cb = POL[(b_, b)] + b * np.maximum(0, qt - POL[(b_, b)])
        r = fl.cluster_bootstrap_diff(ca, cb, subs, n_boot=1000, seed=fs.SEED)
        bt.append({"beta": b, "A": a_, "B": b_, "cost A": ca.mean(), "cost B": cb.mean(), "A - B": r["mean diff"],
                   "95% CI low": r["ci low"], "95% CI high": r["ci high"], "% of B": 100 * r["mean diff"] / cb.mean()})
BT = pd.DataFrame(bt)
to_md(BT.set_index(["beta", "A", "B"]), OUT / "hw6_cluster_bootstrap", floatfmt=".4f")
WT, WV = fs.eval_weights(F, M["test"])["vCPU-hour-weighted"], fs.eval_weights(F, M["val"])["vCPU-hour-weighted"]
b = fl.MAIN_BETA
bfw = min(fl.TIERS, key=lambda t: cost(np.full(len(iva), t), qv, b, WV))
cf = cost(np.full(len(ite), bfw), qt, b, WT)
to_md(pd.DataFrame([{"policy": n, "vCPU-hour-weighted cost": cost(a, qt, b, WT),
                     "% reduction vs weighted best fixed": 100 * (cf - cost(a, qt, b, WT)) / cf}
                    for (n, bb), a in POL.items() if bb == b and "Oracle" not in n]).set_index("policy"),
      OUT / "hw6_cluster_vcpu_hour_b5", floatfmt=".4f")
log("tables + bootstrap done")

# ---------------------------------------------------------------- figures: cluster profiles + OCT diagram
Zc = scaler.transform(h6.cluster_matrix(F.iloc[itr], with_time=(VAR == "with time"))[0])
cent = pd.DataFrame(Zc, columns=cn).groupby(ltr).mean().loc[order]
fig, ax = plt.subplots(figsize=(10.5, 0.55 * K + 1.6))
im = ax.imshow(cent.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(cn)), cn, rotation=30, ha="right", fontsize=8)
ax.set_yticks(range(K), [f"cluster {c}: {prof.loc[c, 'share of train VMs %']:.0f}% of VMs, reserve "
                         f"{prof.loc[c, 'action b=5']} (realized P95 {prof.loc[c, 'realized mean P95 % (train, not a feature)']:.0f}%)"
                         for c in order], fontsize=8)
for i in range(K):
    for j in range(len(cn)):
        ax.text(j, i, f"{cent.iloc[i, j]:+.1f}", ha="center", va="center", fontsize=7,
                color="white" if abs(cent.iloc[i, j]) > 1.2 else C["ink"])
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.025, label="cluster mean (standardized)")
ax.set_title(f"k-means clusters (K = {K}, {VAR}; fit on strict training VMs) and their beta = 5 reservations", fontsize=10)
fig.savefig(FOUT / "hw6_fig_cluster_profiles.png")
plt.close(fig)

TIER_COL = {0: "#86b6ef", 1: "#3987e5", 2: "#1c5cab", 3: "#0d366b"}


def draw_oct(tree, depth_label, path, note=""):
    """Readable OCT diagram: splits in original units, leaves show cluster, beta = 5 reservation and test share."""
    def layout(node, depth, pos):
        if node["leaf"]:
            node["_x"] = pos[0]
            pos[0] += 1
        else:
            layout(node["left"], depth + 1, pos)
            layout(node["right"], depth + 1, pos)
            node["_x"] = (node["left"]["_x"] + node["right"]["_x"]) / 2
        node["_d"] = depth

    def mark(node, idx):
        node["_n"] = len(idx)
        if not node["leaf"]:
            go = Xo[ite][idx, node["feature"]] < node["threshold"]
            mark(node["left"], idx[go])
            mark(node["right"], idx[~go])

    layout(tree, 0, [0])
    mark(tree, np.arange(len(ite)))
    nl, dep = h6.threshold_tree_leaves(tree), h6.threshold_tree_depth(tree)
    fig, ax = plt.subplots(figsize=(max(9, 1.9 * nl), 1.6 * (dep + 1) + 0.8))
    ax.axis("off")

    def draw(node):
        x, yv = node["_x"], -node["_d"]
        if not node["leaf"]:
            for child, lab in ((node["left"], "yes"), (node["right"], "no")):
                ax.plot([x, child["_x"]], [yv - 0.18, -child["_d"] + 0.22], color=C["ink2"], lw=1)
                ax.text((x + child["_x"]) / 2, (yv - child["_d"]) / 2, lab, fontsize=8, color=C["ink2"], ha="center",
                        bbox=dict(fc=C["surface"], ec="none", pad=0.5))
                draw(child)
            txt = (f"{fs.LABELS[oct_cols[node['feature']]]}\n< {node['threshold']:.3g} ?\n"
                   f"{100 * node['_n'] / len(ite):.0f}% of test VMs")
            ax.text(x, yv, txt, ha="center", va="center", fontsize=8.5, color=C["ink"],
                    bbox=dict(boxstyle="round,pad=0.4", fc="#f1f0ec", ec=C["ink2"], lw=0.8))
        else:
            a = ACT[fl.MAIN_BETA][node["label"]]
            txt = f"cluster {node['label']}\nreserve {int(fl.TIERS[a] * 100)}%\n{100 * node['_n'] / len(ite):.1f}% of test VMs"
            ax.text(x, yv, txt, ha="center", va="center", fontsize=8.5, color="white" if a >= 1 else C["ink"],
                    bbox=dict(boxstyle="round,pad=0.45", fc=TIER_COL[a], ec="none"))

    draw(tree)
    ax.set_xlim(-0.7, nl - 0.3)
    ax.set_ylim(-dep - 0.6, 0.6)
    r_ = TG[(TG.method == sel_meth) & (TG["max depth"] == depth_label)].iloc[0]
    ax.set_title(f"{sel_meth} (depth {depth_label}) interpreting the K = {K} k-means clusters{note}; test agreement "
                 f"{r_['test agreement %']:.0f}%, beta = 5 cost {r_['test cost b=5']:.3f}", fontsize=10)
    fig.savefig(path)
    plt.close(fig)


draw_oct(oct_tree, d_sel, FOUT / "hw6_fig_cluster_oct.png", " (selected on validation)")
if (sel_meth, 3) in trees and d_sel != 3:     # compact 8-leaf alternative for slides (not the validation choice)
    draw_oct(trees[(sel_meth, 3)][0], 3, FOUT / "hw6_fig_cluster_oct_depth3.png", " (compact alternative)")
print(KG.to_string())
print(TG.to_string())
print(T[T.beta == 5].to_string())
print(BT.to_string())
log("Stage 9 done")
