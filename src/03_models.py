"""Stage 3: predictive feasibility - baselines, Ridge, CART, Random Forest; ablation; segments; importance; stress tests.

All models are fit on the TRAIN period only, model selection uses VAL, and every number reported as "test" is
computed on the last five days of the trace. Outputs: results/03_*.{csv,md}, figures/fig4-6*.png,
data/processed/predictions.parquet (val+test predictions used by 04_policies.py).
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree

from common import (C, FEATS_A, FEATS_B, FEATS_B_LEAKY, FEATS_C, FIG, PROC, RES, SEED, impute, load_features, plt,
                    to_md)

t0 = time.time()
F = impute(load_features())
tr, va, te = (F.split == s for s in ("train", "val", "test"))
y = F.y_p95.to_numpy()
ytr, yva, yte = y[tr], y[va], y[te]


def metrics(yt, yp):
    return {"MAE": mean_absolute_error(yt, yp), "RMSE": mean_squared_error(yt, yp) ** 0.5, "R2": r2_score(yt, yp)}


def rf(n_estimators=100, **kw):
    # Bootstrap subsamples of 300k rows/tree keep fitting fast; min_samples_leaf limits memorisation.
    return RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=50, max_features=0.5,
                                 max_samples=300_000, n_jobs=-1, random_state=SEED, **kw)


# ---------------------------------------------------------------- Ridge design matrix (simple, interpretable)
LOG_COLS = ["dep_batch_size", "sub_age_days", "sub_n_prior_created", "sub_n_hist", "dep_n_prior_created", "dep_n_hist"]


def ridge_X(cols):
    X = F[cols].astype(float).copy()
    for c_ in set(LOG_COLS) & set(cols):
        X[c_] = np.log1p(X[c_])
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * F.hour / 24), np.cos(2 * np.pi * F.hour / 24)
    X = X.drop(columns=["hour", "dow", "cores", "memory_gb", "mem_per_core"])
    size = F.cores.astype(int).astype(str) + "c_" + F.memory_gb.astype(int).astype(str) + "g"
    X = pd.concat([X, pd.get_dummies(size, prefix="size", drop_first=True, dtype=float),
                   pd.get_dummies(F.dow, prefix="dow", drop_first=True, dtype=float)], axis=1)
    mu, sd = X[tr].mean(), X[tr].std().replace(0, 1)          # standardise with training statistics only
    return ((X - mu) / sd).to_numpy()


results, preds = [], {}


def record(model, fs, yp_va, yp_te):
    results.append({"model": model, "features": fs, "split": "val", **metrics(yva, yp_va)})
    results.append({"model": model, "features": fs, "split": "test", **metrics(yte, yp_te)})
    preds[f"{model}|{fs}"] = np.concatenate([yp_va, yp_te])
    print(f"[{time.time() - t0:6.0f}s] {model:<22} {fs:<8} test {metrics(yte, yp_te)}", flush=True)


# ---------------------------------------------------------------- Model 0: trivial baselines
record("Baseline: train mean", "-", np.full(va.sum(), ytr.mean()), np.full(te.sum(), ytr.mean()))
record("Baseline: train median", "-", np.full(va.sum(), np.median(ytr)), np.full(te.sum(), np.median(ytr)))
size_key = F.cores.astype(str) + "/" + F.memory_gb.astype(str)
gm = pd.Series(ytr).groupby(size_key[tr].to_numpy()).mean()
record("Baseline: size-group mean", "-", size_key[va].map(gm).fillna(ytr.mean()).to_numpy(),
       size_key[te].map(gm).fillna(ytr.mean()).to_numpy())
# History-only heuristic: predict the subscription's historical mean p95 (no model fitting at all).
record("Baseline: sub hist mean", "-", F.loc[va, "sub_hist_mean"].to_numpy(), F.loc[te, "sub_hist_mean"].to_numpy())

# ---------------------------------------------------------------- Models 1-3 on feature sets A, B, C
cart_sel = []
for fs_name, cols in [("A", FEATS_A), ("B", FEATS_B), ("C", FEATS_C)]:
    X = ridge_X(cols)
    m = Ridge(alpha=1.0).fit(X[tr], ytr)
    record("Ridge", fs_name, m.predict(X[va]), m.predict(X[te]))

    Xt = F[cols].to_numpy(np.float32)
    best = None
    for leaves in (8, 32, 128, 512, 2048):               # pruning chosen on validation RMSE
        m = DecisionTreeRegressor(max_leaf_nodes=leaves, min_samples_leaf=500, random_state=SEED).fit(Xt[tr], ytr)
        r = metrics(yva, m.predict(Xt[va]))
        cart_sel.append({"features": fs_name, "max_leaf_nodes": leaves, **r})
        if best is None or r["RMSE"] < best[0]:
            best = (r["RMSE"], leaves, m)
    record(f"CART ({best[1]} leaves)", fs_name, best[2].predict(Xt[va]), best[2].predict(Xt[te]))
    if fs_name == "C":
        cart_C = best[2]

    m = rf().fit(Xt[tr], ytr)
    record("Random Forest", fs_name, m.predict(Xt[va]), m.predict(Xt[te]))
    if fs_name == "C":
        rf_C = m

res = pd.DataFrame(results)
to_md(res.pivot_table(index=["model", "features"], columns="split", values=["MAE", "RMSE", "R2"], sort=False)
      .reorder_levels([1, 0], axis=1).sort_index(axis=1, level=0, ascending=False), RES / "03_model_results")
to_md(pd.DataFrame(cart_sel).set_index(["features", "max_leaf_nodes"]), RES / "03_cart_pruning_validation")

# ---------------------------------------------------------------- Segments by history availability (test set)
Ft = F[te]
segs = {
    "all test VMs": np.ones(te.sum(), bool),
    "brand-new subscription (0 prior VMs created)": Ft.sub_n_prior_created.to_numpy() == 0,
    "no completed sub history (sub_n_hist = 0)": Ft.sub_n_hist.to_numpy() == 0,
    "sub_n_hist >= 1": Ft.sub_n_hist.to_numpy() >= 1,
    "sub_n_hist >= 5": Ft.sub_n_hist.to_numpy() >= 5,
    "sub_n_hist >= 50": Ft.sub_n_hist.to_numpy() >= 50,
    "sub_n_hist >= 1000": Ft.sub_n_hist.to_numpy() >= 1000,
    "1 <= sub_n_hist < 50": (Ft.sub_n_hist.to_numpy() >= 1) & (Ft.sub_n_hist.to_numpy() < 50),
    "no completed deployment history": Ft.dep_n_hist.to_numpy() == 0,
}
nva = va.sum()
seg_rows = []
for seg, msk in segs.items():
    row = {"segment": seg, "n": int(msk.sum()), "subs": Ft.key_sub[msk].nunique()}
    for key, lab in [("Baseline: train mean|-", "mean"), ("Ridge|C", "Ridge C"), ("Random Forest|A", "RF A"),
                     ("Random Forest|B", "RF B"), ("Random Forest|C", "RF C")]:
        p = preds[key][nva:][msk]
        if msk.sum() >= 30:
            row[f"MAE {lab}"] = mean_absolute_error(yte[msk], p)
            row[f"R2 {lab}"] = r2_score(yte[msk], p) if msk.sum() > 1 else np.nan
    seg_rows.append(row)
seg = pd.DataFrame(seg_rows).set_index("segment")
to_md(seg, RES / "03_segments_by_history", floatfmt=".3f")

# ---------------------------------------------------------------- Feature importance (RF C, CART C)
rng = np.random.default_rng(SEED)
sub_idx = rng.choice(np.flatnonzero(te), 60_000, replace=False)
Xs, ys = F.loc[sub_idx, FEATS_C].to_numpy(np.float32), y[sub_idx]
GROUPS = {"VM size (cores, memory)": ["cores", "memory_gb", "mem_per_core"],
          "creation time (hour, dow, day)": ["hour", "dow", "day_in_trace"],
          "deployment batch size": ["dep_batch_size"],
          "subscription meta (age, counts)": ["sub_age_days", "sub_preexisting", "sub_n_prior_created", "sub_n_hist", "sub_cold"],
          "subscription p95 history": [c_ for c_ in FEATS_C if c_.startswith("sub_hist_")],
          "deployment history": [c_ for c_ in FEATS_C if c_.startswith("dep_") and c_ != "dep_batch_size"]}


def perm_importance(model, cols_to_perm, repeats=3):
    base = mean_absolute_error(ys, model.predict(Xs))
    idx = [FEATS_C.index(c_) for c_ in cols_to_perm]
    out = []
    for _ in range(repeats):
        Xp = Xs.copy()
        perm = rng.permutation(len(Xp))
        Xp[:, idx] = Xp[perm][:, idx]                      # permute the block jointly (keeps within-group structure)
        out.append(mean_absolute_error(ys, model.predict(Xp)) - base)
    return np.mean(out)


imp = pd.DataFrame({
    "RF perm. dMAE": [perm_importance(rf_C, [f]) for f in FEATS_C],
    "CART perm. dMAE": [perm_importance(cart_C, [f]) for f in FEATS_C],
    "RF impurity": rf_C.feature_importances_, "CART impurity": cart_C.feature_importances_}, index=FEATS_C)
imp = imp.sort_values("RF perm. dMAE", ascending=False)
to_md(imp, RES / "03_feature_importance", floatfmt=".3f")
gimp = pd.DataFrame({"RF grouped perm. dMAE": {g: perm_importance(rf_C, cols) for g, cols in GROUPS.items()},
                     "CART grouped perm. dMAE": {g: perm_importance(cart_C, cols) for g, cols in GROUPS.items()}})
gimp = gimp.sort_values("RF grouped perm. dMAE", ascending=False)
to_md(gimp, RES / "03_grouped_importance", floatfmt=".3f")
print(f"[{time.time() - t0:6.0f}s] importance done", flush=True)

fig, ax = plt.subplots(figsize=(6.2, 3.0))
yy = np.arange(len(gimp))[::-1]
ax.barh(yy, gimp["RF grouped perm. dMAE"], color=C["blue"], height=0.6)
for yi, v in zip(yy, gimp["RF grouped perm. dMAE"]):
    ax.text(v + 0.2, yi, f"{v:.1f}", va="center", color=C["ink"])
ax.set_yticks(yy, gimp.index)
ax.set_xlabel("increase in test MAE when the feature group is permuted (pct-points)")
ax.set_title("Random Forest (feature set C): grouped permutation importance")
ax.grid(axis="y", visible=False)
fig.savefig(FIG / "fig4_grouped_importance.png")
plt.close(fig)

# Depth-3 regression tree (interpretability proxy)
tree3 = DecisionTreeRegressor(max_depth=3, min_samples_leaf=500, random_state=SEED).fit(F.loc[tr, FEATS_C], ytr)
results_tree3 = metrics(yte, tree3.predict(F.loc[te, FEATS_C]))
fig, ax = plt.subplots(figsize=(17, 6.5))
plot_tree(tree3, feature_names=FEATS_C, filled=False, impurity=False, proportion=True, precision=1, fontsize=8, ax=ax)
ax.set_title(f"Depth-3 CART regressor for P95 CPU (test R2 = {results_tree3['R2']:.3f}); value = mean P95 in node")
fig.savefig(FIG / "fig5_depth3_tree.png")
plt.close(fig)

# ---------------------------------------------------------------- Stress tests
stress = []


def add_stress(name, yt, yp, note=""):
    stress.append({"check": name, "n": len(yt), **metrics(yt, yp), "note": note})


p_rf = preds["Random Forest|C"][nva:]
add_stress("RF C, all test", yte, p_rf)
# (a) distribution shift: by test day + validation
pv = preds["Random Forest|C"][:nva]
add_stress("RF C, validation period (days 21-24)", yva, pv)
day = (Ft.vmcreated // 86400).to_numpy()
for d_ in np.unique(day):
    m_ = day == d_
    add_stress(f"RF C, test day {d_}", yte[m_], p_rf[m_], f"share right-censored={(Ft.post_vmdeleted.to_numpy()[m_] == 2591400).mean():.3f}")
# (b) lifetime buckets (post-hoc slicing only; lifetime is never a feature)
life = Ft.post_lifetime_h.to_numpy()
for lo, hi, lab in [(0, 5 / 60 + 1e-9, "<= 5 min"), (5 / 60 + 1e-9, 1, "5 min - 1 h"), (1, 24, "1 h - 1 day"), (24, 1e9, "> 1 day")]:
    m_ = (life >= lo) & (life < hi)
    add_stress(f"RF C, lifetime {lab}", yte[m_], p_rf[m_], f"mean p95={yte[m_].mean():.1f}")
# (c) concentration: excluding top-10 subscriptions; subscription-macro MAE
top10 = F.loc[tr, "key_sub"].value_counts().index[:10]
m_ = ~Ft.key_sub.isin(top10).to_numpy()
add_stress("RF C, test excl. top-10 training subscriptions", yte[m_], p_rf[m_])
add_stress("Train-mean baseline, test excl. top-10 subs", yte[m_], np.full(m_.sum(), ytr.mean()))
macro = pd.DataFrame({"s": Ft.key_sub.to_numpy(), "e_rf": np.abs(yte - p_rf), "e_mean": np.abs(yte - ytr.mean())}).groupby("s").mean()
stress.append({"check": "Subscription-macro-averaged MAE (each test sub weighted equally)", "n": len(macro),
               "MAE": macro.e_rf.mean(), "note": f"train-mean baseline macro MAE={macro.e_mean.mean():.2f}"})

# (d) leakage sensitivity: lenient history (peeks at running VMs' final p95) vs strict
Xl = F[FEATS_B_LEAKY].to_numpy(np.float32)
m = rf().fit(Xl[tr], ytr)
add_stress("LEAKY: RF with lenient sub history (A + running-VM p95)", yte, m.predict(Xl[te]), "compare with RF B")
# (e) leakage demo: what adding lifetime + vmcategory would do (NOT allowed in the real model)
Xk = np.column_stack([F[FEATS_C].to_numpy(np.float32), np.log1p(F.post_lifetime_h.to_numpy()),
                      pd.factorize(F.post_vmcategory)[0]]).astype(np.float32)
m = rf().fit(Xk[tr], ytr)
add_stress("LEAKY: RF C + lifetime + vmcategory", yte, m.predict(Xk[te]), "illustrates leakage, not usable")
# (f) dep_batch_size sensitivity (is the request batch size known? assume yes, check dependence)
cols_nb = [c_ for c_ in FEATS_C if c_ != "dep_batch_size"]
Xn = F[cols_nb].to_numpy(np.float32)
m = rf().fit(Xn[tr], ytr)
add_stress("RF C without dep_batch_size", yte, m.predict(Xn[te]))
# (g) label availability: train only on VMs whose label was observable before the train cutoff (deleted < day 21)
lab_ok = tr & (F.post_vmdeleted < 21 * 86400)
Xt = F[FEATS_C].to_numpy(np.float32)
m = rf().fit(Xt[lab_ok], y[lab_ok])
add_stress("RF C trained only on labels observable by day 21", yte, m.predict(Xt[te]), f"train rows={lab_ok.sum():,}")
st = pd.DataFrame(stress).set_index("check")
to_md(st, RES / "03_stress_tests", floatfmt=".3f")

# Fig 6: test MAE by history segment and by day (compact two-panel)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axes[0]
show = ["all test VMs", "no completed sub history (sub_n_hist = 0)", "1 <= sub_n_hist < 50", "sub_n_hist >= 50", "sub_n_hist >= 1000"]
lab = ["all", "no sub history", "1-49 prior", ">= 50 prior", ">= 1000 prior"]
xx = np.arange(len(show))
for j, (col, colr, nm) in enumerate([("MAE mean", C["ink2"], "train mean"), ("MAE RF A", C["orange"], "RF A (raw)"),
                                     ("MAE RF C", C["blue"], "RF C (raw+history)")]):
    ax.bar(xx + (j - 1) * 0.26, seg.loc[show, col], width=0.24, color=colr, label=nm)
ax.set_xticks(xx, [f"{l}\n(n={seg.loc[s, 'n']:,})" for l, s in zip(lab, show)], fontsize=8)
ax.set_ylabel("test MAE (pct-points)")
ax.set_title("Test MAE by amount of subscription history")
ax.set_ylim(0, 44)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
ax.grid(axis="x", visible=False)
ax = axes[1]
dd = st.loc[[i for i in st.index if i.startswith("RF C, test day")]]
ax.plot([int(i.split()[-1]) for i in dd.index], dd.R2, marker="o", color=C["blue"], ms=6)
ax.axhline(st.loc["RF C, validation period (days 21-24)", "R2"], color=C["ink2"], ls="--", lw=1)
ax.text(28.05, st.loc["RF C, validation period (days 21-24)", "R2"] - 0.009, "validation R2", color=C["ink2"])
ax.set_xticks(sorted(int(i.split()[-1]) for i in dd.index))
ax.set_xlabel("test day")
ax.set_ylabel("R2 (RF C)")
ax.set_title("Distribution shift: R2 by test day")
fig.savefig(FIG / "fig6_segments_and_shift.png")
plt.close(fig)

# ---------------------------------------------------------------- save predictions for stage 4
P = F.loc[va | te, ["split", "y_p95", "key_sub", "cores", "post_lifetime_h"]].copy()
for k_ in ["Ridge|C", "Random Forest|C", "Random Forest|A", "Random Forest|B"] + [k_ for k_ in preds if k_.startswith("CART") and k_.endswith("|C")]:
    P["pred_" + k_.split("|")[0].split(" (")[0].replace(" ", "_") + "_" + k_.split("|")[1]] = preds[k_]
P.to_parquet(PROC / "predictions.parquet")
print(res.query("split=='test'").to_string())
print(seg.to_string())
print(gimp.to_string())
print(imp.head(12).to_string())
print(st.to_string())
print(f"Stage 3 done in {time.time() - t0:.0f}s")
