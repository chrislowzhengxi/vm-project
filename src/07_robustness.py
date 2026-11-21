"""Stage 7 (final project): uncertainty, segments, ablation, stability, setup and censoring sensitivity.

A. Paired subscription-cluster bootstrap (1,000 reps) of test-cost differences.
B. Policy-level results by segment (lifetime, vCPU bucket, history, cold start, censoring, final day).
C. Feature-set ablation (A / B / C) for OPT-exact, greedy tree and cost-aware RF.
D. OPT sample-size and seed stability (exact DP, greedy, IAI), with runtimes.
E. Strict (label-available) vs. feasibility (all-label) training setup.
F. Censoring sensitivity: excluding right-censored test VMs and the partial final day.
Outputs: results/final_bootstrap*, final_segments*, final_ablation*, final_stability*, final_setup_comparison*,
         final_censoring_sensitivity*; figures/final_fig2_beta_ci.png, final_fig5_opt_scaling.png,
         final_fig6_segments.png
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import final_lib as fl
import final_setup as fs
from common import C, FEATS_A, FEATS_B, FEATS_C, FIG, RES, impute, load_features, plt, to_md

t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)


F, M = fs.load_strict()
q = fs.q_of(F)
itr, iva, ite = (np.flatnonzero(M[k]) for k in ("train", "val", "test"))
qv, qt = q[iva], q[ite]
AT = pd.read_parquet(fs.FINAL_DATA / "policy_actions_test.parquet")
SELT = pd.read_csv(RES / "final_tree_selected.csv")
subs_t = F.key_sub.to_numpy()[ite]


def costvec(a, qq, b):
    return a + b * np.maximum(0, qq - a)


def act(p, b):
    return AT[f"{p}|b={b}"].to_numpy(float)


MAIN = ["OPT-exact [full]", "Greedy [full]", "CART-MSE [full]", "CART-MSE (26 feat.) [full]", "Cost-aware RF",
        "RF predict + margin", "H1: customer mean P95 + margin", "H2: customer critical-ratio rule"]
EQUAL = ["OPT-exact [200k]", "OPT-IAI [200k]", "Greedy [200k]", "CART-MSE [200k]"]
MAIN = [p for p in MAIN if f"{p}|b=5" in AT.columns]
EQUAL = [p for p in EQUAL if f"{p}|b=5" in AT.columns]

# ---------------------------------------------------------------- A. bootstrap
boot = []
pairs = [("OPT-exact [full]", "Best fixed (val)"), ("OPT-exact [full]", "Greedy [full]"),
         ("OPT-exact [full]", "CART-MSE [full]"), ("OPT-exact [full]", "CART-MSE (26 feat.) [full]"),
         ("OPT-exact [full]", "Cost-aware RF"), ("OPT-exact [full]", "H1: customer mean P95 + margin"),
         ("Greedy [full]", "Best fixed (val)"), ("Cost-aware RF", "Best fixed (val)"),
         ("OPT-exact [200k]", "Greedy [200k]"), ("OPT-exact [200k]", "OPT-IAI [200k]"),
         ("OPT-IAI [200k]", "Greedy [200k]")]
pairs += [(p, "Best fixed (val)") for p in MAIN + EQUAL if (p, "Best fixed (val)") not in pairs]
for b in fl.BETAS:
    for a_, b_ in pairs:
        if f"{a_}|b={b}" not in AT.columns or f"{b_}|b={b}" not in AT.columns:
            continue
        ca, cb = costvec(act(a_, b), qt, b), costvec(act(b_, b), qt, b)
        r = fl.cluster_bootstrap_diff(ca, cb, subs_t, n_boot=1000, seed=fs.SEED)
        boot.append({"beta": b, "policy A": a_, "policy B": b_, "cost A": ca.mean(), "cost B": cb.mean(),
                     "A - B": r["mean diff"], "95% CI low": r["ci low"], "95% CI high": r["ci high"],
                     "% of B": 100 * r["mean diff"] / cb.mean(), "share of reps A<B": r["share reps < 0"]})
BT = pd.DataFrame(boot)
to_md(BT.set_index(["beta", "policy A", "policy B"]), RES / "final_bootstrap", floatfmt=".4f")
log("bootstrap done")

# Fig 2: % reduction vs best fixed with cluster-bootstrap CIs, across beta
fig_p = [p for p in ["H1: customer mean P95 + margin", "RF predict + margin", "CART-MSE (26 feat.) [full]",
                     "Greedy [full]", "OPT-IAI [200k]", "OPT-exact [full]", "Cost-aware RF"] if f"{p}|b=5" in AT.columns]
colors = [C["violet"], C["orange"], "#b7d3f6", "#86b6ef", C["green"], C["blue"], C["aqua"]]
fig, ax = plt.subplots(figsize=(9.2, 3.8))
w = 0.8 / len(fig_p)
for j, (p, colr) in enumerate(zip(fig_p, colors)):
    for i, b in enumerate(fl.BETAS):
        r = BT[(BT.beta == b) & (BT["policy A"] == p) & (BT["policy B"] == "Best fixed (val)")].iloc[0]
        mid, lo, hi = -r["% of B"], -100 * r["95% CI high"] / r["cost B"], -100 * r["95% CI low"] / r["cost B"]
        x = i + (j - (len(fig_p) - 1) / 2) * w
        ax.bar(x, mid, width=w * 0.92, color=colr, label=p if i == 0 else None)
        ax.errorbar(x, mid, yerr=[[mid - lo], [hi - mid]], color=C["ink"], lw=0.8, capsize=2)
ax.set_xticks(range(len(fl.BETAS)), [f"beta = {b}" for b in fl.BETAS])
ax.set_ylabel("% cost reduction vs best fixed\n(95% subscription-cluster CI)")
ax.set_title("Savings relative to the best fixed reservation, by under-provisioning penalty (test period)")
ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
ax.grid(axis="x", visible=False)
fig.savefig(FIG / "final_fig2_beta_ci.png")
plt.close(fig)

# ---------------------------------------------------------------- B. segments (beta = 2, 5, 10)
Ft = F.iloc[ite]
life = Ft.post_lifetime_h.to_numpy()
segs = {
    "lifetime <= 5 min": life <= 5 / 60 + 1e-9, "lifetime 5 min - 1 h": (life > 5 / 60 + 1e-9) & (life <= 1),
    "lifetime 1 h - 1 day": (life > 1) & (life <= 24), "lifetime > 1 day": life > 24,
    **{f"{int(c)} vCPU bucket" if c < 30 else ">24 vCPU bucket": Ft.cores.to_numpy() == c for c in (2, 4, 8, 24, 30)},
    "no completed sub history": Ft.sub_n_hist.to_numpy() == 0,
    "1-49 completed sub VMs": (Ft.sub_n_hist.to_numpy() >= 1) & (Ft.sub_n_hist.to_numpy() < 50),
    "50-999 completed sub VMs": (Ft.sub_n_hist.to_numpy() >= 50) & (Ft.sub_n_hist.to_numpy() < 1000),
    ">= 1000 completed sub VMs": Ft.sub_n_hist.to_numpy() >= 1000,
    "brand-new subscription (cold start)": Ft.sub_n_prior_created.to_numpy() == 0,
    "completed by trace end": Ft.post_vmdeleted.to_numpy() < fl.TRACE_END,
    "right-censored (alive at trace end)": Ft.post_vmdeleted.to_numpy() >= fl.TRACE_END,
    "days 25-28": Ft.vmcreated.to_numpy() < 29 * fl.DAY, "final partial day 29": Ft.vmcreated.to_numpy() >= 29 * fl.DAY,
}
seg_rows = []
for b in fl.BETAS:
    cf = costvec(act("Best fixed (val)", b), qt, b)
    tot_sav = {p: (cf - costvec(act(p, b), qt, b)).sum() for p in ["OPT-exact [full]", "Cost-aware RF", "Greedy [full]"]}
    for sname, msk in segs.items():
        row = {"beta": b, "segment": sname, "n": int(msk.sum()), "% of test VMs": 100 * msk.mean(),
               "subscriptions": len(np.unique(subs_t[msk])), "best fixed cost": cf[msk].mean(),
               "oracle cost": costvec(act("Oracle (hindsight)", b), qt, b)[msk].mean()}
        for p, lab in [("OPT-exact [full]", "OPT"), ("Greedy [full]", "greedy"), ("Cost-aware RF", "RF")]:
            cp_ = costvec(act(p, b), qt, b)
            row[f"{lab} cost"] = cp_[msk].mean()
            row[f"{lab} % saving"] = 100 * (cf[msk].mean() - cp_[msk].mean()) / cf[msk].mean()
            row[f"{lab} share of total saving %"] = 100 * (cf[msk] - cp_[msk]).sum() / tot_sav[p]
        seg_rows.append(row)
SG = pd.DataFrame(seg_rows)
SG.to_csv(RES / "final_segments.csv", index=False)
to_md(SG[SG.beta == fl.MAIN_BETA].set_index("segment").drop(columns="beta"), RES / "final_segments_b5", floatfmt=".3f")

fig, ax = plt.subplots(figsize=(8.6, 5.6))
s5 = SG[SG.beta == fl.MAIN_BETA].set_index("segment")
yy = np.arange(len(s5))[::-1]
ax.barh(yy + 0.2, s5["OPT % saving"], height=0.38, color=C["blue"], label="OPT-exact (depth 3)")
ax.barh(yy - 0.2, s5["RF % saving"], height=0.38, color=C["aqua"], label="cost-aware RF")
ax.axvline(0, color=C["ink2"], lw=0.8)
ax.set_yticks(yy, [f"{s}  (n={int(s5.loc[s, 'n']):,})" for s in s5.index], fontsize=8)
ax.set_xlabel("% cost reduction vs best fixed within segment (beta = 5, test period)")
ax.set_title("Where do the savings come from? Segment-level results")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.grid(axis="y", visible=False)
fig.savefig(FIG / "final_fig6_segments.png")
plt.close(fig)
log("segments done")

# ---------------------------------------------------------------- shared helpers for refitting tree policies
ranking = pd.read_csv(RES / "final_feature_selection.csv").feature.tolist()


def tree_pipeline(cols, train_idx, val_idx, FF, qq, betas=fl.BETAS, depths=fs.DEPTHS, fracs=fs.MIN_LEAF_FRACS):
    """Bin on train, fit OPT-exact and greedy over the grid, select on val, return test actions + configs."""
    binner = fl.Binner(fs.MAX_BINS).fit(FF[cols].to_numpy(float)[train_idx], cols)
    Xb = binner.transform(FF[cols].to_numpy(float))
    P, S, B = fl.aggregate(Xb[train_idx], qq[train_idx], binner.n_bins)
    n = len(train_idx)
    cfgs = [(b, max(1, int(round(f * n)))) for b in betas for f in fracs]
    Cv = {b: fl.cost_matrix(qq[val_idx], b) for b in betas}
    out = {}
    cands = {("OPT-exact", d, cb, cm): r["tree"] for d in depths for r, (cb, cm) in
             zip(fl.solve_opt_exact(P, S, B, d, cfgs), cfgs)}
    cands |= {("Greedy", d, cb, cm): fl.fit_greedy_tree(P, S, B, cb, d, cm)["tree"] for d in depths for cb, cm in cfgs}
    for method in ("OPT-exact", "Greedy"):
        for b in betas:
            best = min(((k, t) for k, t in cands.items() if k[0] == method and k[2] == b),
                       key=lambda kt: Cv[b][np.arange(len(val_idx)), fl.predict_tree(kt[1], Xb[val_idx])].mean())
            out[(method, b)] = (best[0], best[1], fl.TIERS[fl.predict_tree(best[1], Xb[ite])])
    return out


def rf_cost_actions(cols, train_idx, FF, qq):
    X = FF[cols].to_numpy(np.float32)
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=50, max_features=0.5 if len(cols) > 2 else 1.0,
                               max_samples=300_000, n_jobs=-1, random_state=fs.SEED)
    rf.fit(X[train_idx], fl.shortfall_stats(qq[train_idx])[:, 1:])
    Sh = rf.predict(X[ite])
    return {b: fl.TIERS[np.argmin(np.column_stack([fl.TIERS[:3] + b * Sh, np.ones(len(Sh))]), axis=1)]
            for b in fl.BETAS}


WT = fs.eval_weights(F, M["test"])
best_fixed = {b: act("Best fixed (val)", b)[0] for b in fl.BETAS}


def summarize(label, actions_by_method, extra=None):
    rows = []
    for (method, b), a in actions_by_method.items():
        for wname in ("per VM", "vCPU-hour-weighted"):
            w = WT[wname]
            c_ = np.average(costvec(a, qt, b), weights=w)
            cf = np.average(costvec(np.full(len(ite), best_fixed[b]), qt, b), weights=w)
            rows.append({"variant": label, "method": method, "beta": b, "weighting": wname, "test cost": c_,
                         "% reduction vs best fixed": 100 * (cf - c_) / cf} | (extra or {}))
    return rows


# ---------------------------------------------------------------- C. feature ablation
abl = []
sets = {"A: size + time (7)": FEATS_A,
        "B: A + subscription history (top 8)": [f for f in ranking if f in FEATS_B][:fs.N_OPT_FEATURES],
        "C: B + deployment history (top 8, main)": [f for f in ranking if f in FEATS_C][:fs.N_OPT_FEATURES]}
for label, cols in sets.items():
    res = tree_pipeline(cols, itr, iva, F, q)
    acts = {(m_, b): r[2] for (m_, b), r in res.items()}
    full_cols = {"A": FEATS_A, "B": FEATS_B, "C": FEATS_C}[label[0]]
    acts |= {("Cost-aware RF (all set features)", b): a for b, a in rf_cost_actions(full_cols, itr, F, q).items()}
    abl += summarize(label, acts, {"tree features": ", ".join(cols)})
    log(f"ablation {label} done")
AB = pd.DataFrame(abl)
AB.to_csv(RES / "final_ablation.csv", index=False)
to_md(AB[AB.weighting == "per VM"].pivot_table(index=["method", "variant"], columns="beta",
                                                values="% reduction vs best fixed", sort=False),
      RES / "final_ablation", floatfmt=".2f")

# ---------------------------------------------------------------- E. strict vs feasibility setup
Ff = impute(load_features(), train_mask=fs.masks_for(load_features(), strict=False)["train"])
Mf = fs.masks_for(Ff, strict=False)
itr_f, iva_f = np.flatnonzero(Mf["train"]), np.flatnonzero(Mf["val"])
SEL = sets["C: B + deployment history (top 8, main)"]
setup_rows = []
for label, FF, tr_i, va_i in [("strict (labels available by cutoff)", F, itr, iva),
                              ("feasibility (all labels, incl. future-completed)", Ff, itr_f, iva_f)]:
    res = tree_pipeline(SEL, tr_i, va_i, FF, q)
    acts = {(m_, b): r[2] for (m_, b), r in res.items()}
    acts |= {("Cost-aware RF", b): a for b, a in rf_cost_actions(FEATS_C, tr_i, FF, q).items()}
    setup_rows += summarize(label, acts, {"train rows": len(tr_i), "val rows": len(va_i)})
    log(f"setup {label} done")
SU = pd.DataFrame(setup_rows)
SU.to_csv(RES / "final_setup_comparison.csv", index=False)
to_md(SU.pivot_table(index=["method", "weighting"], columns=["variant", "beta"], values="% reduction vs best fixed",
                     sort=False), RES / "final_setup_comparison", floatfmt=".2f")

# ---------------------------------------------------------------- F. censoring / final-day sensitivity
cens = Ft.post_vmdeleted.to_numpy() >= fl.TRACE_END
day29 = Ft.vmcreated.to_numpy() >= 29 * fl.DAY
subsets = {"all test VMs": np.ones(len(ite), bool), "excl. right-censored": ~cens, "excl. day 29": ~day29,
           "excl. both": ~cens & ~day29}
cs_rows = []
for b in fl.BETAS:
    for sname, msk in subsets.items():
        cf = costvec(act("Best fixed (val)", b), qt, b)[msk].mean()
        row = {"beta": b, "subset": sname, "n": int(msk.sum()), "best fixed cost": cf}
        for p in ["OPT-exact [full]", "Greedy [full]", "Cost-aware RF", "OPT-IAI [200k]"]:
            if f"{p}|b={b}" in AT.columns:
                cp_ = costvec(act(p, b), qt, b)[msk].mean()
                row[f"{p} % red."] = 100 * (cf - cp_) / cf
        cs_rows.append(row)
to_md(pd.DataFrame(cs_rows).set_index(["beta", "subset"]), RES / "final_censoring_sensitivity", floatfmt=".3f")
log("censoring sensitivity done")

# ---------------------------------------------------------------- D. sample size and seed stability (beta = 5)
binner_js = json.loads((fs.FINAL_DATA / "binner.json").read_text())
Xb = np.load(fs.FINAL_DATA / "Xb_opt.npy")
nb = np.array([len(t) + 1 for t in binner_js["thresholds"]])
r5 = SELT[(SELT.policy == "OPT-exact [full]") & (SELT.beta == fl.MAIN_BETA)].iloc[0]
depth, frac = int(r5.max_depth), float(r5.min_leaf_frac)
ref_tree_full = fl.tree_from_json(json.loads((fs.MODELS / "OPT-exact_full_b5.json").read_text()),
                                  type("B", (), {"names": binner_js["names"]})())
ref_act = fl.predict_tree(ref_tree_full, Xb[ite])
C5 = fl.cost_matrix(qt, fl.MAIN_BETA)
stab = []
for n in (10_000, 50_000, 200_000, len(itr)):
    for seed in (range(5) if n < len(itr) else [0]):
        idx = fs.sample_indices(itr, n, seed) if n < len(itr) else itr
        P, S, B = fl.aggregate(Xb[idx], q[idx], nb)
        m = max(1, int(round(frac * len(idx))))
        fits = {"OPT-exact": fl.solve_opt_exact(P, S, B, depth, [(fl.MAIN_BETA, m)])[0],
                "Greedy": fl.fit_greedy_tree(P, S, B, fl.MAIN_BETA, depth, m)}
        if fl.iai_available() and n <= 200_000:
            o = fl.run_iai_policy_trees(Xb[idx], fl.cost_matrix(q[idx], fl.MAIN_BETA),
                                        [{"depth": depth, "minbucket": m, "seed": 1}],
                                        fs.FINAL_DATA / "iai_stability", threads=10)[0]
            fits["OPT-IAI"] = {"tree": o["tree"], "runtime_s": o["runtime_s"],
                               "train_mean_cost": fl.tree_train_cost(o["tree"], P, S, fl.MAIN_BETA) / len(idx)}
        for meth, r in fits.items():
            a = fl.predict_tree(r["tree"], Xb[ite])
            stab.append({"n": len(idx), "seed": seed, "method": meth, "train cost": r["train_mean_cost"],
                         "test cost": C5[np.arange(len(ite)), a].mean(), "runtime_s": r["runtime_s"],
                         "root feature": binner_js["names"][r["tree"]["feature"]] if not r["tree"]["leaf"] else "-",
                         "features used": ", ".join(sorted(binner_js["names"][j] for j in fl.tree_features(r["tree"]))),
                         "leaves": fl.tree_leaves(r["tree"]),
                         "test action agreement with full-data OPT %": 100 * np.mean(a == ref_act),
                         "signature": fl.tree_signature(r["tree"])})
    log(f"stability n={n:,} done")
ST = pd.DataFrame(stab)
ST.to_csv(RES / "final_stability.csv", index=False)
agg = ST.groupby(["method", "n"]).agg(runs=("seed", "size"), train_cost=("train cost", "mean"),
                                      test_cost_mean=("test cost", "mean"), test_cost_sd=("test cost", "std"),
                                      test_cost_min=("test cost", "min"), test_cost_max=("test cost", "max"),
                                      runtime_s=("runtime_s", "mean"), distinct_trees=("signature", "nunique"),
                                      root_features=("root feature", lambda s: ", ".join(sorted(set(s)))),
                                      agreement_with_full_OPT=("test action agreement with full-data OPT %", "mean"))
to_md(agg, RES / "final_stability", floatfmt=".4f")

MIO = RES / "final_mio_crosscheck.csv"
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
mcol = {"OPT-exact": C["blue"], "Greedy": "#86b6ef", "OPT-IAI": C["green"]}
for meth, g in agg.reset_index().groupby("method"):
    axes[0].errorbar(g.n, g.test_cost_mean, yerr=[g.test_cost_mean - g.test_cost_min, g.test_cost_max - g.test_cost_mean],
                     marker="o", ms=5, color=mcol[meth], label=meth, capsize=3)
    axes[1].plot(g.n, g.runtime_s, marker="o", ms=5, color=mcol[meth], label=meth)
axes[0].set_xscale("log")
axes[0].set_xlabel("training VMs (log)")
axes[0].set_ylabel("test cost, beta = 5 (mean, min-max over 5 seeds)")
axes[0].set_title(f"Test cost vs. training sample size (depth {depth})")
axes[0].legend(frameon=False, fontsize=8)
if MIO.exists():
    mio = pd.read_csv(MIO)
    mm = mio[mio.method.str.startswith("Gurobi MIO") & ~mio.method.str.contains("warm")]
    for d_, g in mm.groupby("depth"):
        axes[1].plot(g.n, g.runtime_s, marker="s", ms=6, ls="--", color=C["orange"] if d_ == 3 else C["yellow"],
                     label=f"Gurobi MIO depth {d_} (8 bins; {', '.join(g.status.astype(str))})")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("training VMs (log)")
axes[1].set_ylabel("fit time per tree (s, log)")
axes[1].set_title("Runtime: exact DP vs. local search vs. MIO")
axes[1].legend(frameon=False, fontsize=7.5, loc="lower right")
fig.savefig(FIG / "final_fig5_opt_scaling.png")
plt.close(fig)

print(BT[BT.beta == fl.MAIN_BETA].to_string())
print(agg.to_string())
log("Stage 7 done")
