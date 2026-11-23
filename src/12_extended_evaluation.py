"""Stage 12: extended evaluation beyond modeled cost (operational risk, customers, segments, prediction, regret).

Question: are the policies operationally sensible and robust, not merely cheaper under
    C(r, Y; beta) = r + beta * max(0, Y - r)?
The objective is unchanged and no policy is retrained. Everything is computed from saved test actions, saved forest
predictions and saved tree JSONs (the 4-leaf, leaf-penalized and certified depth-4 OPT trees are applied to the saved
binned features; their test costs are asserted equal to the recorded stage-11 values). Ridge and the stage-6 CART
regressor (32 leaves) are refit only to recover their test predictions (seconds); both are verified against the
saved stage-6 actions and R^2 values.

  A  operational metrics at beta 2/5/10: reservation, tier mix, exceedance, shortfall severity, cost decomposition
  B  savings mechanism: reserving less vs. targeting (untargeted counterfactuals with the same tier mix / reservation)
  C  cost vs. risk: Pareto dominance; risk frontiers over the stage-10 beta grid; matched-risk comparisons
  D  customer (subscription) heterogeneity; VM- vs. equal-subscription weighting
  E  workload segments (creation-time segments; lifetime and realized P95 are retrospective diagnostics only)
  F  predictive performance (Ridge / CART / RF), RF calibration, prediction errors -> reservation mistakes
  G  oracle regret, share of attainable saving captured, policy disagreement
  H  paired subscription-cluster bootstrap for a small pre-specified set of comparisons (unadjusted)
Test data are used only for retrospective evaluation: nothing here selects beta, a model or a configuration.
Outputs: results/extended_evaluation/ext_*, figures/extended_evaluation/ext_fig*.png
"""
import argparse
import json
import pickle
import platform
import resource
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

import evaluation_metrics_lib as em
import final_lib as fl
import final_setup as fs
from common import C, FEATS_C, FIG, PROC, RES, plt, to_md

ap = argparse.ArgumentParser()
ap.add_argument("--n-boot", type=int, default=1000)
ap.add_argument("--min-sub-vms", type=int, default=5, help="minimum test VMs for the primary subscription analysis")
args = ap.parse_args()

OUT, FOUT, SENS = RES / "extended_evaluation", FIG / "extended_evaluation", RES / "sensitivity"
for d in (OUT, FOUT):
    d.mkdir(parents=True, exist_ok=True)
t_start = time.time()
runlog = {"seed": fs.SEED, "n_boot": args.n_boot, "min_sub_vms": args.min_sub_vms, "python": platform.python_version(),
          "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "checks": []}


def log(msg):
    print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)


def check(cond, what):
    assert cond, what
    runlog["checks"].append(what)


BETAS = fl.BETAS
F, M = fs.load_strict()
q = fs.q_of(F)
itr, iva, ite = (np.flatnonzero(M[k]) for k in ("train", "val", "test"))
qt = q[ite]
Ft = F.iloc[ite]
subs = Ft.key_sub.to_numpy()
NT = len(ite)


def cvec(a, b):
    return a + b * np.maximum(0, qt - a)


# =========================================================================== policies (saved actions / saved trees)
PA = pd.read_parquet(fs.FINAL_DATA / "policy_actions_test.parquet")
HW = pd.read_parquet(PROC / "hw6" / "forest_actions_test.parquet")
check(np.array_equal(PA.index.to_numpy(), ite) and np.array_equal(HW.index.to_numpy(), ite), "saved actions aligned with strict test rows")
js = json.loads((fs.FINAL_DATA / "binner.json").read_text())
BN = type("Bn", (), {"names": js["names"]})()
Xbt = np.load(fs.FINAL_DATA / "Xb_opt.npy")[ite]
SELT = pd.read_csv(RES / "final_tree_selected.csv")
E5 = pd.read_csv(SENS / "s11_e5_four_vs_eight_leaves.csv").set_index("beta")
E6 = pd.read_csv(SENS / "s11_e6_selection.csv")
RUNS = [json.loads(p.read_text()) for p in sorted((SENS / "depth4").glob("run_*.json"))]


def tree_act(tjs):
    t = fl.tree_from_json(tjs, BN)
    return fl.TIERS[fl.predict_tree(t, Xbt)], fl.tree_leaves(t), fl.tree_depth(t)


POL, INFO = {}, {}


def add(name, b, a, info):
    POL.setdefault(name, {})[b] = np.asarray(a, float)
    INFO[(name, b)] = info


for b in BETAS:
    bf = PA[f"Best fixed (val)|b={b}"].to_numpy(float)
    add("Best fixed (val)", b, bf, f"always {int(round(100 * bf[0]))}% (chosen on validation)")
    add("H1 customer mean + margin", b, PA[f"H1: customer mean P95 + margin|b={b}"], "1 parameter")
    add("RF point -> tier", b, PA[f"RF predict -> tier|b={b}"], "100 trees, P95 regression")
    add("RF point + margin", b, PA[f"RF predict + margin|b={b}"], "same forest + val-tuned margin")
    for nm, src in (("Greedy depth 3", "Greedy [full]"), ("OPT depth 3", "OPT-exact [full]")):
        r = SELT[(SELT.policy == src) & (SELT.beta == b)].iloc[0]
        add(nm, b, PA[f"{src}|b={b}"], f"depth {int(r.depth)}, {int(r.leaves)} leaves")
    a, L, D = tree_act(json.loads((SENS / f"s11_e5_opt_depth2_b{b}.txt").read_text()))
    check(np.isclose(cvec(a, b).mean(), E5.loc[b, "test cost 4 leaves"], rtol=1e-12), f"E5 4-leaf tree reproduces b={b}")
    add("OPT 4-leaf (depth 2)", b, a, f"depth {D}, {L} leaves")
    a, L, D = tree_act(json.loads((SENS / f"s11_e6_selected_b{b}_simple.json").read_text()))
    e6 = E6[(E6.beta == b) & E6["selection rule"].str.startswith("simplest")].iloc[0]
    check(np.isclose(cvec(a, b).mean(), e6["test cost"], rtol=1e-12), f"E6 penalized tree reproduces b={b}")
    add("OPT leaf-penalized (val-selected)", b, a, f"depth {D}, {L} leaves")
    cand = [r for r in RUNS if r["beta"] == b and r["bins"] == 32 and r["status"].startswith("OPTIMAL")]
    r = min(cand, key=lambda x: x["depth4"]["val_cost"])
    a, L, D = tree_act(r["depth4"]["tree"])
    check(np.isclose(cvec(a, b).mean(), r["depth4"]["test_cost"], rtol=1e-12), f"E7 depth-4 tree reproduces b={b}")
    add("OPT depth 4 (certified)", b, a, f"depth {D}, {L} leaves, min leaf {100 * r['min_leaf_frac']:g}%")
    add("Cost-aware RF", b, PA[f"Cost-aware RF|b={b}"], "100 trees, shortfall targets")
    add("Forest-weighted prescription", b, HW[f"C: forest-weighted prescription [main forest, msl=50]|b={b}"],
        "same forest as RF point; weighted shortfalls")
    add("Oracle (hindsight)", b, PA[f"Oracle (hindsight)|b={b}"], "infeasible benchmark")

MAIN = ["Best fixed (val)", "H1 customer mean + margin", "RF point -> tier", "RF point + margin", "Greedy depth 3",
        "OPT depth 3", "OPT 4-leaf (depth 2)", "OPT leaf-penalized (val-selected)", "OPT depth 4 (certified)",
        "Cost-aware RF", "Forest-weighted prescription", "Oracle (hindsight)"]
DEPLOY = [p for p in MAIN if p != "Oracle (hindsight)"]

# consistency with the finalized stage-6 and HW6 tables
MT = pd.read_csv(RES / "final_main_table.csv")
SRC6 = {"Best fixed (val)": "Best fixed (val)", "H1 customer mean + margin": "H1: customer mean P95 + margin",
        "RF point -> tier": "RF predict -> tier", "RF point + margin": "RF predict + margin", "Greedy depth 3": "Greedy [full]",
        "OPT depth 3": "OPT-exact [full]", "Cost-aware RF": "Cost-aware RF", "Oracle (hindsight)": "Oracle (hindsight)"}
HWR = pd.read_csv(RES / "hw6" / "hw6_forest_results.csv")
for b in BETAS:
    for nm, src in SRC6.items():
        row = MT[(MT.beta == b) & (MT.policy == src)].iloc[0]
        ev = fl.evaluate(POL[nm][b], qt, betas=(b,))
        check(all(np.isclose(ev[k], row[k2], rtol=1e-10) for k, k2 in ((f"cost b={b}", "test cost"), ("avg reserved", "avg reserved"),
                                                                       ("% under-provisioned", "% under-provisioned"),
                                                                       ("mean shortfall", "mean shortfall"))),
              f"stage-6 metrics reproduced: {nm} b={b}")
    row = HWR[(HWR.beta == b) & (HWR.policy == "C: forest-weighted prescription [main forest, msl=50]")].iloc[0]
    check(np.isclose(cvec(POL["Forest-weighted prescription"][b], b).mean(), row["test cost"], rtol=1e-10),
          f"HW6 forest-weighted cost reproduced b={b}")
log(f"policies loaded and verified ({len(runlog['checks'])} checks)")

# =========================================================================== A. operational metrics + regret
rows = []
for b in BETAS:
    cf = cvec(POL["Best fixed (val)"][b], b).mean()
    co = cvec(POL["Oracle (hindsight)"][b], b).mean()
    for p in MAIN:
        m = em.policy_metrics(POL[p][b], qt, b)
        rows.append({"beta": b, "policy": p, "complexity": INFO[(p, b)]} | m | em.oracle_regret(m["cost"], cf, co))
OP = pd.DataFrame(rows)
OP.to_csv(OUT / "ext_operational_metrics.csv", index=False)


def pct(x):
    return 100 * x


def main_table(d):
    return pd.DataFrame({
        "test cost": d.cost.values, "% saving vs best fixed": d["% saving vs fixed"].values,
        "E[r] (% of VM)": pct(d["E[r]"]).values, "P(Y>r) %": pct(d["P(Y>r)"]).values, "E[(Y-r)+] (pp)": pct(d["E[sf]"]).values,
        "E[Y-r | Y>r] (pp)": pct(d["E[sf | Y>r]"]).values, "P(Y-r>10pp) %": pct(d["P(sf>10pp)"]).values,
        "P(Y-r>20pp) %": pct(d["P(sf>20pp)"]).values,
        "% of attainable saving captured": d["% of attainable saving captured"].values,
        "regret vs oracle (% of oracle cost)": d["regret % of oracle cost"].values, "complexity": d.complexity.values},
        index=d.policy.values)


def severity_table(d):
    return pd.DataFrame({
        "P(Y>r) %": pct(d["P(Y>r)"]).values, "E[(Y-r)+] (pp)": pct(d["E[sf]"]).values,
        "mean | Y>r (pp)": pct(d["E[sf | Y>r]"]).values, "P50 | Y>r": pct(d["sf P50 | Y>r"]).values,
        "P90 | Y>r": pct(d["sf P90 | Y>r"]).values, "P95 | Y>r": pct(d["sf P95 | Y>r"]).values,
        "P99 | Y>r": pct(d["sf P99 | Y>r"]).values, "P90 (all VMs)": pct(d["sf P90"]).values,
        "P95 (all VMs)": pct(d["sf P95"]).values, "P99 (all VMs)": pct(d["sf P99"]).values,
        "P(Y-r>5pp) %": pct(d["P(sf>5pp)"]).values, "P(Y-r>10pp) %": pct(d["P(sf>10pp)"]).values,
        "P(Y-r>20pp) %": pct(d["P(sf>20pp)"]).values, "max (extreme, 1 VM)": pct(d["max sf"]).values}, index=d.policy.values)


def decomposition_table(d, b):
    return pd.DataFrame({
        "reservation cost E[r]": d["reservation cost"].values, f"shortfall penalty {b}*E[(Y-r)+]": d["shortfall penalty"].values,
        "total cost": d.cost.values, "penalty share of cost %": 100 * (d["shortfall penalty"] / d.cost).values,
        "E[unused] = E[(r-Y)+]": d["E[unused]"].values,
        **{f"% at {t}%": pct(d[f"share {t}%"]).values for t in (25, 50, 75, 100)}}, index=d.policy.values)


for b in BETAS:
    d = OP[OP.beta == b]
    to_md(main_table(d), OUT / f"ext_main_table_b{b}", floatfmt=".3f")
    to_md(severity_table(d), OUT / f"ext_shortfall_severity_b{b}", floatfmt=".2f")
    to_md(decomposition_table(d, b), OUT / f"ext_cost_decomposition_b{b}", floatfmt=".4f")
log("A operational metrics done")

# =========================================================================== B. savings mechanism
mech = []
for b in BETAS:
    mf = em.policy_metrics(POL["Best fixed (val)"][b], qt, b)
    for p in MAIN[1:]:
        a = POL[p][b]
        m = em.policy_metrics(a, qt, b)
        sf_u = em.untargeted_shortfall(a, qt)
        c_u = m["E[r]"] + b * sf_u
        r_mix = em.fixed_mixture_reservation(qt, m["E[sf]"])
        mech.append({"beta": b, "policy": p, "% saving vs best fixed": 100 * (mf["cost"] - m["cost"]) / mf["cost"],
                     "reservation reduction vs fixed (pp of VM)": 100 * (mf["E[r]"] - m["E[r]"]),
                     "added shortfall penalty (pp of cost)": 100 * (m["shortfall penalty"] - mf["shortfall penalty"]),
                     "E[(Y-r)+] policy (pp)": 100 * m["E[sf]"],
                     "E[(Y-r)+] same tier mix, random assignment (pp)": 100 * sf_u,
                     "E[(Y-r)+] fixed-tier mixture with same E[r] (pp)": 100 * em.fixed_mixture_shortfall(qt, m["E[r]"]),
                     "% of untargeted shortfall avoided": 100 * (1 - m["E[sf]"] / sf_u) if sf_u > 0 else np.nan,
                     "% saving of same tier mix at random": 100 * (mf["cost"] - c_u) / mf["cost"],
                     "E[r] of fixed-tier mixture with same E[(Y-r)+] (%)": 100 * r_mix,
                     "reservation saved at equal E[(Y-r)+] vs fixed mixture (pp)": 100 * (r_mix - m["E[r]"])})
MECH = pd.DataFrame(mech)
MECH.to_csv(OUT / "ext_savings_mechanism.csv", index=False)
for b in BETAS:
    to_md(MECH[MECH.beta == b].drop(columns="beta").set_index("policy"), OUT / f"ext_savings_mechanism_b{b}", floatfmt=".2f")
log("B mechanism done")

# =========================================================================== C. cost vs risk: Pareto, frontier, matched risk
DIMS = {"reservation vs expected shortfall": ["E[r]", "E[sf]"], "reservation vs exceedance": ["E[r]", "P(Y>r)"],
        "reservation vs severe shortfall (>20pp)": ["E[r]", "P(sf>20pp)"], "cost vs exceedance": ["cost", "P(Y>r)"],
        "reservation, shortfall, exceedance, severe (all 4)": ["E[r]", "E[sf]", "P(Y>r)", "P(sf>20pp)"]}
par = []
for b in BETAS:
    d = OP[(OP.beta == b) & OP.policy.isin(DEPLOY)].reset_index(drop=True)
    for dn, cols in DIMS.items():
        Pm = d[cols].to_numpy(float)
        dom = em.pareto_dominated(Pm)
        for i, p in enumerate(d.policy):
            by = [d.policy[j] for j in range(len(d)) if j != i and np.all(Pm[j] <= Pm[i] + 1e-12) and np.any(Pm[j] < Pm[i] - 1e-12)]
            par.append({"beta": b, "dimensions": dn, "policy": p, "dominated": bool(dom[i]), "dominated by": "; ".join(by)})
PAR = pd.DataFrame(par)
PAR.to_csv(OUT / "ext_pareto.csv", index=False)
to_md(PAR.pivot_table(index="policy", columns=["beta", "dimensions"], values="dominated", aggfunc="first", sort=False)
      .loc[DEPLOY].replace({True: "dominated", False: "-"}), OUT / "ext_pareto_summary")

# risk frontier over the stage-10 beta grid (existing validation-selected configurations; retrospective test metrics)
E2P = pickle.load(open(PROC / "sensitivity" / "e2_policies.pkl", "rb"))["POLT_tier_index"]
GRID = sorted({k[1] for k in E2P})
FRM = {"OPT depth 3": "OPT-exact (val-selected)", "Greedy depth 3": "Greedy (val-selected)", "Cost-aware RF": "Cost-aware RF",
       "Forest-weighted prescription": "Forest-weighted prescription", "RF point + margin": "RF point + margin",
       "H1 customer mean + margin": "H1 customer mean + margin"}
fr = []
for nm, key in FRM.items():
    for g in GRID:
        a = fl.TIERS[E2P[(key, g)]]
        if g in BETAS:
            check(np.array_equal(a, POL[nm][int(g)]), f"stage-10 grid actions equal main actions: {nm} b={g:g}")
        m = em.policy_metrics(a, qt, g)
        fr.append({"method": nm, "beta": g, "E[r]": m["E[r]"], "E[sf]": m["E[sf]"], "P(Y>r)": m["P(Y>r)"],
                   "P(sf>20pp)": m["P(sf>20pp)"], "cost at own beta": m["cost"]})
for t in fl.TIERS:
    m = em.policy_metrics(np.full(NT, t), qt, 5)
    fr.append({"method": "Fixed tier", "beta": np.nan, "tier": t, "E[r]": m["E[r]"], "E[sf]": m["E[sf]"], "P(Y>r)": m["P(Y>r)"],
               "P(sf>20pp)": m["P(sf>20pp)"]})
FR = pd.DataFrame(fr)
FR.to_csv(OUT / "ext_risk_frontier_grid.csv", index=False)

mr = []
for b in BETAS:
    ref = OP[(OP.beta == b) & (OP.policy == "OPT depth 3")].iloc[0]
    for risk in ("E[sf]", "P(Y>r)", "P(sf>20pp)"):
        x0 = ref[risk]
        for nm in list(FRM) + ["Fixed tier"]:
            g = FR[FR.method == nm]
            fv = em.mixture_frontier(g[risk], g["E[r]"], x0)
            fixed = nm == "Fixed tier"
            j = int(np.argmin(np.abs(g[risk].to_numpy() - x0)))
            near = g.iloc[j]
            rel = abs(near[risk] - x0) / x0 if x0 > 0 else np.nan
            unit = "pp of VM" if risk == "E[sf]" else "% of VMs"
            mr.append({"reference": f"OPT depth 3 at beta={b}", "risk measure": {"E[sf]": "E[(Y-r)+]", "P(Y>r)": "P(Y>r)", "P(sf>20pp)": "P(Y-r>20pp)"}[risk],
                       "risk unit": unit, "reference risk": 100 * x0, "reference E[r] (%)": 100 * ref["E[r]"], "method": nm,
                       "frontier E[r] at reference risk (%)": 100 * fv,
                       "frontier E[r] minus OPT's own E[r] (pp)": 100 * (fv - ref["E[r]"]),
                       "nearest single configuration": "-" if fixed else f"beta={near.beta:g}",
                       "nearest config risk": np.nan if fixed else 100 * near[risk],
                       "nearest config E[r] (%)": np.nan if fixed else 100 * near["E[r]"],
                       "nearest config relative risk gap": np.nan if fixed else rel,
                       "close single-config match (<= 10%)": "-" if fixed else ("yes" if rel <= 0.10 else "no")})
MR = pd.DataFrame(mr)
to_md(MR.set_index(["reference", "risk measure", "method"]), OUT / "ext_matched_risk", floatfmt=".2f")   # one file pair, % / pp units
log("C cost-vs-risk done")

# =========================================================================== D. customer-level heterogeneity
CUST = ["OPT depth 3", "Cost-aware RF", "Forest-weighted prescription", "Greedy depth 3", "RF point + margin",
        "OPT 4-leaf (depth 2)", "H1 customer mean + margin"]
STRATA = [("1-4 test VMs", 1, 4), ("5-19", 5, 19), ("20-99", 20, 99), ("100-999", 100, 999), (">= 1000", 1000, np.inf)]
cust, persub = [], {}
for b in BETAS:
    cb = cvec(POL["Best fixed (val)"][b], b)
    for p in CUST:
        a = POL[p][b]
        sf = np.maximum(0, qt - a)
        g, s = em.group_sums(subs, cb=cb, cp=cvec(a, b), sf=sf, exc=sf > em.EPS)
        n = s["n"]
        sav = 100 * (s["cb"] - s["cp"]) / s["cb"]
        dpv = (s["cb"] - s["cp"]) / n
        status = np.where(dpv > 1e-9, "lower", np.where(dpv < -1e-9, "higher", "equal"))
        exc_s, sf_s = 100 * s["exc"] / n, 100 * s["sf"] / n
        tot = (s["cb"] - s["cp"]).sum()
        if b == fl.MAIN_BETA:
            persub[p] = pd.DataFrame({"key_sub": g, "test VMs": n.astype(int), "baseline cost": s["cb"] / n, "policy cost": s["cp"] / n,
                                      "% saving": sav, "status": status, "exceedance %": exc_s, "E[(Y-r)+] pp": sf_s})

        def summ(mask, label):
            ps = sav[mask]
            return {"beta": b, "policy": p, "subscriptions": label, "n subs": int(mask.sum()), "test VMs": int(n[mask].sum()),
                    "% of test VMs": 100 * n[mask].sum() / NT,
                    "VM-weighted % saving": 100 * (s["cb"][mask] - s["cp"][mask]).sum() / s["cb"][mask].sum(),
                    "equal-sub mean % saving": ps.mean(), "median % saving": np.median(ps),
                    "P25": np.percentile(ps, 25), "P75": np.percentile(ps, 75), "P10": np.percentile(ps, 10),
                    "P90": np.percentile(ps, 90), "% subs lower cost": 100 * np.mean(status[mask] == "lower"),
                    "% subs equal": 100 * np.mean(status[mask] == "equal"), "% subs higher cost": 100 * np.mean(status[mask] == "higher"),
                    "% of VMs in higher-cost subs": 100 * n[mask & (status == "higher")].sum() / n[mask].sum(),
                    "median loss among higher-cost subs %": -np.median(ps[status[mask] == "higher"]) if (status[mask] == "higher").any() else np.nan,
                    "% subs losing > 10%": 100 * np.mean(ps < -10),
                    "share of total net saving %": 100 * (s["cb"][mask] - s["cp"][mask]).sum() / tot,
                    "VM-weighted exceedance %": 100 * s["exc"][mask].sum() / n[mask].sum(),
                    "equal-sub mean exceedance %": exc_s[mask].mean(), "median sub exceedance %": np.median(exc_s[mask]),
                    "P90 sub exceedance %": np.percentile(exc_s[mask], 90),
                    "% subs with exceedance > 25%": 100 * np.mean(exc_s[mask] > 25),
                    "equal-sub mean E[(Y-r)+] pp": sf_s[mask].mean(), "P90 sub E[(Y-r)+] pp": np.percentile(sf_s[mask], 90)}
        cust.append(summ(n >= 1, "all (>= 1 test VM)"))
        cust.append(summ(n >= args.min_sub_vms, f">= {args.min_sub_vms} test VMs (primary)"))
        for lab, lo, hi in STRATA:
            cust.append(summ((n >= lo) & (n <= hi), lab))
        order = np.argsort(-n)
        for k in (10, 50):
            cust.append({"beta": b, "policy": p, "subscriptions": f"top {k} by test VMs", "n subs": k,
                         "test VMs": int(n[order[:k]].sum()), "% of test VMs": 100 * n[order[:k]].sum() / NT,
                         "VM-weighted % saving": 100 * (s["cb"][order[:k]] - s["cp"][order[:k]]).sum() / s["cb"][order[:k]].sum(),
                         "share of total net saving %": 100 * (s["cb"][order[:k]] - s["cp"][order[:k]]).sum() / tot})
CU = pd.DataFrame(cust)
CU.to_csv(OUT / "ext_customer_heterogeneity.csv", index=False)
for b in BETAS:
    to_md(CU[(CU.beta == b) & CU.policy.isin(["OPT depth 3", "Cost-aware RF", "Forest-weighted prescription", "Greedy depth 3"])]
          .drop(columns="beta").set_index(["policy", "subscriptions"]), OUT / f"ext_customer_heterogeneity_b{b}", floatfmt=".1f")
pd.concat([v.assign(policy=k) for k, v in persub.items()]).to_csv(OUT / "ext_customer_per_subscription_b5.csv", index=False)

# VM-weighted vs equal-subscription-weighted policy metrics (subscriptions with >= min test VMs)
_, inv, cnt = np.unique(subs, return_inverse=True, return_counts=True)
keep = cnt[inv] >= args.min_sub_vms
wq = em.equal_group_weights(subs[keep])
wrows = []
for b in BETAS:
    for wn, w in (("VM-weighted", None), ("equal-subscription-weighted", wq)):
        mf = em.policy_metrics(POL["Best fixed (val)"][b][keep], qt[keep], b, w)
        for p in DEPLOY[1:]:
            m = em.policy_metrics(POL[p][b][keep], qt[keep], b, w)
            wrows.append({"beta": b, "weighting": wn, "policy": p, "cost": m["cost"],
                          "% saving vs best fixed": 100 * (mf["cost"] - m["cost"]) / mf["cost"], "E[r] %": 100 * m["E[r]"],
                          "P(Y>r) %": 100 * m["P(Y>r)"], "E[(Y-r)+] pp": 100 * m["E[sf]"], "P(Y-r>20pp) %": 100 * m["P(sf>20pp)"]})
WQ = pd.DataFrame(wrows)
WQ.to_csv(OUT / "ext_weighting_comparison.csv", index=False)
to_md(WQ.pivot_table(index="policy", columns=["beta", "weighting"], values="% saving vs best fixed", sort=False),
      OUT / "ext_weighting_comparison_saving", floatfmt=".1f")
to_md(WQ[WQ.beta == fl.MAIN_BETA].drop(columns="beta").set_index(["weighting", "policy"]), OUT / "ext_weighting_comparison_b5",
      floatfmt=".2f")
# the same weighting comparison on the VALIDATION period (legitimate for any model choice; test above is retrospective)
PAv = pd.read_parquet(fs.FINAL_DATA / "policy_actions_val.parquet")
E2V = pickle.load(open(PROC / "sensitivity" / "e2_policies.pkl", "rb"))["POLV_tier_index"]
check(np.array_equal(PAv.index.to_numpy(), iva), "saved validation actions aligned with strict validation rows")
qv, subs_v = q[iva], F.key_sub.to_numpy()[iva]
Xbv = np.load(fs.FINAL_DATA / "Xb_opt.npy")[iva]
_, inv_v, cnt_v = np.unique(subs_v, return_inverse=True, return_counts=True)
keep_v = cnt_v[inv_v] >= args.min_sub_vms
wv = em.equal_group_weights(subs_v[keep_v])
vrows = []
for b in BETAS:
    tv = lambda tjs: fl.TIERS[fl.predict_tree(fl.tree_from_json(tjs, BN), Xbv)]
    r4 = min([r for r in RUNS if r["beta"] == b and r["bins"] == 32 and r["status"].startswith("OPTIMAL")], key=lambda x: x["depth4"]["val_cost"])
    VPOL = {"Best fixed (val)": PAv[f"Best fixed (val)|b={b}"].to_numpy(float), "RF point + margin": PAv[f"RF predict + margin|b={b}"].to_numpy(float),
            "Greedy depth 3": PAv[f"Greedy [full]|b={b}"].to_numpy(float), "OPT depth 3": PAv[f"OPT-exact [full]|b={b}"].to_numpy(float),
            "OPT 4-leaf (depth 2)": tv(json.loads((SENS / f"s11_e5_opt_depth2_b{b}.txt").read_text())),
            "OPT leaf-penalized (val-selected)": tv(json.loads((SENS / f"s11_e6_selected_b{b}_simple.json").read_text())),
            "OPT depth 4 (certified)": tv(r4["depth4"]["tree"]), "Cost-aware RF": PAv[f"Cost-aware RF|b={b}"].to_numpy(float),
            "Forest-weighted prescription": fl.TIERS[E2V[("Forest-weighted prescription", float(b))]]}
    check(np.isclose(np.mean(VPOL["OPT depth 4 (certified)"] + b * np.maximum(0, qv - VPOL["OPT depth 4 (certified)"])),
                     r4["depth4"]["val_cost"], rtol=1e-12), f"depth-4 validation cost reproduced b={b}")
    for wn, w in (("VM-weighted", None), ("equal-subscription-weighted", wv)):
        mf = em.policy_metrics(VPOL["Best fixed (val)"][keep_v], qv[keep_v], b, w)
        for p, a in VPOL.items():
            if p == "Best fixed (val)":
                continue
            m = em.policy_metrics(a[keep_v], qv[keep_v], b, w)
            vrows.append({"beta": b, "weighting": wn, "policy": p, "val cost": m["cost"],
                          "% saving vs best fixed": 100 * (mf["cost"] - m["cost"]) / mf["cost"], "P(Y>r) %": 100 * m["P(Y>r)"],
                          "E[(Y-r)+] pp": 100 * m["E[sf]"]})
VQ = pd.DataFrame(vrows)
to_md(VQ.pivot_table(index="policy", columns=["beta", "weighting"], values=["% saving vs best fixed", "P(Y>r) %"], sort=False),
      OUT / "ext_weighting_comparison_validation", floatfmt=".2f")
runlog["subscriptions_val"] = {"val subscriptions": int(len(cnt_v)), f">= {args.min_sub_vms} val VMs": int((cnt_v >= args.min_sub_vms).sum()),
                               "VM share of those": float(keep_v.mean())}
runlog["subscriptions"] = {"test subscriptions": int(len(cnt)), f">= {args.min_sub_vms} test VMs": int((cnt >= args.min_sub_vms).sum()),
                           "VM share of those": float(keep.mean())}
log("D customers done")

# =========================================================================== E. workload segments
life, cores, nh = Ft.post_lifetime_h.to_numpy(), Ft.cores.to_numpy(), Ft.sub_n_hist.to_numpy()
depc = Ft.dep_cold.to_numpy()
SEGS = [("vCPU bucket (creation-time)", f"{int(c)} vCPU" if c < 30 else "> 24 vCPU", cores == c) for c in (2, 4, 8, 24, 30)]
SEGS += [("subscription history (creation-time)", "no completed VMs", nh == 0),
         ("subscription history (creation-time)", "1-49 completed VMs", (nh >= 1) & (nh < 50)),
         ("subscription history (creation-time)", "50-999 completed VMs", (nh >= 50) & (nh < 1000)),
         ("subscription history (creation-time)", ">= 1000 completed VMs", nh >= 1000),
         ("deployment history (creation-time)", "none (dep_cold = 1)", depc == 1),
         ("deployment history (creation-time)", "available (dep_cold = 0)", depc == 0),
         ("lifetime (RETROSPECTIVE)", "<= 5 min", life <= 5 / 60 + 1e-9),
         ("lifetime (RETROSPECTIVE)", "5 min - 1 h", (life > 5 / 60 + 1e-9) & (life <= 1)),
         ("lifetime (RETROSPECTIVE)", "1 h - 1 day", (life > 1) & (life <= 24)),
         ("lifetime (RETROSPECTIVE)", "> 1 day", life > 24),
         ("realized P95 (RETROSPECTIVE)", "Y <= 25%", qt <= 0.25),
         ("realized P95 (RETROSPECTIVE)", "25% < Y <= 50%", (qt > 0.25) & (qt <= 0.5)),
         ("realized P95 (RETROSPECTIVE)", "50% < Y <= 75%", (qt > 0.5) & (qt <= 0.75)),
         ("realized P95 (RETROSPECTIVE)", "Y > 75%", qt > 0.75)]
SEGP = ["Best fixed (val)", "RF point + margin", "Greedy depth 3", "OPT depth 3", "OPT 4-leaf (depth 2)", "Cost-aware RF",
        "Forest-weighted prescription", "Oracle (hindsight)"]
seg = []
for b in BETAS:
    cb = cvec(POL["Best fixed (val)"][b], b)
    for p in SEGP:
        a = POL[p][b]
        cp = cvec(a, b)
        tot = (cb - cp).sum()
        for grp, lab, msk in SEGS:
            sf = np.maximum(0, qt[msk] - a[msk])
            seg.append({"beta": b, "group": grp, "segment": lab, "policy": p, "n": int(msk.sum()), "% of test VMs": 100 * msk.mean(),
                        "subscriptions": len(np.unique(subs[msk])), "mean realized P95 %": 100 * qt[msk].mean(),
                        "E[r] %": 100 * a[msk].mean(), "P(Y>r) %": 100 * np.mean(sf > em.EPS), "E[(Y-r)+] pp": 100 * sf.mean(),
                        "P(Y-r>20pp) %": 100 * np.mean(sf > 0.2 + em.EPS), "cost": cp[msk].mean(), "baseline cost": cb[msk].mean(),
                        "% saving vs best fixed": 100 * (cb[msk].mean() - cp[msk].mean()) / cb[msk].mean(),
                        "share of policy's total saving %": 100 * (cb[msk] - cp[msk]).sum() / tot if abs(tot) > 0 else np.nan})
SG = pd.DataFrame(seg)
SG.to_csv(OUT / "ext_segments.csv", index=False)
for b in BETAS:
    s = SG[SG.beta == b]
    sv = s.pivot_table(index=["group", "segment"], columns="policy", values="% saving vs best fixed", sort=False)[SEGP[1:]]
    sv.insert(0, "n", s[s.policy == "OPT depth 3"].set_index(["group", "segment"])["n"])
    sv.insert(1, "mean realized P95 %", s[s.policy == "OPT depth 3"].set_index(["group", "segment"])["mean realized P95 %"])
    to_md(sv, OUT / f"ext_segments_saving_b{b}", floatfmt=".1f")
    rk = s[s.policy.isin(["Best fixed (val)", "RF point + margin", "OPT depth 3", "Cost-aware RF", "Forest-weighted prescription"])]
    rk = rk.pivot_table(index=["group", "segment"], columns="policy", values=["E[r] %", "P(Y>r) %", "E[(Y-r)+] pp"], sort=False)
    to_md(rk, OUT / f"ext_segments_risk_b{b}", floatfmt=".1f")
log("E segments done")

# =========================================================================== F. predictive performance
LOG_COLS = ["dep_batch_size", "sub_age_days", "sub_n_prior_created", "sub_n_hist", "dep_n_prior_created", "dep_n_hist"]
X = F[FEATS_C].astype(float).copy()       # identical design matrix to stage 6
for c_ in LOG_COLS:
    X[c_] = np.log1p(X[c_])
X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * F.hour / 24), np.cos(2 * np.pi * F.hour / 24)
size = F.cores.astype(int).astype(str) + "c_" + F.memory_gb.astype(int).astype(str) + "g"
X = pd.concat([X.drop(columns=["hour", "dow", "cores", "memory_gb", "mem_per_core"]),
               pd.get_dummies(size, prefix="size", drop_first=True, dtype=float),
               pd.get_dummies(F.dow, prefix="dow", drop_first=True, dtype=float)], axis=1)
mu, sd = X.iloc[itr].mean(), X.iloc[itr].std().replace(0, 1)
XR = ((X - mu) / sd).to_numpy()
del X
y = F.y_p95.to_numpy()
t0 = time.time()
rm = Ridge(alpha=1.0).fit(XR[itr], y[itr])
PRED = {"Ridge": (rm.predict(XR[iva]), rm.predict(XR[ite]))}
del XR
Xt = F[FEATS_C].to_numpy(np.float32)
cm = DecisionTreeRegressor(max_leaf_nodes=32, min_samples_leaf=500, random_state=fs.SEED).fit(Xt[itr], y[itr])
PRED["CART"] = (cm.predict(Xt[iva]), cm.predict(Xt[ite]))
del Xt
runlog["ridge_cart_refit_s"] = time.time() - t0
npz = np.load(PROC / "sensitivity" / "forest_msl50_preds.npz")
PRED["RF"] = (npz["pred_val"], npz["pred_test"])
PQ = pd.read_csv(RES / "final_prediction_quality.csv", index_col=0)
for nm, (pv, pt) in PRED.items():
    check(np.isclose(1 - mean_squared_error(y[ite], pt) / y[ite].var(), PQ.loc[nm, "test R2"], rtol=1e-9)
          and np.isclose(1 - mean_squared_error(y[iva], pv) / y[iva].var(), PQ.loc[nm, "val R2"], rtol=1e-9),
          f"{nm} predictions reproduce stage-6 R2")
    for b in BETAS:
        check(np.array_equal(fl.route(pt), PA[f"{nm} predict -> tier|b={b}"].to_numpy(float)), f"{nm} predictions reproduce saved tier actions b={b}")
yt, yv = y[ite], y[iva]
pr = []
for nm, (pv, pt) in PRED.items():
    for split, yy, pp in (("val", yv, pv), ("test", yt, pt)):
        e = yy - pp                                   # positive = under-prediction (realized above predicted)
        pr.append({"model": nm, "split": split, "R2": 1 - np.mean(e ** 2) / yy.var(), "RMSE (pp)": np.sqrt(np.mean(e ** 2)),
                   "MAE (pp)": np.mean(np.abs(e)), "bias: mean(pred - Y) (pp)": -e.mean(), "median(Y - pred) (pp)": np.median(e),
                   "% under-predicted (Y > pred)": 100 * np.mean(e > 0), "P(Y - pred > 10pp) %": 100 * np.mean(e > 10),
                   "P(Y - pred > 25pp) %": 100 * np.mean(e > 25), "P(Y - pred > 50pp) %": 100 * np.mean(e > 50),
                   "P95 of (Y - pred) (pp)": np.percentile(e, 95), "P99 of (Y - pred) (pp)": np.percentile(e, 99)})
PRQ = pd.DataFrame(pr)
to_md(PRQ.set_index(["model", "split"]), OUT / "ext_prediction_quality", floatfmt=".3f")

# predictive accuracy vs downstream decision cost (test)
pdc = []
for nm in ("Ridge", "CART", "RF"):
    r = PRQ[(PRQ.model == nm) & (PRQ.split == "test")].iloc[0]
    for rule, col in (("point -> tier", f"{nm} predict -> tier"), ("point + val-tuned margin", f"{nm} predict + margin")):
        row = {"prediction model": nm, "decision rule": rule, "test R2": r["R2"], "test RMSE (pp)": r["RMSE (pp)"]}
        for b in BETAS:
            cf = cvec(POL["Best fixed (val)"][b], b).mean()
            row[f"% saving b={b}"] = 100 * (cf - cvec(PA[f"{col}|b={b}"].to_numpy(float), b).mean()) / cf
        pdc.append(row)
for rule, nm in (("forest-weighted prescription (same RF)", "Forest-weighted prescription"),
                 ("cost-aware RF (separate forest, shortfall targets)", "Cost-aware RF"), ("OPT depth 3 (no prediction)", "OPT depth 3")):
    row = {"prediction model": "RF" if "same RF" in rule else "-", "decision rule": rule,
           "test R2": PRQ[(PRQ.model == "RF") & (PRQ.split == "test")].R2.iloc[0] if "same RF" in rule else np.nan, "test RMSE (pp)": np.nan}
    for b in BETAS:
        cf = cvec(POL["Best fixed (val)"][b], b).mean()
        row[f"% saving b={b}"] = 100 * (cf - cvec(POL[nm][b], b).mean()) / cf
    pdc.append(row)
to_md(pd.DataFrame(pdc).set_index(["prediction model", "decision rule"]), OUT / "ext_prediction_vs_decision", floatfmt=".3f")

# RF calibration by predicted P95 bin (test)
pt = PRED["RF"][1]
edges = np.arange(0, 101, 10)
bins = np.clip(np.digitize(pt, edges[1:-1], right=False), 0, len(edges) - 2)
cal = []
for k in range(len(edges) - 1):
    mk = bins == k
    if mk.sum() == 0:
        continue
    yy = yt[mk]
    row = {"predicted P95 bin (%)": f"[{edges[k]}, {edges[k + 1]})", "n": int(mk.sum()), "mean predicted": pt[mk].mean(),
           "mean realized": yy.mean(), "realized P10": np.percentile(yy, 10), "realized P50": np.percentile(yy, 50),
           "realized P90": np.percentile(yy, 90), "% Y > pred": 100 * np.mean(yy > pt[mk]),
           "% Y <= 25": 100 * np.mean(yy <= 25), "% 25 < Y <= 50": 100 * np.mean((yy > 25) & (yy <= 50)),
           "% 50 < Y <= 75": 100 * np.mean((yy > 50) & (yy <= 75)), "% Y > 75": 100 * np.mean(yy > 75)}
    for p, lab in (("RF point -> tier", "point->tier"), ("RF point + margin", "margin"), ("Forest-weighted prescription", "forest-weighted")):
        a = POL[p][fl.MAIN_BETA][mk]
        row[f"E[r] % {lab} (b=5)"] = 100 * a.mean()
        row[f"P(Y>r) % {lab} (b=5)"] = 100 * np.mean(qt[mk] > a + em.EPS)
    cal.append(row)
CAL = pd.DataFrame(cal)
to_md(CAL.set_index("predicted P95 bin (%)"), OUT / "ext_rf_calibration", floatfmt=".1f")

# calibration of the shortfall predictions used by the two distribution-aware rules (test)
Shat = {"Cost-aware RF": np.load(fs.FINAL_DATA / "rf_cost_shortfall_preds.npz")["test"], "Forest-weighted prescription": npz["wshort_test"]}
real = np.column_stack([np.maximum(0, qt - t) for t in fl.TIERS[:3]])
sc = []
for nm, S_ in Shat.items():
    for k, t in enumerate(fl.TIERS[:3]):
        e = real[:, k] - S_[:, k]
        sc.append({"model": nm, "tier": f"{int(100 * t)}%", "mean predicted E[(Y-t)+] (pp)": 100 * S_[:, k].mean(),
                   "mean realized (pp)": 100 * real[:, k].mean(), "R2 (per VM)": 1 - np.mean(e ** 2) / real[:, k].var(),
                   "MAE (pp)": 100 * np.mean(np.abs(e))})
    # decision-relevant calibration: 100% vs 75% at beta = 5 hinges on E[(Y-0.75)+] > 0.25 / 5 = 0.05
    dec = pd.qcut(S_[:, 2], 10, labels=False, duplicates="drop")
    for k in np.unique(dec):
        mk = dec == k
        sc.append({"model": nm, "tier": f"75%, predicted-decile {k + 1}", "mean predicted E[(Y-t)+] (pp)": 100 * S_[mk, 2].mean(),
                   "mean realized (pp)": 100 * real[mk, 2].mean(), "n": int(mk.sum())})
to_md(pd.DataFrame(sc).set_index(["model", "tier"]), OUT / "ext_shortfall_prediction_calibration", floatfmt=".3f")

# prediction errors -> reservation mistakes: chosen tier vs the smallest sufficient tier
need = np.searchsorted(fl.TIERS, qt - 1e-12)
te_rows = []
for b in BETAS:
    for p in ["RF point -> tier", "RF point + margin", "Forest-weighted prescription", "Cost-aware RF", "OPT depth 3",
              "Greedy depth 3", "Best fixed (val)", "Oracle (hindsight)"]:
        a = POL[p][b]
        dt = np.argmin(np.abs(a[:, None] - fl.TIERS[None, :]), axis=1) - need
        under1 = dt == -1
        te_rows.append({"beta": b, "policy": p, "% >= 2 tiers below needed": 100 * np.mean(dt <= -2),
                        "% 1 tier below": 100 * np.mean(under1), "% exactly needed tier": 100 * np.mean(dt == 0),
                        "% 1 tier above": 100 * np.mean(dt == 1), "% >= 2 tiers above": 100 * np.mean(dt >= 2),
                        "mean shortfall when 1 tier below (pp)": 100 * np.maximum(0, qt[under1] - a[under1]).mean() if under1.any() else np.nan})
TE = pd.DataFrame(te_rows)
TE.to_csv(OUT / "ext_tier_errors.csv", index=False)
for b in BETAS:
    to_md(TE[TE.beta == b].drop(columns="beta").set_index("policy"), OUT / f"ext_tier_errors_b{b}", floatfmt=".1f")
log("F prediction done")

# =========================================================================== G. regret and disagreement
to_md(OP[OP.policy != "Oracle (hindsight)"].pivot_table(index="policy", columns="beta", sort=False,
                                                          values=["regret", "regret % of oracle cost", "% of attainable saving captured"]),
      OUT / "ext_oracle_regret", floatfmt=".3f")
DIS = ["RF point + margin", "Greedy depth 3", "OPT depth 3", "OPT 4-leaf (depth 2)", "OPT leaf-penalized (val-selected)",
       "OPT depth 4 (certified)", "Cost-aware RF", "Forest-weighted prescription"]
mats = []
for b in BETAS:
    A = np.column_stack([POL[p][b] for p in DIS])
    mats.append(pd.DataFrame([[100 * np.mean(~np.isclose(A[:, i], A[:, j])) for j in range(len(DIS))] for i in range(len(DIS))],
                             index=pd.MultiIndex.from_product([[b], DIS], names=["beta", "policy"]), columns=DIS))
to_md(pd.concat(mats), OUT / "ext_disagreement_matrix", floatfmt=".1f")
PAIRS = [("Cost-aware RF", "OPT depth 3"), ("Forest-weighted prescription", "OPT depth 3"), ("Greedy depth 3", "OPT depth 3"),
         ("OPT 4-leaf (depth 2)", "OPT depth 3"), ("OPT leaf-penalized (val-selected)", "OPT depth 3"),
         ("OPT depth 4 (certified)", "OPT depth 3"), ("Forest-weighted prescription", "Cost-aware RF"),
         ("Forest-weighted prescription", "RF point + margin")]
dp = []
for b in BETAS:
    for p1, p2 in PAIRS:
        dp.append({"beta": b, "policy 1": p1, "policy 2": p2} | em.disagreement(POL[p1][b], POL[p2][b], qt, b, groups=subs))
DP = pd.DataFrame(dp)
DP.to_csv(OUT / "ext_disagreement_pairs.csv", index=False)
for b in BETAS:
    to_md(DP[DP.beta == b].drop(columns="beta").set_index(["policy 1", "policy 2"]).T, OUT / f"ext_disagreement_pairs_b{b}", floatfmt=".3f")
log("G regret / disagreement done")

# =========================================================================== H. paired subscription-cluster bootstrap
COMPS = [("Cost-aware RF", "OPT depth 3", BETAS), ("Forest-weighted prescription", "RF point + margin", BETAS),
         ("OPT 4-leaf (depth 2)", "OPT depth 3", BETAS), ("OPT leaf-penalized (val-selected)", "OPT depth 3", (5,)),
         ("OPT depth 4 (certified)", "OPT depth 3", BETAS)]
METRICS = {"cost": lambda a, b: cvec(a, b), "E[r]": lambda a, b: a, "P(Y>r)": lambda a, b: (qt > a + em.EPS).astype(float),
           "E[(Y-r)+]": lambda a, b: np.maximum(0, qt - a), "P(Y-r>20pp)": lambda a, b: (qt - a > 0.2 + em.EPS).astype(float)}
bt = []
for p1, p2, bs in COMPS:
    for b in bs:
        diffs = {k: f(POL[p1][b], b) - f(POL[p2][b], b) for k, f in METRICS.items()}
        res = em.cluster_bootstrap_multi(diffs, subs, n_boot=args.n_boot, seed=fs.SEED)
        for k, r in res.items():
            bt.append({"beta": b, "comparison (A - B)": f"{p1} - {p2}", "metric": k, "A": float(METRICS[k](POL[p1][b], b).mean()),
                       "B": float(METRICS[k](POL[p2][b], b).mean()), "A - B": r["mean diff"], "95% CI low": r["ci low"],
                       "95% CI high": r["ci high"], "CI excludes 0": bool(r["ci low"] > 0 or r["ci high"] < 0),
                       "new (not previously reported)": k != "cost"})
BT = pd.DataFrame(bt)
# cost CIs must equal the previously reported ones (same draws, seed 0)
old = pd.read_csv(RES / "final_bootstrap.csv")
hb = pd.read_csv(RES / "hw6" / "hw6_forest_bootstrap.csv")
e7 = pd.read_csv(SENS / "s11_e7_selected_vs_existing.csv")
if args.n_boot == 1000:
    for b in BETAS:
        r = BT[(BT.beta == b) & (BT["comparison (A - B)"] == "Cost-aware RF - OPT depth 3") & (BT.metric == "cost")].iloc[0]
        o = old[(old.beta == b) & (old["policy A"] == "OPT-exact [full]") & (old["policy B"] == "Cost-aware RF")].iloc[0]
        check(np.allclose([r["A - B"], r["95% CI low"], r["95% CI high"]], [-o["A - B"], -o["95% CI high"], -o["95% CI low"]], atol=1e-12),
              f"bootstrap reproduces final_bootstrap OPT vs RF b={b}")
        r = BT[(BT.beta == b) & (BT["comparison (A - B)"] == "Forest-weighted prescription - RF point + margin") & (BT.metric == "cost")].iloc[0]
        o = hb[(hb.beta == b) & (hb.A == "C: forest-weighted prescription [main forest, msl=50]")
               & (hb.B == "B: point prediction + margin [main forest, msl=50]")].iloc[0]
        check(np.allclose([r["A - B"], r["95% CI low"], r["95% CI high"]], [o["A - B"], o["95% CI low"], o["95% CI high"]], atol=1e-12),
              f"bootstrap reproduces hw6_forest_bootstrap C vs B b={b}")
        r = BT[(BT.beta == b) & (BT["comparison (A - B)"] == "OPT 4-leaf (depth 2) - OPT depth 3") & (BT.metric == "cost")].iloc[0]
        check(np.allclose([r["A - B"], r["95% CI low"], r["95% CI high"]],
                          E5.loc[b, ["4-leaf minus depth-3 (test)", "95% CI low", "95% CI high"]].to_numpy(float), atol=1e-12),
              f"bootstrap reproduces s11_e5 4-leaf b={b}")
        r = BT[(BT.beta == b) & (BT["comparison (A - B)"] == "OPT depth 4 (certified) - OPT depth 3") & (BT.metric == "cost")].iloc[0]
        o = e7[(e7.beta == b) & (e7.comparator == "OPT-exact [full]")].iloc[0]
        check(np.allclose([r["A - B"], r["95% CI low"], r["95% CI high"]], [o["depth4 - comparator"], o["95% CI low"], o["95% CI high"]], atol=1e-12),
              f"bootstrap reproduces s11_e7 depth-4 b={b}")
    r = BT[(BT.beta == 5) & (BT["comparison (A - B)"] == "OPT leaf-penalized (val-selected) - OPT depth 3") & (BT.metric == "cost")].iloc[0]
    e6 = E6[(E6.beta == 5) & E6["selection rule"].str.startswith("simplest")].iloc[0]
    check(np.allclose([r["A - B"], r["95% CI low"], r["95% CI high"]], e6[["test minus original OPT", "95% CI low", "95% CI high"]].to_numpy(float),
                      atol=1e-12), "bootstrap reproduces s11_e6 penalized b=5")
UNIT = {"cost": "cost", "E[r]": "E[r] (pp of VM)", "P(Y>r)": "P(Y>r) (% pts)", "E[(Y-r)+]": "E[(Y-r)+] (pp of VM)",
        "P(Y-r>20pp)": "P(Y-r>20pp) (% pts)"}


def unitize(T):
    T = T.copy()
    scale = T.metric != "cost"
    for c_ in [c for c in ("A", "B", "A - B", "95% CI low", "95% CI high") if c in T.columns]:
        T.loc[scale, c_] = 100 * T.loc[scale, c_]
    T["metric"] = T.metric.map(UNIT)
    return T


to_md(unitize(BT).set_index(["comparison (A - B)", "beta", "metric"]), OUT / "ext_bootstrap", floatfmt=".4f")  # csv = md units

# equal-subscription weighting (subscriptions with >= min test VMs), beta = 5 only: every subscription counts once.
# One row per subscription (its mean difference), clusters = subscriptions -> bootstrap of the equal-weighted mean.
EQ = [("Cost-aware RF", "OPT depth 3"), ("Greedy depth 3", "OPT depth 3"), ("OPT 4-leaf (depth 2)", "OPT depth 3"),
      ("Forest-weighted prescription", "RF point + margin")]
beq = []
b = fl.MAIN_BETA
for p1, p2 in EQ:
    diffs = {}
    for k in ("cost", "P(Y>r)", "E[(Y-r)+]"):
        g_, s_ = em.group_sums(subs[keep], d=(METRICS[k](POL[p1][b], b) - METRICS[k](POL[p2][b], b))[keep])
        diffs[k] = s_["d"] / s_["n"]
    res = em.cluster_bootstrap_multi(diffs, g_, n_boot=args.n_boot, seed=fs.SEED)
    for k, r in res.items():
        beq.append({"beta": b, "weighting": f"equal-subscription (subs with >= {args.min_sub_vms} test VMs, n = {len(g_)})",
                    "comparison (A - B)": f"{p1} - {p2}", "metric": k, "A - B": r["mean diff"], "95% CI low": r["ci low"],
                    "95% CI high": r["ci high"], "CI excludes 0": bool(r["ci low"] > 0 or r["ci high"] < 0)})
to_md(unitize(pd.DataFrame(beq)).set_index(["comparison (A - B)", "metric"]), OUT / "ext_bootstrap_equal_subscription", floatfmt=".4f")
log("H bootstrap done")

# =========================================================================== figures
COL = {"OPT depth 3": C["blue"], "Greedy depth 3": "#86b6ef", "Cost-aware RF": C["aqua"], "Forest-weighted prescription": C["green"],
       "RF point + margin": C["orange"], "H1 customer mean + margin": C["violet"], "RF point -> tier": C["red"],
       "OPT 4-leaf (depth 2)": C["magenta"], "OPT depth 4 (certified)": C["ink"], "OPT leaf-penalized (val-selected)": C["yellow"],
       "Oracle (hindsight)": C["ink"], "Best fixed (val)": C["ink2"], "Fixed tier": C["ink2"]}
MK = {2: "s", 5: "o", 10: "^"}

# Fig 1: reservation vs risk frontiers + cost vs exceedance
fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.6))
panels = [("E[sf]", "expected shortfall E[(Y-r)+] (pp of VM capacity)"), ("P(Y>r)", "exceedance frequency P(Y > r) (% of VMs)"),
          ("P(sf>20pp)", "severe shortfall P(Y - r > 20 pp) (% of VMs)")]
for ax, (risk, xl) in zip(axes.ravel()[:3], panels):
    xmax = 0
    for nm in FRM:
        g = FR[FR.method == nm].sort_values("beta")
        ax.plot(100 * g[risk], 100 * g["E[r]"], color=COL[nm], lw=1.4, marker=".", ms=4, label=f"{nm} (beta grid 1.5-20)")
        xmax = max(xmax, 100 * g[risk].max())
        for b in BETAS:
            r = g[g.beta == b].iloc[0]
            ax.plot(100 * r[risk], 100 * r["E[r]"], marker=MK[b], ms=7, color=COL[nm], mec="white", mew=0.6, ls="none")
    for nm in ("OPT 4-leaf (depth 2)", "OPT depth 4 (certified)"):
        for b in BETAS:
            r = OP[(OP.beta == b) & (OP.policy == nm)].iloc[0]
            ax.plot(100 * r[risk], 100 * r["E[r]"], marker=MK[b], ms=7, mfc="none", mec=COL[nm], mew=1.3, ls="none")
    h = em.lower_hull(FR[FR.method == "Fixed tier"][risk], FR[FR.method == "Fixed tier"]["E[r]"])
    ax.plot(100 * h[:, 0], 100 * h[:, 1], color=C["ink2"], ls="--", lw=1.1, label="fixed tiers and their random mixtures")
    o = OP[(OP.beta == 5) & (OP.policy == "Oracle (hindsight)")].iloc[0]
    ax.plot(100 * o[risk], 100 * o["E[r]"], marker="*", ms=12, color=C["ink"], ls="none", label="oracle, beta = 5 (hindsight)")
    ax.set_xlim(-0.02 * xmax * 1.05, xmax * 1.05)
    ax.set_ylim(35, 101)
    ax.set_xlabel(xl)
    ax.set_ylabel("mean reservation E[r] (% of VM)")
ins = axes[0, 0].inset_axes([0.42, 0.42, 0.55, 0.52])          # zoom on the beta = 5 configurations
for nm in FRM:
    g = FR[FR.method == nm].sort_values("beta")
    ins.plot(100 * g["E[sf]"], 100 * g["E[r]"], color=COL[nm], lw=1.0)
    r = g[g.beta == 5].iloc[0]
    ins.plot(100 * r["E[sf]"], 100 * r["E[r]"], marker="o", ms=7, color=COL[nm], mec="white", mew=0.6, ls="none")
for nm in ("OPT 4-leaf (depth 2)", "OPT depth 4 (certified)", "OPT leaf-penalized (val-selected)"):
    r = OP[(OP.beta == 5) & (OP.policy == nm)].iloc[0]
    ins.plot(100 * r["E[sf]"], 100 * r["E[r]"], marker="o", ms=7, mfc="none", mec=COL[nm], mew=1.3, ls="none")
ins.set_xlim(0.95, 1.75)
ins.set_ylim(71.5, 75.5)
ins.set_title("zoom: beta = 5 configurations", fontsize=7.5)
ins.tick_params(labelsize=7)
for b, mk in MK.items():
    axes[0, 0].plot([], [], marker=mk, color=C["ink2"], ls="none", label=f"beta = {b}")
axes[0, 0].plot([], [], marker="o", mfc="none", mec=COL["OPT leaf-penalized (val-selected)"], ls="none",
                label="OPT leaf-penalized, beta = 5 (zoom only)")
axes[0, 0].plot([], [], marker="o", mfc="none", mec=COL["OPT 4-leaf (depth 2)"], ls="none", label="OPT 4-leaf (depth 2), beta 2/5/10")
axes[0, 0].plot([], [], marker="o", mfc="none", mec=COL["OPT depth 4 (certified)"], ls="none", label="OPT depth 4 (certified), beta 2/5/10")
axes[0, 0].set_title("(a) reservation vs expected shortfall (down-left is better)", fontsize=10)
axes[0, 1].set_title("(b) reservation vs exceedance frequency", fontsize=10)
axes[1, 0].set_title("(c) reservation vs severe-shortfall frequency", fontsize=10)
ax = axes[1, 1]
d = OP[(OP.beta == 5) & OP.policy.isin(DEPLOY) & ~OP.policy.isin(["Best fixed (val)", "RF point -> tier"])].sort_values("cost")
OFF = {"Cost-aware RF": (8, -10), "Forest-weighted prescription": (8, 2), "OPT depth 4 (certified)": (-8, -12),
       "OPT leaf-penalized (val-selected)": (-8, 6), "OPT depth 3": (8, -4), "Greedy depth 3": (-8, 7),
       "OPT 4-leaf (depth 2)": (8, 4), "RF point + margin": (8, 2), "H1 customer mean + margin": (-8, -12)}
for _, r in d.iterrows():
    ax.plot(100 * r["P(Y>r)"], r.cost, marker="o", ms=8, color=COL.get(r.policy, C["ink2"]), ls="none")
    dx, dy = OFF.get(r.policy, (6, 2))
    ax.annotate(r.policy, (100 * r["P(Y>r)"], r.cost), xytext=(dx, dy), textcoords="offset points", fontsize=7.5,
                color=C["ink2"], ha="left" if dx > 0 else "right")
xs_ = 100 * d["P(Y>r)"].to_numpy()
ax.set_xlim(xs_.min() - 2.2, xs_.max() + 2.2)
ax.text(0.02, 0.30, "off panel: best fixed (always 100%): 0.0% exceedance, cost 1.000\n"
        "RF point -> tier: 29.9% exceedance, cost 0.858", transform=ax.transAxes, fontsize=7.5, va="top", color=C["ink2"])
ax.set_xlabel("exceedance frequency P(Y > r) (% of VMs)")
ax.set_ylabel("mean modeled cost, beta = 5")
ax.set_title("(d) beta = 5: cost vs exceedance frequency (zoomed)", fontsize=10)
h_, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=7.8, bbox_to_anchor=(0.5, -0.07))
fig.suptitle("Reservation vs shortfall risk on the test period (retrospective evaluation; configurations chosen on validation)", fontsize=11.5)
fig.tight_layout()
fig.savefig(FOUT / "ext_fig1_risk_frontier.png")
plt.close(fig)

# Fig 2: customer-level distribution of modeled savings (beta = 5)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.25, 1, 1]})
CP = ["OPT depth 3", "Cost-aware RF", "Forest-weighted prescription", "RF point + margin"]
for p in CP:
    s = persub[p]
    v = np.sort(s[s["test VMs"] >= args.min_sub_vms]["% saving"].to_numpy())
    axes[0].step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=COL[p], lw=1.6, label=p)
axes[0].axvline(0, color=C["ink2"], lw=0.8)
axes[0].set_xlim(-60, 80)
axes[0].set_xlabel("% modeled saving vs best fixed (always 100%), per subscription")
axes[0].set_ylabel("cumulative share of subscriptions")
axes[0].set_title(f"(a) per-subscription savings, subscriptions with >= {args.min_sub_vms} test VMs\n"
                  f"(x clipped to [-60, 80]; n = {int((cnt >= args.min_sub_vms).sum())})", fontsize=9.5)
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
cu5 = CU[(CU.beta == 5) & CU.subscriptions.isin([s_[0] for s_ in STRATA])]
xs = np.arange(len(STRATA))
w = 0.8 / len(CP)
for j, p in enumerate(CP):
    d = cu5[cu5.policy == p].set_index("subscriptions").loc[[s_[0] for s_ in STRATA]]
    x = xs + (j - (len(CP) - 1) / 2) * w
    axes[1].bar(x, d["VM-weighted % saving"], width=w * 0.9, color=COL[p], alpha=0.35)
    axes[1].errorbar(x, d["median % saving"], yerr=[d["median % saving"] - d.P25, d.P75 - d["median % saving"]], fmt="o", ms=4,
                     color=COL[p], lw=1, capsize=2)
    axes[2].bar(x, d["% subs higher cost"], width=w * 0.9, color=COL[p], label=p)
labs = [f"{s_[0]}\n({int(cu5[(cu5.policy == 'OPT depth 3') & (cu5.subscriptions == s_[0])]['n subs'].iloc[0])} subs)" for s_ in STRATA]
for ax in axes[1:]:
    ax.set_xticks(xs, labs, fontsize=7.5)
    ax.set_xlabel("subscription size (test VMs)")
    ax.grid(axis="x", visible=False)
axes[1].axhline(0, color=C["ink2"], lw=0.8)
axes[1].set_ylabel("% saving vs best fixed")
axes[1].set_title("(b) bars: VM-weighted saving; dots: median subscription\n(whiskers: interquartile range)", fontsize=9.5)
axes[2].set_ylabel("% of subscriptions with HIGHER modeled cost")
axes[2].set_title("(c) subscriptions made worse off than always-100%", fontsize=9.5)
fig.suptitle("Customer-level heterogeneity of modeled savings, beta = 5 (test period, retrospective)", fontsize=11)
fig.tight_layout()
fig.savefig(FOUT / "ext_fig2_customer_savings.png")
plt.close(fig)

# Fig 3: shortfall severity (complementary CDF of Y - r)
SP = ["RF point -> tier", "RF point + margin", "H1 customer mean + margin", "Greedy depth 3", "OPT depth 3", "Cost-aware RF",
      "Forest-weighted prescription", "Oracle (hindsight)"]
xg = np.linspace(0, 0.75, 151)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), sharey=True)
for ax, b in zip(axes, BETAS):
    for p in SP + (["Best fixed (val)"] if POL["Best fixed (val)"][b][0] < 1 else []):
        sf = np.sort(np.maximum(0, qt - POL[p][b]))
        ccdf = 1 - np.searchsorted(sf, xg + em.EPS, side="left") / NT
        ccdf = np.where(ccdf > 0, ccdf, np.nan)
        ax.plot(100 * xg, 100 * ccdf, color=COL[p], lw=1.5 if p != "Oracle (hindsight)" else 1.1,
                ls="-" if p != "Oracle (hindsight)" else ":", label=p)
    ax.set_yscale("log")
    ax.set_xlabel("shortfall threshold x (pp of VM capacity)")
    bf = int(round(100 * POL["Best fixed (val)"][b][0]))
    ax.set_title(f"beta = {b}  (best fixed = always {bf}%" + (", no shortfall)" if bf == 100 else ")"), fontsize=10)
axes[0].set_ylabel("% of test VMs with Y - r > x (log scale)")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.1))
fig.suptitle("Shortfall severity: how often, and by how much, realized P95 exceeds the reservation (test, retrospective)", fontsize=11)
fig.tight_layout()
fig.savefig(FOUT / "ext_fig3_shortfall_severity.png")
plt.close(fig)

# Fig 4: RF calibration and the spread of realized P95 within prediction bins
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
ax = axes[0]
mids = CAL["mean predicted"]
ax.fill_between(mids, CAL["realized P10"], CAL["realized P90"], color=C["blue"], alpha=0.15, label="realized P95: 10th-90th percentile")
ax.plot(mids, CAL["mean realized"], color=C["blue"], marker="o", ms=5, label="mean realized P95")
ax.plot(mids, CAL["realized P50"], color=C["blue"], ls="--", lw=1.2, label="median realized P95")
ax.plot([0, 100], [0, 100], color=C["ink2"], lw=0.8, label="perfect calibration")
for t in (25, 50, 75):
    ax.axhline(t, color=C["grid"], lw=1.2, zorder=0)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel("RF predicted P95 (%), mean within 10-point prediction bin")
ax.set_ylabel("realized P95 (%)")
ax.set_title("(a) RF point predictions are calibrated on average,\nbut each bin spans several tiers", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax = axes[1]
xs = np.arange(len(CAL))
bottom = np.zeros(len(CAL))
for col_, lab, colr in (("% Y <= 25", "Y <= 25%", "#cfe1f8"), ("% 25 < Y <= 50", "25-50%", "#86b6ef"),
                        ("% 50 < Y <= 75", "50-75%", "#3987e5"), ("% Y > 75", "> 75%", "#0d366b")):
    ax.bar(xs, CAL[col_], bottom=bottom, color=colr, width=0.8, label=f"realized {lab}")
    bottom += CAL[col_].to_numpy()
ax.plot(xs, CAL["E[r] % point->tier (b=5)"], color=C["red"], marker="o", ms=4, lw=1.3, label="mean reservation: point -> tier")
ax.plot(xs, CAL["E[r] % forest-weighted (b=5)"], color=C["yellow"], marker="D", ms=4, lw=1.3,
        label="mean reservation: forest-weighted (beta = 5)")
ax.set_xticks(xs, [s_.replace("[", "").replace(")", "").replace(", ", "-") for s_ in CAL["predicted P95 bin (%)"]], fontsize=7.5)
ax.set_xlabel("RF predicted P95 bin (%)  [n per bin in ext_rf_calibration]")
ax.set_ylabel("% of VMs in bin (bars) / mean reservation % (lines)")
ax.set_title("(b) realized tier mix within each prediction bin, and the reservations chosen", fontsize=10)
ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.17))
ax.grid(axis="x", visible=False)
fig.suptitle("Why accurate-on-average point predictions are not enough (main RF, test period)", fontsize=11)
fig.tight_layout()
fig.savefig(FOUT / "ext_fig4_rf_calibration.png")
plt.close(fig)

runlog["runtime_s"] = time.time() - t_start
runlog["peak_rss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
(OUT / "ext_runlog.json").write_text(json.dumps(runlog, indent=1))
print(OP[OP.beta == 5].set_index("policy")[["cost", "E[r]", "P(Y>r)", "E[sf]", "P(sf>20pp)", "% of attainable saving captured"]].to_string())
log(f"Stage 12 done ({len(runlog['checks'])} consistency checks passed)")
