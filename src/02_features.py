"""Stage 2: leakage-safe creation-time features + chronological split.

Leakage rules enforced here
* A prior VM's p95maxcpu summarizes its WHOLE lifetime, so it is only observable once that VM has been
  deleted. "Strict" history for a VM created at time c uses only same-group VMs with vmdeleted < c.
* Counts of previously *created* VMs use vmcreated < c (a creation is observable immediately).
* Strict inequality everywhere: VMs sharing the current VM's 5-minute timestamp never contribute.
* "Lenient" history (all VMs created before c, using their final p95) is built ONLY for a leakage-
  sensitivity check; it peeks at the future utilization of still-running VMs and is not a primary feature.
* vmcategory, vmdeleted / lifetime, avgcpu, maxcpu and the deployment table's deploymentsize are never features.

Output: data/processed/features.parquet (one row per VM created during the trace, vmcreated > 0).
"""
import numpy as np
import pandas as pd

from common import CORE_MAP, DAY, MEM_MAP, PROC, RES, to_md

TRAIN_END_DAY, VAL_END_DAY = 21, 25  # train: days [0,21), val: [21,25), test: [25,30)

vm = pd.read_parquet(PROC / "vmtable.parquet")
vm["sub"] = vm.subscriptionid.cat.codes.astype(np.int64)
vm["dep"] = vm.deploymentid.cat.codes.astype(np.int64)
y = vm.p95maxcpu.to_numpy()
c = vm.vmcreated.to_numpy()
d = vm.vmdeleted.to_numpy()


def prefix_history(g_ev, t_ev, x_ev, g_q, t_q, full=True, n_bins=50, last_k=5, exact_upto=200):
    """Stats over events of the same group with event time strictly < query time.

    Events are sorted by (group, time); for each query, searchsorted gives the slice of earlier events in its
    group, and differences of cumulative sums give the statistics in O(n log n) without Python loops.
    """
    T = np.int64(1 << 22)  # > trace length in seconds
    key = g_ev * T + t_ev
    order = np.lexsort((x_ev, key))  # ties broken by value -> deterministic
    key_s, x_s = key[order], x_ev[order]
    pos = np.searchsorted(key_s, g_q * T + t_q, side="left")
    start = np.searchsorted(key_s, g_q * T, side="left")
    k = pos - start
    n = k.astype(float)
    nz = np.where(k > 0, n, np.nan)

    def csum(a):
        return np.concatenate([[0.0], np.cumsum(a, dtype=float)])

    S1 = csum(x_s)
    out = {"n": k, "mean": (S1[pos] - S1[start]) / nz,
           "last": np.where(k > 0, x_s[np.maximum(pos - 1, 0)], np.nan)}
    if not full:
        return out
    S2 = csum(x_s ** 2)
    var = (S2[pos] - S2[start]) / nz - out["mean"] ** 2
    out["std"] = np.where(k >= 2, np.sqrt(np.clip(var * n / np.maximum(n - 1, 1), 0, None)), np.nan)
    lo = np.maximum(pos - last_k, start)
    out[f"last{last_k}_mean"] = np.where(k > 0, (S1[pos] - S1[lo]) / np.maximum(pos - lo, 1), np.nan)
    for thr in (25, 50, 75):
        Ct = csum(x_s <= thr)
        out[f"frac_le{thr}"] = (Ct[pos] - Ct[start]) / nz
    # Median from cumulative histogram counts (2-pt bins, linear interpolation inside the bin).
    w = 100 / n_bins
    b = np.minimum((x_s / w).astype(np.int64), n_bins - 1)
    cnt = np.zeros((len(g_q), n_bins), dtype=np.int32)
    for j in range(n_bins):
        Hj = np.concatenate([[0], np.cumsum(b == j, dtype=np.int64)])
        cnt[:, j] = Hj[pos] - Hj[start]
    cum = np.cumsum(cnt, axis=1)
    half = n / 2
    jstar = np.argmax(cum >= half[:, None], axis=1)
    prev = np.where(jstar > 0, cum[np.arange(len(jstar)), np.maximum(jstar - 1, 0)], 0)
    inbin = cnt[np.arange(len(jstar)), jstar]
    med = (jstar + (half - prev) / np.maximum(inbin, 1)) * w
    # Exact median where the history is short (binning is inaccurate for small, often bimodal samples).
    small = np.flatnonzero((k > 0) & (k <= exact_upto))
    med[small] = [np.median(x_s[s:p]) for s, p in zip(start[small], pos[small])]
    out["median"] = np.where(k > 0, med, np.nan)
    return out


q = vm.vmcreated > 0                      # modelling population: VMs created during the trace
gq_sub, gq_dep, tq = vm.loc[q, "sub"].to_numpy(), vm.loc[q, "dep"].to_numpy(), c[q]
ones = np.ones(len(vm))

F = pd.DataFrame(index=vm.index[q])
# ---- A: raw creation-time features
F["cores"] = vm.loc[q, "vmcorecountbucket"].astype(str).map(CORE_MAP).astype(np.float32)
F["memory_gb"] = vm.loc[q, "vmmemorybucket"].astype(str).map(MEM_MAP).astype(np.float32)
F["mem_per_core"] = F.memory_gb / F.cores
F["hour"] = (tq % DAY) // 3600
F["dow"] = (tq // DAY) % 7                 # relative day of week (absolute calendar date is not published)
F["day_in_trace"] = tq / DAY
# VMs of the same deployment requested in the same 5-minute slot (a multi-VM request); see report caveat.
batch = vm.groupby(["dep", "vmcreated"]).vmid.transform("size")
F["dep_batch_size"] = batch[q].to_numpy()

# ---- B: subscription history (strict)
sub_first = vm.groupby("sub").vmcreated.transform("min")   # min <= c always, so observable at c
F["sub_age_days"] = (tq - sub_first[q].to_numpy()) / DAY
F["sub_preexisting"] = (sub_first[q].to_numpy() == 0).astype(np.int8)  # had VMs running at trace start
F["sub_n_prior_created"] = prefix_history(vm["sub"].to_numpy(), c, ones, gq_sub, tq, full=False)["n"]
h = prefix_history(vm["sub"].to_numpy(), d, y, gq_sub, tq)
F["sub_n_hist"] = h.pop("n")
for k_, v in h.items():
    F[f"sub_hist_{k_}"] = v
F["sub_cold"] = (F.sub_n_hist == 0).astype(np.int8)

# ---- C: deployment history (strict)
F["dep_n_prior_created"] = prefix_history(vm["dep"].to_numpy(), c, ones, gq_dep, tq, full=False)["n"]
h = prefix_history(vm["dep"].to_numpy(), d, y, gq_dep, tq)
F["dep_n_hist"] = h["n"]
F["dep_hist_mean"], F["dep_hist_median"], F["dep_hist_last"] = h["mean"], h["median"], h["last"]
F["dep_cold"] = (F.dep_n_hist == 0).astype(np.int8)

# ---- Leakage-sensitivity only: lenient subscription history (uses final p95 of still-running VMs)
h = prefix_history(vm["sub"].to_numpy(), c, y, gq_sub, tq)
F["LEAKY_sub_n"] = h["n"]
for k_ in ("mean", "median", "std", "last", "last5_mean", "frac_le25", "frac_le50", "frac_le75"):
    F[f"LEAKY_sub_{k_}"] = h[k_]

# ---- Non-feature columns (target, grouping keys, post-deployment info for stress tests only)
F["y_p95"] = y[q]
F["post_avgcpu"] = vm.loc[q, "avgcpu"].to_numpy()
F["post_maxcpu"] = vm.loc[q, "maxcpu"].to_numpy()
F["post_lifetime_h"] = (d[q] - tq) / 3600
F["post_vmdeleted"] = d[q]
F["post_vmcategory"] = vm.loc[q, "vmcategory"].astype(str).to_numpy()
F["key_sub"] = gq_sub
F["key_dep"] = gq_dep
F["vmcreated"] = tq
F["split"] = np.select([tq < TRAIN_END_DAY * DAY, tq < VAL_END_DAY * DAY], ["train", "val"], "test")

F = F.sort_values(["vmcreated", "key_sub", "key_dep"]).reset_index(drop=True)
for col in F.columns:
    if F[col].dtype == np.float64 and col not in ("y_p95", "day_in_trace", "sub_age_days"):
        F[col] = F[col].astype(np.float32)
F.to_parquet(PROC / "features.parquet", index=False)

# ---- Split + history-coverage summary
sp = F.groupby("split", sort=False).agg(
    n=("y_p95", "size"), start_day=("vmcreated", lambda s: s.min() / DAY), end_day=("vmcreated", lambda s: s.max() / DAY),
    mean_p95=("y_p95", "mean"), subs=("key_sub", "nunique"),
    share_sub_cold=("sub_cold", "mean"), share_sub_hist_ge1=("sub_n_hist", lambda s: (s >= 1).mean()),
    share_sub_hist_ge5=("sub_n_hist", lambda s: (s >= 5).mean()), share_sub_hist_ge50=("sub_n_hist", lambda s: (s >= 50).mean()),
    share_brand_new_sub=("sub_n_prior_created", lambda s: (s == 0).mean()),
    share_dep_hist_ge1=("dep_n_hist", lambda s: (s >= 1).mean()),
    share_dep_prior_ge1=("dep_n_prior_created", lambda s: (s >= 1).mean()),
).loc[["train", "val", "test"]]
sp["share_of_rows"] = sp.n / sp.n.sum()
to_md(sp, RES / "02_split_and_history_coverage", floatfmt=".3f")
print(sp.T.to_string())
print(F.filter(regex="^(sub_|dep_)").describe().T[["count", "mean", "50%", "max"]].to_string())
print("rows", len(F), "cols", F.shape[1])
