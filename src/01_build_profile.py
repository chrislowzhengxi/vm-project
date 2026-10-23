"""Stage 1: load the official VM table, compute dataset statistics, profile columns, make profile plots.

Inputs : data/raw/trace_data_vmtable_vmtable.csv.gz (+ deployments / subscriptions tables)
Outputs: data/processed/vmtable.parquet, results/01_*.{csv,md}, figures/fig1-3*.png
"""
import gzip

import numpy as np
import pandas as pd

from common import C, CORE_MAP, DAY, FIG, MEM_MAP, PROC, RAW, RES, TRACE_END, VM_COLS, plt, to_md

# ---------------------------------------------------------------- load
pq = PROC / "vmtable.parquet"
if pq.exists():
    vm = pd.read_parquet(pq)
else:
    vm = pd.read_csv(RAW / "trace_data_vmtable_vmtable.csv.gz", header=None, names=VM_COLS,
                     dtype={"vmcorecountbucket": str, "vmmemorybucket": str})
    for c in ["vmid", "subscriptionid", "deploymentid", "vmcategory", "vmcorecountbucket", "vmmemorybucket"]:
        vm[c] = vm[c].astype("category")
    vm.to_parquet(pq, index=False)
dep = pd.read_csv(RAW / "trace_data_deployments_deployments.csv.gz", header=None,
                  names=["deploymentid", "deploymentsize"])
sub = pd.read_csv(RAW / "trace_data_subscriptions_subscriptions.csv.gz", header=None,
                  names=["subscriptionid", "first_vm_ts", "count_vms"])

vm["lifetime_h"] = (vm.vmdeleted - vm.vmcreated) / 3600
vm["new_in_trace"] = vm.vmcreated > 0       # created == 0 means already running at trace start (left-censored)

# ---------------------------------------------------------------- Step 1: dataset statistics
files = {
    "trace_data_vmtable_vmtable.csv.gz": vm,
    "trace_data_deployments_deployments.csv.gz": dep,
    "trace_data_subscriptions_subscriptions.csv.gz": sub,
}


def gunzipped_bytes(path):
    n = 0
    with gzip.open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            n += len(chunk)
    return n


rows = []
for fn, d in files.items():
    rows.append({"file": fn, "compressed_MB": (RAW / fn).stat().st_size / 1e6,
                 "uncompressed_MB": gunzipped_bytes(RAW / fn) / 1e6, "rows": len(d),
                 "columns": len(VM_COLS) if d is vm else d.shape[1]})
ft = pd.DataFrame(rows).set_index("file")

stats = pd.Series({
    "VMs (rows)": len(vm),
    "columns": len(VM_COLS),
    "unique subscriptions": vm.subscriptionid.nunique(),
    "unique deployments": vm.deploymentid.nunique(),
    "trace start (s)": int(vm.vmcreated.min()),
    "trace end (s)": int(vm.vmdeleted.max()),
    "trace length (days)": vm.vmdeleted.max() / DAY,
    "timestamp resolution (s)": 300,
    "VMs created at t=0 (pre-existing, left-censored)": int((~vm.new_in_trace).sum()),
    "VMs alive at trace end (right-censored)": int((vm.vmdeleted == TRACE_END).sum()),
    "VMs newly created in trace (modelling population)": int(vm.new_in_trace.sum()),
    "subscriptions with >=1 new VM": vm.loc[vm.new_in_trace, "subscriptionid"].nunique(),
    "deployments with >=1 new VM": vm.loc[vm.new_in_trace, "deploymentid"].nunique(),
    "deployment-table rows": len(dep),
    "subscription-table rows": len(sub),
}, name="value").to_frame()
to_md(ft, RES / "01_files", floatfmt=".1f")
to_md(stats, RES / "01_dataset_stats", floatfmt=",.1f")

# ---------------------------------------------------------------- Step 2: column profile
creation_time = {
    "vmid": "ID only (not a feature)", "subscriptionid": "yes - grouping key only", "deploymentid": "yes - grouping key only",
    "vmcreated": "yes", "vmdeleted": "NO (future)", "maxcpu": "NO (target-like, post-run)",
    "avgcpu": "NO (post-run)", "p95maxcpu": "NO (TARGET)", "vmcategory": "NO (assigned from >=3 days of utilization)",
    "vmcorecountbucket": "yes (requested size)", "vmmemorybucket": "yes (requested size)",
}
prof = []
for c in VM_COLS:
    s = vm[c]
    ex = s.dropna().astype(str).unique()[:3]
    prof.append({"column": c, "dtype": str(s.dtype), "missing_%": 100 * s.isna().mean(),
                 "n_unique": s.nunique(), "examples": ", ".join(e[:14] + ("..." if len(e) > 14 else "") for e in ex),
                 "known_at_creation": creation_time[c]})
prof = pd.DataFrame(prof).set_index("column")
to_md(prof, RES / "01_column_profile", floatfmt=".2f")

# Numeric distributions (all VMs and new-in-trace VMs)
q = [0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
num = vm.loc[vm.new_in_trace, ["p95maxcpu", "avgcpu", "maxcpu", "lifetime_h"]].describe(percentiles=q).T
to_md(num, RES / "01_numeric_distributions", floatfmt=".2f")

# Bucket tables with mean target
def bucket_tab(col, order):
    g = vm[vm.new_in_trace].groupby(col, observed=True).p95maxcpu
    t = pd.DataFrame({"n": g.size(), "share_%": 100 * g.size() / vm.new_in_trace.sum(),
                      "mean_p95": g.mean(), "median_p95": g.median(),
                      "share_p95<=25_%": 100 * g.apply(lambda s: (s <= 25).mean())})
    return t.loc[[o for o in order if o in t.index]]

bt = pd.concat({"cores": bucket_tab("vmcorecountbucket", list(CORE_MAP)),
                "memory_GB": bucket_tab("vmmemorybucket", list(MEM_MAP))})
to_md(bt, RES / "01_bucket_tables", floatfmt=".1f")
cross = pd.crosstab(vm.vmcorecountbucket, vm.vmmemorybucket).loc[list(CORE_MAP), list(MEM_MAP)]
to_md(cross, RES / "01_core_memory_crosstab", floatfmt=".0f")

# VMs per subscription / deployment
vps = vm.groupby("subscriptionid", observed=True).size()
vpd = vm.groupby("deploymentid", observed=True).size()
grp = pd.DataFrame({"VMs per subscription": vps.describe(percentiles=[.25, .5, .75, .9, .99]),
                    "VMs per deployment": vpd.describe(percentiles=[.25, .5, .75, .9, .99])})
to_md(grp, RES / "01_group_sizes", floatfmt=",.1f")
# Concentration: share of new VMs from top subscriptions
newv = vm[vm.new_in_trace]
top = newv.subscriptionid.value_counts()
conc = pd.Series({f"top {k} subs share of new VMs": top.iloc[:k].sum() / len(newv) for k in (1, 10, 100)}
                 | {"subs with >=5 VMs": int((vps >= 5).sum()), "subs with >=50 VMs": int((vps >= 50).sum()),
                    "share of new VMs in subs with >=50 VMs": newv.subscriptionid.map(vps).ge(50).mean()},
                 name="value").to_frame()
to_md(conc, RES / "01_concentration", floatfmt=".3f")

# Category vs lifetime (leakage evidence)
cat = vm.groupby("vmcategory", observed=True).agg(
    n=("vmid", "size"), min_lifetime_h=("lifetime_h", "min"), median_lifetime_h=("lifetime_h", "median"),
    share_lifetime_ge_72h=("lifetime_h", lambda s: (s >= 72).mean()), mean_p95=("p95maxcpu", "mean"))
to_md(cat, RES / "01_vmcategory_vs_lifetime", floatfmt=".2f")

# Deployment table check: is deploymentsize a creation-time quantity?
cnt = vm.groupby("deploymentid", observed=True).size().rename("vms_in_vmtable")
dchk = dep.merge(cnt, left_on="deploymentid", right_index=True)
dep_check = pd.Series({
    "deployments": len(dchk),
    "share where deploymentsize == #VMs in vmtable": (dchk.deploymentsize == dchk.vms_in_vmtable).mean(),
    "mean deploymentsize": dchk.deploymentsize.mean(), "mean #VMs in vmtable": dchk.vms_in_vmtable.mean(),
    "corr(deploymentsize, #VMs)": np.corrcoef(dchk.deploymentsize, dchk.vms_in_vmtable)[0, 1],
}, name="value").to_frame()
to_md(dep_check, RES / "01_deploymentsize_check", floatfmt=".3f")

# Target concentration
p = newv.p95maxcpu
tgt = pd.Series({
    "share p95 < 1%": (p < 1).mean(), "share p95 >= 99%": (p >= 99).mean(), "share p95 == 100%": (p == 100).mean(),
    "share p95 <= 25%": (p <= 25).mean(), "share 25 < p95 <= 50%": ((p > 25) & (p <= 50)).mean(),
    "share 50 < p95 <= 75%": ((p > 50) & (p <= 75)).mean(), "share p95 > 75%": (p > 75).mean(),
    "distinct p95 values": p.nunique(), "share in 10 most common values": p.value_counts().iloc[:10].sum() / len(p),
    "share VMs with lifetime <= 5 min": (newv.lifetime_h <= 5 / 60).mean(),
    "share VMs with lifetime < 1 h": (newv.lifetime_h < 1).mean(),
    "share p95 == maxcpu (few readings)": (newv.p95maxcpu == newv.maxcpu).mean(),
}, name="value").to_frame()
to_md(tgt, RES / "01_target_concentration", floatfmt=".3f")

# ---------------------------------------------------------------- plots (3 only)
# Fig 1: target distributions
fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), sharey=True)
bins = np.linspace(0, 100, 51)
for ax, col, color in zip(axes, ["p95maxcpu", "avgcpu", "maxcpu"], [C["blue"], C["orange"], C["aqua"]]):
    w = np.full(len(newv), 100 / len(newv))
    ax.hist(newv[col], bins=bins, weights=w, color=color, edgecolor=C["surface"], linewidth=0.5)
    ax.set_title(col + ("  (target)" if col == "p95maxcpu" else ""))
    ax.set_xlabel("CPU utilization (% of VM's vCPUs)")
    for t in (25, 50, 75):
        ax.axvline(t, color=C["ink2"], lw=0.8, ls=":")
axes[0].set_ylabel("% of new VMs (2-pt bins)")
fig.suptitle("CPU utilization summaries of VMs created during the trace (n = %s)" % f"{len(newv):,}", y=1.03)
fig.savefig(FIG / "fig1_target_distributions.png")
plt.close(fig)

# Fig 2: VMs per subscription / deployment (CCDF, log-log)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for s, lab, color, xy in [(vps, "subscriptions", C["blue"], (1.5e3, 9e-2)),
                          (vpd, "deployments", C["orange"], (4, 1.5e-3))]:
    x = np.sort(s.values)
    y = 1 - np.arange(len(x)) / len(x)
    ax.loglog(x, y, color=color, drawstyle="steps-post")
    ax.text(*xy, f"{lab} (n={len(s):,})", color=C["ink"])
ax.set_xlabel("VMs in group (log)")
ax.set_ylabel("share of groups with >= x VMs")
ax.set_title("VMs per subscription and per deployment")
fig.savefig(FIG / "fig2_vms_per_group.png")
plt.close(fig)

# Fig 3: creation time with chronological split
hours = newv.vmcreated // 3600
hc = hours.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 2.8))
ax.plot(hc.index / 24, hc.values, color=C["blue"], lw=1)
for d, lab in [(21, "train | val"), (25, "val | test")]:
    ax.axvline(d, color=C["ink2"], lw=1, ls="--")
    ax.text(d + 0.15, hc.max() * 0.95, lab, color=C["ink2"])
ax.set_xlabel("day in trace")
ax.set_ylabel("VMs created per hour")
ax.set_title("VM creations over the 30-day trace (pre-existing VMs at t=0 excluded)")
fig.savefig(FIG / "fig3_creation_time.png")
plt.close(fig)

print(stats.to_string())
print(tgt.to_string())
print("Stage 1 done.")
