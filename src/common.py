"""Shared paths, constants and small helpers for the Azure V2 right-sizing feasibility study."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = ROOT / "figures"
for d in (PROC, RES, FIG):
    d.mkdir(parents=True, exist_ok=True)

VM_COLS = ["vmid", "subscriptionid", "deploymentid", "vmcreated", "vmdeleted",
           "maxcpu", "avgcpu", "p95maxcpu", "vmcategory", "vmcorecountbucket", "vmmemorybucket"]
TRACE_END = 2_591_400          # last timestamp in the trace (30 days, 5-minute grid)
DAY = 86_400
SEED = 0

# Bucket labels -> numeric value used as an ordinal feature (">24" / ">64" mapped like the official notebook).
CORE_MAP = {"2": 2, "4": 4, "8": 8, "24": 24, ">24": 30}
MEM_MAP = {"2": 2, "4": 4, "8": 8, "32": 32, "64": 64, ">64": 70}

TIERS = np.array([0.25, 0.50, 0.75, 1.00])
LAMBDAS = [1, 2, 5, 10]

# Plot styling: reference categorical palette, fixed order; recessive axes.
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "surface": "#fcfcfb"}
plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
    "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "text.color": C["ink"],
    "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.grid": True, "grid.color": C["grid"],
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.labelsize": 10, "font.size": 9, "lines.linewidth": 2,
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def to_md(df, path, floatfmt=".3f"):
    """Write a DataFrame as a CSV plus a Markdown table next to it; return the markdown string."""
    df.to_csv(Path(path).with_suffix(".csv"))
    md = df.to_markdown(floatfmt=floatfmt)
    Path(path).with_suffix(".md").write_text(md + "\n")
    return md


# ---------------------------------------------------------------- feature sets (see 02_features.py for definitions)
FEATS_A = ["cores", "memory_gb", "mem_per_core", "hour", "dow", "day_in_trace", "dep_batch_size"]
SUB_META = ["sub_age_days", "sub_preexisting", "sub_n_prior_created"]
SUB_STATS = ["mean", "median", "std", "last", "last5_mean", "frac_le25", "frac_le50", "frac_le75"]
FEATS_B = FEATS_A + SUB_META + ["sub_n_hist"] + [f"sub_hist_{s}" for s in SUB_STATS] + ["sub_cold"]
FEATS_C = FEATS_B + ["dep_n_prior_created", "dep_n_hist", "dep_hist_mean", "dep_hist_median", "dep_hist_last", "dep_cold"]
# Leakage-sensitivity set: subscription history built from final p95 of VMs that may still be running.
FEATS_B_LEAKY = FEATS_A + SUB_META + ["LEAKY_sub_n"] + [f"LEAKY_sub_{s}" for s in SUB_STATS]


def load_features():
    import pandas as pd
    return pd.read_parquet(PROC / "features.parquet")


def impute(F):
    """Fill missing history statistics using TRAINING rows only.

    * Subscription stats with no completed history -> training-set statistic of the target
      (mean / median / last -> mean p95; std -> median observed std; frac_le* -> training share below threshold).
    * Deployment stats with no completed history -> the same VM's subscription statistic (hierarchical fallback),
      which is itself strictly historical.
    The *_cold indicator columns keep the information that a value was imputed.
    """
    F = F.copy()
    tr = F.split == "train"
    ytr = F.loc[tr, "y_p95"]
    fill = {"mean": ytr.mean(), "median": ytr.median(), "last": ytr.mean(), "last5_mean": ytr.mean(),
            "std": F.loc[tr, "sub_hist_std"].median(),
            "frac_le25": (ytr <= 25).mean(), "frac_le50": (ytr <= 50).mean(), "frac_le75": (ytr <= 75).mean()}
    for s, v in fill.items():
        F[f"sub_hist_{s}"] = F[f"sub_hist_{s}"].fillna(v)
        F[f"LEAKY_sub_{s}"] = F[f"LEAKY_sub_{s}"].fillna(v)
    for s in ("mean", "median", "last"):
        F[f"dep_hist_{s}"] = F[f"dep_hist_{s}"].fillna(F[f"sub_hist_{s}"])
    return F
