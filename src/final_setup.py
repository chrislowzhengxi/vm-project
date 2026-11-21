"""Shared data setup for the final-project scripts (05-07).

Primary ("strict") chronological setup:
  train = VMs created in days [0, 21) whose label was observable by the training cutoff (deleted before day 21);
  val   = VMs created in days [21, 25) whose label was observable by the validation cutoff (deleted before day 25);
  test  = ALL VMs created in days [25, 30) (right-censored ones included; sensitivity checks exclude them).
History features (02_features.py) already use only VMs deleted strictly before each VM's creation.
The feasibility setup (all train / all val rows) is kept for comparison via strict=False.
"""
import numpy as np

import final_lib as fl
from common import FEATS_A, FEATS_B, FEATS_C, FIG, PROC, RES, impute, load_features  # noqa: F401

FINAL_DATA = PROC / "final"
FINAL_DATA.mkdir(parents=True, exist_ok=True)
MODELS = RES / "final_models"
MODELS.mkdir(parents=True, exist_ok=True)

OPT_CANDIDATES = list(FEATS_C)     # pool for the compact OPT feature set (selected on validation in 05)
N_OPT_FEATURES = 8
MAX_BINS = 32
MIN_LEAF_FRACS = (0.001, 0.005, 0.01, 0.02, 0.05)
DEPTHS = (1, 2, 3)
SEED = 0

LABELS = {
    "cores": "vCPU bucket", "memory_gb": "memory bucket (GB)", "mem_per_core": "GB per vCPU",
    "hour": "creation hour", "dow": "creation weekday", "day_in_trace": "day in trace",
    "dep_batch_size": "VMs requested in same deployment slot",
    "sub_age_days": "subscription age (days)", "sub_preexisting": "subscription active at trace start",
    "sub_n_prior_created": "# earlier VMs (subscription)", "sub_n_hist": "# completed earlier VMs (subscription)",
    "sub_hist_mean": "subscription mean past P95 (%)", "sub_hist_median": "subscription median past P95 (%)",
    "sub_hist_std": "subscription SD of past P95", "sub_hist_last": "last completed VM's P95 (%)",
    "sub_hist_last5_mean": "mean P95 of last 5 completed VMs (%)",
    "sub_hist_frac_le25": "share of past VMs with P95 <= 25%", "sub_hist_frac_le50": "share of past VMs with P95 <= 50%",
    "sub_hist_frac_le75": "share of past VMs with P95 <= 75%", "sub_cold": "no completed subscription history",
    "dep_n_prior_created": "# earlier VMs (deployment)", "dep_n_hist": "# completed earlier VMs (deployment)",
    "dep_hist_mean": "deployment mean past P95 (%)", "dep_hist_median": "deployment median past P95 (%)",
    "dep_hist_last": "deployment last completed P95 (%)", "dep_cold": "no completed deployment history",
}


def masks_for(F, strict=True):
    tr, va, te = (F.split == "train"), (F.split == "val"), (F.split == "test")
    if strict:
        tr = tr & (F.post_vmdeleted < fl.TRAIN_END)
        va = va & (F.post_vmdeleted < fl.VAL_END)
    return {"train": tr.to_numpy(), "val": va.to_numpy(), "test": te.to_numpy()}


def load_strict(strict=True):
    """Features with training-only imputation (imputation values from the rows actually used for training)."""
    F = load_features()
    m = masks_for(F, strict)
    return impute(F, train_mask=m["train"]), m


def q_of(F):
    return fl.p95_to_fraction(F.y_p95.to_numpy())


def sample_indices(idx, n, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.asarray(idx), size=min(n, len(idx)), replace=False))


def eval_weights(F, mask):
    """Evaluation-only weights: allocated vCPUs (bucket value) and vCPU-hours (lifetime is post hoc, never a feature)."""
    cores = F.loc[mask, "cores"].to_numpy(float)
    life = np.maximum(F.loc[mask, "post_lifetime_h"].to_numpy(float), 5 / 60)
    return {"per VM": None, "vCPU-weighted": cores, "vCPU-hour-weighted": cores * life}
