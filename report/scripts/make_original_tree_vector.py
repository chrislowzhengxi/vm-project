"""Vector, white-background redraw of figures/final_fig4_opt_tree.png (the original stage-6 tree figure), sized for a
6.5 in LaTeX text width. Same content and style as the original: splits/tiers from the saved model JSON, node shares
and mean P95 from applying the saved tree to the saved binned test features. No model is refit.
Output: figures_pdf/final_fig4_opt_tree.pdf (+ a 300-dpi white PNG next to it)."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import final_lib as fl  # noqa: E402
from final_setup import LABELS  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "figure.facecolor": "white",
                     "savefig.facecolor": "white", "axes.facecolor": "white"})
js = json.loads((ROOT / "data/processed/final/binner.json").read_text())
tj = json.loads((ROOT / "results/final_models/OPT-exact_full_b5.json").read_text())
tree = fl.tree_from_json(tj, type("Bn", (), {"names": js["names"]})())
F = pd.read_parquet(ROOT / "data/processed/features.parquet", columns=["split", "y_p95"])
ite = np.flatnonzero((F.split == "test").to_numpy())
Xb = np.load(ROOT / "data/processed/final/Xb_opt.npy")[ite]
qt = F.y_p95.to_numpy()[ite] / 100
N = len(ite)
TIER_COL = {0.25: "#86b6ef", 0.5: "#3987e5", 0.75: "#1c5cab", 1.0: "#0d366b"}
SHORT = {"sub_hist_frac_le75": "share of past VMs with P95 <= 75%", "sub_hist_frac_le50": "share of past VMs with P95 <= 50%",
         "sub_hist_median": "subscription median past P95 (%)", "sub_hist_mean": "subscription mean past P95 (%)",
         "dep_hist_mean": "deployment mean past P95 (%)", "cores": "vCPU bucket"}


def mark(nj, node, idx, d, pos):
    nj["_n"], nj["_q"], nj["_d"] = len(idx), qt[idx].mean(), d
    if "action" in nj:
        nj["_x"] = pos[0]; pos[0] += 1; return
    go = Xb[idx, node["feature"]] <= node["t"]
    mark(nj["left"], node["left"], idx[go], d + 1, pos)
    mark(nj["right"], node["right"], idx[~go], d + 1, pos)
    nj["_x"] = (nj["left"]["_x"] + nj["right"]["_x"]) / 2


mark(tj, tree, np.arange(N), 0, [0])
fig = plt.figure(figsize=(6.5, 2.9))
ax = fig.add_axes((0, 0, 1, 1)); ax.axis("off")
FS = 6.2


def draw(n):
    x, y = n["_x"], -n["_d"]
    if "action" in n:
        a = n["action"]
        ax.text(x, y, f"reserve {int(a * 100)}%\n{100 * n["_n"] / N:.1f}% of test VMs\nmean P95 {100 * n['_q']:.0f}%",
                ha="center", va="center", fontsize=FS - 0.7, color="white" if a >= 0.5 else "#0b0b0b", linespacing=1.25,
                bbox=dict(boxstyle="round,pad=0.32", fc=TIER_COL[a], ec="none"))
        return
    for ch, lab in ((n["left"], "yes"), (n["right"], "no")):
        ax.plot([x, ch["_x"]], [y - 0.2, -ch["_d"] + 0.24], color="#52514e", lw=0.6, zorder=0)
        ax.text((x + ch["_x"]) / 2, (y - ch["_d"]) / 2, lab, fontsize=FS - 0.4, color="#52514e", ha="center",
                va="center", bbox=dict(fc="white", ec="none", pad=0.3))
        draw(ch)
    thr = n["threshold_le"]
    t = f"{thr:.3g}"
    ax.text(x, y, f"{SHORT[n['feature']]}\n<= {t} ?\n{100 * n['_n'] / N:.0f}% of test VMs", ha="center", va="center",
            fontsize=FS, color="#0b0b0b", linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f1f0ec", ec="#52514e", lw=0.6))


draw(tj)
ax.set_xlim(-0.6, 7.6); ax.set_ylim(-3.45, 0.4)
out = ROOT / "figures_pdf"
fig.savefig(out / "final_fig4_opt_tree.pdf")
fig.savefig(ROOT / "figures_pdf_src/_preview/final_fig4_opt_tree.png", dpi=300)
print("ok")
