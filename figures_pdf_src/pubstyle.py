"""Shared publication style for the vector PDF versions of the project figures (same look as report/figures).

Usage (scripts in figures_pdf_src/, run from the repository root):
    from pubstyle import plt, STY, PAL, INK, GREY, GRID, W, save, ROOT, RES, PROC
Rules: white background; LaTeX text in Latin Modern; widths W (6.5 in, full text width) or W/2 (single column);
8 pt labels, 7.5 pt ticks/legends; light horizontal grid; no top/right spines; frameless legends; no figure titles
(captions go in LaTeX), panel titles like "(a) ..." allowed; vector output only (never raster images inside the PDF).
Remember that with usetex, "%", "&", "_" and "#" in any text must be escaped (\\%, \\&, \\_, \\#).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, PROC = ROOT / "results", ROOT / "data" / "processed"
OUT = ROOT / "figures_pdf"
sys.path.insert(0, str(ROOT / "src"))
W = 6.5
INK, GREY, LIGHT, GRID = "#1d1d1b", "#5f5e5a", "#9a9994", "#e3e2de"
PAL = {"blue": "#2a78d6", "lightblue": "#86b6ef", "green": "#1baf7a", "orange": "#eb6834", "red": "#e34948",
       "violet": "#6250d6", "yellow": "#eda100", "magenta": "#e87ba4", "darkgreen": "#008300", "grey": GREY}
plt.rcParams.update({
    "text.usetex": True, "font.family": "serif",
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": GREY,
    "ytick.color": GREY, "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.3, "lines.markersize": 3.6, "legend.frameon": False, "legend.handlelength": 2.2,
    "axes.titlepad": 4, "figure.dpi": 150, "pdf.fonttype": 42, "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.05, "savefig.bbox": None,
})
# one style per method, identical to the report figures
STY = {
    "opt": dict(color=PAL["blue"], ls="-", marker="o"),
    "greedy": dict(color=PAL["lightblue"], ls=(0, (4, 1.5)), marker="s"),
    "rf": dict(color=PAL["green"], ls="-", marker="D"),
    "fw": dict(color=PAL["darkgreen"], ls=(0, (1, 1)), marker="v"),
    "margin": dict(color=PAL["orange"], ls=(0, (5, 1.5, 1, 1.5)), marker="^"),
    "round": dict(color=PAL["red"], ls=(0, (1, 1.2)), marker="x"),
    "h1": dict(color=PAL["violet"], ls="-", marker="P"),
    "iai": dict(color=PAL["yellow"], ls="-", marker="*"),
    "cart": dict(color=PAL["magenta"], ls=(0, (3, 1)), marker="d"),
    "fixed": dict(color=GREY, ls=(0, (3, 2)), marker="s"),
}


def save(fig, relpath):
    """relpath like 'final_fig1_benchmark_b5' or 'hw6/hw6_fig_forest_rules' -> figures_pdf/<relpath>.pdf (+ preview png)."""
    p = OUT / f"{relpath}.pdf"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    prev = ROOT / "figures_pdf_src" / "_preview" / f"{relpath}.png"
    prev.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prev, dpi=160)
    plt.close(fig)
    return p
