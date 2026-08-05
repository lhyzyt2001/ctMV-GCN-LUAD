from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

from config import RESULT_ROOT


OUT_DIR = RESULT_ROOT / "00_workflow"
FIGURE_BASENAME = "Figure_1_ctMV_GCN_workflow"

BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
PURPLE = "#7A5195"
DARK = "#263238"
MID = "#52636B"
LIGHT_LINE = "#B7C2C8"
PALE_BLUE = "#E8F3FA"
PALE_SKY = "#EAF7FB"
PALE_GREEN = "#E8F5EF"
PALE_ORANGE = "#FFF3DD"
PALE_PURPLE = "#F1EAF5"
PALE_GREY = "#F4F6F7"


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "text.color": DARK,
        "axes.edgecolor": DARK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str = "",
    *,
    face: str = "white",
    edge: str = BLUE,
    title_color: str = DARK,
    title_size: float = 6.8,
    body_size: float = 5.45,
    linewidth: float = 0.9,
    radius: float = 0.08,
    zorder: int = 3,
) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.035,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h - 0.16, title,
        ha="center", va="top", fontsize=title_size,
        fontweight="bold", color=title_color, zorder=zorder + 1,
    )
    if body:
        ax.text(
            x + w / 2, y + h - 0.47, body,
            ha="center", va="top", fontsize=body_size,
            color=DARK, linespacing=1.22, zorder=zorder + 1,
        )


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MID,
    width: float = 0.85,
    mutation: float = 8,
    connectionstyle: str = "arc3",
    zorder: int = 2,
) -> None:
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=width,
        color=color,
        shrinkA=1.5,
        shrinkB=1.5,
        connectionstyle=connectionstyle,
        zorder=zorder,
    ))


def section_label(ax, y: float, letter: str, title: str, color: str, line_start: float) -> None:
    ax.text(0.28, y, letter, ha="left", va="center", fontsize=9.2, fontweight="bold", color=color)
    ax.text(0.62, y, title, ha="left", va="center", fontsize=8.0, fontweight="bold", color=DARK)
    ax.plot([line_start, 12.72], [y, y], color=color, linewidth=1.0, alpha=0.65, solid_capstyle="round")


def small_tag(ax, x: float, y: float, text: str, color: str) -> None:
    ax.text(
        x, y, text, ha="center", va="center", fontsize=5.2,
        color="white", fontweight="bold",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": color, "edgecolor": color, "linewidth": 0.5},
        zorder=7,
    )


def make_figure() -> plt.Figure:
    set_style()
    fig, ax = plt.subplots(figsize=(170 / 25.4, 180 / 25.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 13.2)
    ax.axis("off")
    fig.subplots_adjust(left=0.015, right=0.985, top=0.99, bottom=0.015)

    ax.text(
        6.5, 12.94,
        "Robustness-aware ctMV-GCN workflow for LUAD gene prioritization",
        ha="center", va="center", fontsize=10.0, fontweight="bold", color=DARK,
    )

    # A. Data integration
    section_label(ax, 12.48, "A", "Data sources and multiview graph construction", BLUE, 5.95)
    source_y, source_h, derived_y, derived_h = 11.28, 0.85, 10.04, 0.91
    columns = [0.28, 3.45, 6.62, 9.79]
    source_titles = ["TISCH", "STRING v12", "KEGG Medicus", "Open Targets LUAD"]
    source_bodies = [
        "15 aggregated cell-type\nexpression profiles",
        "Functional associations\ncombined score ≥ 700",
        "Gene–pathway\nincidence sets",
        "Clinical-evidence label\nmax trial phase > 0",
    ]
    derived_titles = ["Node features + cell view", "STRING view", "Pathway view", "Binary endpoint"]
    derived_bodies = [
        "Expression matrix; mutual\ncosine 12-nearest-neighbour graph",
        "Weighted, symmetrized\nfunctional-association graph",
        "Size-weighted shared pathways;\ntop-12 sparsification",
        "1,184 positive genes\namong 25,242 nodes (4.69%)",
    ]
    colors = [SKY, BLUE, ORANGE, PURPLE]
    pale = [PALE_SKY, PALE_BLUE, PALE_ORANGE, PALE_PURPLE]
    for x, st, sb, dt, db, color, fill in zip(
        columns, source_titles, source_bodies, derived_titles, derived_bodies, colors, pale
    ):
        box(ax, x, source_y, 2.9, source_h, st, sb, face=fill, edge=color, title_size=6.5, body_size=4.95)
        box(ax, x, derived_y, 2.9, derived_h, dt, db, face="white", edge=color, title_size=6.05, body_size=4.65)
        arrow(ax, (x + 1.45, source_y - 0.02), (x + 1.45, derived_y + derived_h + 0.02), color=color)

    box(
        ax, 2.02, 9.20, 8.96, 0.46,
        "Integrated 25,242-node PyG dataset: 15-dimensional expression matrix + three weighted graph views + clinical-evidence labels",
        face=PALE_GREY, edge=MID, title_size=5.55, linewidth=0.8, radius=0.05,
    )
    for x, color in zip(columns, colors):
        arrow(ax, (x + 1.45, derived_y - 0.02), (6.5, 9.68), color=color, width=0.7, mutation=6)

    # B. Model architecture
    section_label(ax, 8.70, "B", "Primary ctMV-GCN architecture", GREEN, 3.90)
    arrow(ax, (6.5, 9.18), (6.5, 8.46), color=GREEN, width=1.1, mutation=9)
    model_y, model_h = 5.78, 2.58
    ax.add_patch(FancyBboxPatch(
        (0.28, model_y), 12.44, model_h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=0.95, edgecolor=GREEN, facecolor="#FBFDFC", zorder=1,
    ))

    box(
        ax, 0.46, 6.40, 1.56, 1.28,
        "Preprocessing", "Training-fitted\nstandardization\nPCA (5 components)",
        face=PALE_GREEN, edge=GREEN, title_size=5.9, body_size=4.5,
    )
    ax.text(1.24, 6.08, "Fit on training nodes within CV", ha="center", va="center", fontsize=4.35, color=MID)

    branch_x, branch_w, branch_h = 2.42, 2.18, 0.62
    branch_ys = [7.48, 6.76, 6.04]
    branch_names = [
        "STRING GCN branch",
        "Cell-profile GCN branch",
        "Pathway GCN branch",
    ]
    branch_colors = [BLUE, SKY, ORANGE]
    for y, name, color in zip(branch_ys, branch_names, branch_colors):
        box(ax, branch_x, y, branch_w, branch_h, name, "2 weighted layers · 20D", face="white", edge=color, title_size=5.0, body_size=4.05, radius=0.05)
        arrow(ax, (2.04, 7.04), (branch_x - 0.02, y + branch_h / 2), color=color, width=0.7, mutation=6)

    box(
        ax, 4.92, 6.40, 2.26, 1.35,
        "Node-level local attention",
        "MLP on concatenated view embeddings\nNode-specific softmax weights\n(alpha_PPI, alpha_Cell, alpha_Path)",
        face=PALE_GREEN, edge=GREEN, title_size=5.65, body_size=3.95,
    )
    for y, color in zip(branch_ys, branch_colors):
        arrow(ax, (branch_x + branch_w + 0.02, y + branch_h / 2), (4.90, 7.08), color=color, width=0.7, mutation=6)

    box(
        ax, 7.50, 6.40, 1.76, 1.35,
        "Weighted fusion", "alpha_PPI × h_PPI\nalpha_Cell × h_Cell\nalpha_Path × h_Path\nConcatenate (60D)",
        face=PALE_SKY, edge=SKY, title_size=5.65, body_size=3.75,
    )
    arrow(ax, (7.20, 7.08), (7.48, 7.08), color=GREEN)

    box(
        ax, 9.59, 6.40, 1.34, 1.35,
        "Classifier", "Linear (60 → 2)\n+ softmax",
        face=PALE_ORANGE, edge=ORANGE, title_size=5.8, body_size=4.4,
    )
    arrow(ax, (9.28, 7.08), (9.57, 7.08), color=ORANGE)

    box(
        ax, 11.27, 6.40, 1.25, 1.35,
        "Output", "Clinical-evidence\nprobability",
        face=PALE_PURPLE, edge=PURPLE, title_size=5.8, body_size=4.3,
    )
    arrow(ax, (10.95, 7.08), (11.25, 7.08), color=PURPLE)
    small_tag(ax, 10.25, 6.03, "class-weighted cross-entropy", VERMILION)
    ax.text(6.2, 5.98, "ReLU and dropout (p=0.5) within each branch", ha="center", va="center", fontsize=4.4, color=MID)

    # C. Evaluation, robust prioritization and validation
    section_label(ax, 5.26, "C", "Evaluation, robustness-aware prioritization and biological characterization", PURPLE, 10.35)
    arrow(ax, (11.90, 6.38), (11.90, 5.38), color=PURPLE, width=1.0, mutation=8, connectionstyle="arc3,rad=0.0")

    bottom_y, bottom_h = 0.90, 3.83
    box(
        ax, 0.28, bottom_y, 2.80, bottom_h,
        "Transductive evaluation",
        "Repeated 5-fold CV × 3\n15% inner validation\nTest labels hidden; topology visible\n\nRWR, degree, LR/RF,\nGCN/GAT/GraphSAGE\n\nROC-AUC · AUPRC · Top-K",
        face=PALE_PURPLE, edge=PURPLE, title_size=6.2, body_size=4.55,
    )
    box(
        ax, 3.37, bottom_y, 1.82, bottom_h,
        "Final ensemble",
        "10 fixed seeds\n\nMean score + SD\nRank stability\n\nLocal attention\nChannel ablation",
        face=PALE_GREEN, edge=GREEN, title_size=6.1, body_size=4.55,
    )
    arrow(ax, (3.10, 2.23), (3.35, 2.23), color=GREEN)

    box(
        ax, 5.48, bottom_y, 2.77, bottom_h,
        "Degree-aware robustness",
        "Degree-stratified evaluation\n1:1 matched bootstrap\nConditional permutation\nNonlinear residualization\nEdge-dropout sensitivity\n\nPost hoc exclusions:\ntop 5% STRING hubs; structural\nand declared housekeeping genes",
        face=PALE_BLUE, edge=BLUE, title_size=6.1, body_size=4.45,
    )
    arrow(ax, (5.21, 2.23), (5.46, 2.23), color=BLUE)

    box(
        ax, 8.54, 1.35, 1.35, 2.80,
        "Robust top 20",
        "0.50 residual\npercentile\n+ 0.25 top-500\nfrequency\n+ 0.25 rank\nstability\n\nNo external outcome",
        face=PALE_ORANGE, edge=ORANGE, title_size=6.0, body_size=4.25,
    )
    arrow(ax, (8.27, 2.23), (8.52, 2.23), color=ORANGE)

    box(
        ax, 10.18, bottom_y, 2.54, bottom_h,
        "Biological validation",
        "TCGA paired expression, stage,\nOS and new-tumor event\n\nGSEA sensitivity\nGSE68465 expression/survival\nCPTAC tumor protein profile\nDepMap LUAD dependency\n\nBenjamini–Hochberg FDR",
        face=PALE_GREEN, edge=GREEN, title_size=5.9, body_size=4.35,
    )
    arrow(ax, (9.91, 2.23), (10.16, 2.23), color=GREEN)

    ax.text(
        6.5, 0.40,
        "Claim boundary: exploratory, transductive gene prioritization; downstream associations do not establish therapeutic efficacy or causality.",
        ha="center", va="center", fontsize=5.35, color=DARK, fontstyle="italic",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": PALE_GREY, "edgecolor": LIGHT_LINE, "linewidth": 0.6},
    )

    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    base = OUT_DIR / FIGURE_BASENAME
    pdf_path = base.with_suffix(".pdf")
    svg_path = base.with_suffix(".svg")
    png_path = base.with_suffix(".png")
    tiff_path = base.with_suffix(".tiff")
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    with Image.open(png_path) as image:
        image.convert("RGB").save(tiff_path, dpi=(600, 600), compression="tiff_lzw")
    print(f"Workflow figure written to {OUT_DIR}")


if __name__ == "__main__":
    main()
