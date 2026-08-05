from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image

from config import RESULT_ROOT


OUT_DIR = RESULT_ROOT / "00_workflow"
BASE_NAME = "Figure_1_ctMV_GCN_framework_schematic"

INK = "#263238"
GREY = "#4E5A5F"
LIGHT_GREY = "#EEF1F2"
BLUE = "#0072B2"
SKY = "#2B8CBE"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
PURPLE = "#7A5195"
PINK = "#CC79A7"
YELLOW = "#F0E442"


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "text.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def panel(ax, x: float, y: float, w: float, h: float, letter: str, title: str) -> None:
    ax.add_patch(Rectangle(
        (x, y), w, h, fill=False, edgecolor=INK,
        linewidth=0.8, linestyle=(0, (3, 2)), zorder=1,
    ))
    ax.text(x + 0.12, y + h - 0.13, letter, ha="left", va="top", fontsize=9, fontweight="bold")
    ax.text(x + 0.43, y + h - 0.13, title, ha="left", va="top", fontsize=7.2, fontweight="bold")


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = ORANGE,
    width: float = 1.2,
    mutation: float = 10,
    style_name: str = "-|>",
    connection: str = "arc3",
    zorder: int = 4,
) -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style_name, mutation_scale=mutation,
        linewidth=width, color=color, shrinkA=1, shrinkB=1,
        connectionstyle=connection, zorder=zorder,
    ))


def rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    face: str = "white",
    edge: str = GREY,
    fontsize: float = 5.7,
    weight: str = "normal",
    radius: float = 0.06,
    linewidth: float = 0.8,
    zorder: int = 3,
) -> None:
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=linewidth, zorder=zorder,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, linespacing=1.12, zorder=zorder + 1)


def matrix_stack(ax, x: float, y: float, scale: float = 1.0) -> None:
    palette = [PURPLE, GREEN, ORANGE, BLUE]
    for layer, color in enumerate(palette):
        dx = layer * 0.08 * scale
        dy = layer * 0.11 * scale
        w, h = 0.92 * scale, 0.46 * scale
        ax.add_patch(Rectangle((x + dx, y + dy), w, h, facecolor="white", edgecolor=INK, linewidth=0.55, zorder=2 + layer))
        for col in range(5):
            ax.add_patch(Rectangle(
                (x + dx + col * w / 5, y + dy), w / 5, h,
                facecolor=color, edgecolor="white", linewidth=0.25,
                alpha=0.82, zorder=3 + layer,
            ))
    ax.text(x + 0.63 * scale, y - 0.15 * scale, "TISCH profiles", ha="center", va="top", fontsize=5.2, fontweight="bold")
    ax.text(x + 0.63 * scale, y - 0.35 * scale, "25,242 genes × 15 cell types", ha="center", va="top", fontsize=4.3, color=GREY)


def network_icon(
    ax,
    x: float,
    y: float,
    scale: float,
    color: str,
    label: str,
    *,
    highlighted: int | None = None,
) -> None:
    coords = np.array([
        [0.03, 0.50], [0.32, 0.88], [0.68, 0.76],
        [0.92, 0.38], [0.56, 0.08], [0.24, 0.18], [0.49, 0.47],
    ])
    edges = [(0, 1), (0, 5), (1, 2), (1, 6), (2, 3), (2, 6), (3, 4), (3, 6), (4, 5), (4, 6), (5, 6)]
    for i, j in edges:
        ax.plot(
            [x + coords[i, 0] * scale, x + coords[j, 0] * scale],
            [y + coords[i, 1] * scale, y + coords[j, 1] * scale],
            color=GREY, linewidth=0.55, zorder=2,
        )
    for idx, (cx, cy) in enumerate(coords):
        face = VERMILION if highlighted == idx else (color if idx in {0, 2, 4} else "white")
        ax.add_patch(Circle(
            (x + cx * scale, y + cy * scale), 0.085 * scale,
            facecolor=face, edgecolor=INK, linewidth=0.55, zorder=3,
        ))
    ax.text(x + 0.46 * scale, y - 0.12 * scale, label, ha="center", va="top", fontsize=4.7, fontweight="bold")


def gcn_pair(ax, x: float, y: float, color: str, label: str) -> None:
    ax.text(
        x - 0.10, y + 0.29, label,
        ha="right", va="center", fontsize=4.5, fontweight="bold", color=color,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.45}, zorder=6,
    )
    for idx in range(2):
        ax.add_patch(Rectangle(
            (x + idx * 0.74, y), 0.64, 0.58,
            facecolor=color, edgecolor=INK, linewidth=0.55, alpha=0.92, zorder=3,
        ))
        ax.text(x + idx * 0.74 + 0.32, y + 0.29, f"GCN {idx + 1}", ha="center", va="center", fontsize=4.7, fontweight="bold", color="white")
        if idx == 0:
            arrow(ax, (x + 0.65, y + 0.29), (x + 0.73, y + 0.29), color=GREY, width=0.65, mutation=6)


def attention_heatmap(ax, x: float, y: float, w: float = 0.78, h: float = 1.12) -> None:
    values = np.array([
        [0.70, 0.18, 0.12],
        [0.25, 0.63, 0.12],
        [0.15, 0.22, 0.63],
        [0.50, 0.32, 0.18],
        [0.24, 0.51, 0.25],
    ])
    cmap = mpl.colormaps["cividis"]
    rows, cols = values.shape
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle(
                (x + c * w / cols, y + (rows - 1 - r) * h / rows),
                w / cols, h / rows,
                facecolor=cmap(values[r, c]), edgecolor="white", linewidth=0.28, zorder=3,
            ))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=INK, linewidth=0.6, zorder=4))
    ax.text(x + w / 2, y + h + 0.12, "Local attention", ha="center", va="bottom", fontsize=5.1, fontweight="bold")
    ax.text(
        x + w / 2, y - 0.12,
        "MLP 60→64→3\nnode-level softmax α",
        ha="center", va="top", fontsize=3.75, color=GREY, linespacing=1.0,
    )


def embedding_bar(ax, x: float, y: float, height: float = 1.2) -> None:
    colors = [BLUE, SKY, ORANGE]
    for idx, color in enumerate(colors):
        ax.add_patch(Rectangle((x, y + idx * height / 3), 0.28, height / 3, facecolor=color, edgecolor=INK, linewidth=0.45, zorder=3))
    ax.text(x + 0.14, y - 0.13, "60D fused\nembedding", ha="center", va="top", fontsize=4.2, linespacing=1.0)


def cv_icon(ax, x: float, y: float, radius: float = 0.55) -> None:
    colors = [BLUE, SKY, GREEN, ORANGE, PURPLE]
    for idx, color in enumerate(colors):
        angle = math.radians(90 - idx * 72)
        cx, cy = x + radius * math.cos(angle), y + radius * math.sin(angle)
        ax.add_patch(Rectangle((cx - 0.16, cy - 0.12), 0.32, 0.24, facecolor=color, edgecolor=INK, linewidth=0.45, zorder=3))
        ax.text(cx, cy, str(idx + 1), ha="center", va="center", fontsize=4.0, color="white", fontweight="bold")
    arrow(ax, (x + 0.48, y + 0.35), (x + 0.56, y - 0.18), color=PURPLE, width=0.8, mutation=7, connection="arc3,rad=-0.55")
    arrow(ax, (x - 0.48, y - 0.35), (x - 0.56, y + 0.18), color=PURPLE, width=0.8, mutation=7, connection="arc3,rad=-0.55")


def funnel(ax, x: float, y: float, w: float = 1.0, h: float = 1.2) -> None:
    ax.add_patch(Polygon(
        [[x, y + h], [x + w, y + h], [x + 0.65 * w, y + 0.55 * h], [x + 0.58 * w, y], [x + 0.42 * w, y], [x + 0.35 * w, y + 0.55 * h]],
        closed=True, facecolor="#FBE7D6", edgecolor=VERMILION, linewidth=0.8, zorder=3,
    ))
    ax.text(x + w / 2, y + 0.68 * h, "degree /\nstructural filters", ha="center", va="center", fontsize=4.2, linespacing=1.0)


def make_figure() -> plt.Figure:
    style()
    fig, ax = plt.subplots(figsize=(170 / 25.4, 108 / 25.4))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.985, bottom=0.02)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.25)
    ax.axis("off")

    # Panel a: exact model architecture
    panel(ax, 0.18, 3.75, 12.64, 4.28, "a", "Data integration and primary ctMV-GCN model")

    matrix_stack(ax, 0.48, 6.72, 1.0)
    ax.text(1.10, 5.75, "+", ha="center", va="center", fontsize=19, fontweight="bold", color=ORANGE)
    network_icon(ax, 0.55, 4.55, 0.78, BLUE, "STRING")
    network_icon(ax, 1.43, 4.55, 0.78, SKY, "Cell-type")
    network_icon(ax, 2.31, 4.55, 0.78, ORANGE, "Pathway")

    arrow(ax, (3.13, 6.00), (3.68, 6.00), color=ORANGE, width=1.6, mutation=12)
    rounded_box(ax, 3.72, 5.56, 1.08, 0.88, "Training-fitted\nscaling + PCA (5)", face="#E8F5EF", edge=GREEN, fontsize=4.6, weight="bold")

    branch_y = [6.88, 5.92, 4.96]
    labels = ["STRING", "Cell-type", "Pathway"]
    embedding_labels = ["h_STRING", "h_Cell", "h_Path"]
    colors = [BLUE, SKY, ORANGE]
    for row_y, label, embedding_label, color in zip(branch_y, labels, embedding_labels, colors):
        gcn_pair(ax, 5.30, row_y, color, label)
        arrow(ax, (4.82, 6.00), (5.27, row_y + 0.29), color=color, width=0.72, mutation=6)
        ax.add_patch(Rectangle((6.90, row_y + 0.11), 0.20, 0.36, facecolor=color, edgecolor=INK, linewidth=0.45, zorder=3))
        arrow(ax, (6.80, row_y + 0.29), (6.88, row_y + 0.29), color=color, width=0.7, mutation=5)
        ax.text(7.15, row_y + 0.29, embedding_label, ha="left", va="center", fontsize=4.5, color=color, fontweight="bold")

    ax.add_patch(FancyBboxPatch(
        (5.04, 4.70), 2.32, 3.05,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        facecolor="none", edgecolor=VERMILION, linewidth=0.8,
        linestyle=(0, (5, 3)), zorder=2,
    ))
    ax.text(6.20, 7.82, "Three-view GCN encoder", ha="center", va="bottom", fontsize=5.3, fontweight="bold")
    ax.text(6.20, 4.78, "Weighted GCN 5→20 · ReLU/dropout 0.5 · weighted GCN 20→20", ha="center", va="bottom", fontsize=3.85, color=GREY)

    for row_y, color in zip(branch_y, colors):
        arrow(ax, (7.40, row_y + 0.29), (7.73, 6.12), color=color, width=0.65, mutation=5)
    attention_heatmap(ax, 7.76, 5.50, 0.80, 1.20)
    arrow(ax, (8.58, 6.10), (8.92, 6.10), color=GREEN, width=1.0, mutation=8)
    embedding_bar(ax, 8.96, 5.50, 1.20)
    ax.text(9.10, 7.02, "attention-weighted\nconcatenation", ha="center", va="bottom", fontsize=4.3, fontweight="bold")

    arrow(ax, (9.26, 6.10), (9.61, 6.10), color=ORANGE, width=1.2, mutation=9)
    rounded_box(ax, 9.65, 5.55, 1.18, 1.10, "Linear classifier\n60D → 2\n+ softmax", face="#FFF1DD", edge=ORANGE, fontsize=4.7, weight="bold")
    ax.text(10.24, 5.38, "class-weighted CE", ha="center", va="top", fontsize=4.25, color=VERMILION, fontweight="bold")
    arrow(ax, (10.85, 6.10), (11.24, 6.10), color=PURPLE, width=1.3, mutation=10)
    network_icon(ax, 11.38, 5.48, 0.98, PURPLE, "Prioritized unlabeled gene", highlighted=2)
    ax.text(11.86, 7.04, "Clinical-evidence\nprobability", ha="center", va="bottom", fontsize=4.5, fontweight="bold")

    # Panel b: evaluation
    panel(ax, 0.18, 0.18, 3.66, 3.35, "b", "Transductive evaluation")
    cv_icon(ax, 1.00, 2.32, 0.58)
    ax.text(1.00, 1.50, "Repeated 5-fold CV × 3\n15% inner validation", ha="center", va="top", fontsize=4.8, fontweight="bold", linespacing=1.15)
    ax.text(1.00, 0.82, "Test labels hidden\nall nodes and topology visible", ha="center", va="top", fontsize=4.45, color=GREY, fontweight="medium")
    rounded_box(ax, 1.78, 2.49, 0.88, 0.42, "RWR", face="#E8F3FA", edge=BLUE, fontsize=4.3, weight="bold")
    rounded_box(ax, 2.72, 2.49, 0.88, 0.42, "Degree", face="#E8F3FA", edge=BLUE, fontsize=4.3, weight="bold")
    rounded_box(ax, 1.78, 1.92, 0.88, 0.42, "LR / RF", face="#EEF1F2", edge=GREY, fontsize=4.3, weight="bold")
    rounded_box(ax, 2.72, 1.92, 0.88, 0.42, "GCN / GAT", face="#EEF1F2", edge=GREY, fontsize=4.2, weight="bold")
    rounded_box(ax, 1.78, 1.35, 1.82, 0.42, "GraphSAGE + GNN ablations", face="#EEF1F2", edge=GREY, fontsize=4.1, weight="bold")
    for idx, metric in enumerate(["ROC-AUC", "AUPRC", "Top-K"]):
        rounded_box(ax, 1.78 + idx * 0.62, 0.56, 0.54, 0.40, metric, face="#F1EAF5", edge=PURPLE, fontsize=3.7, weight="bold")

    # Panel c: final ensemble and robustness-aware selection
    panel(ax, 4.03, 0.18, 4.56, 3.35, "c", "Robust candidate selection")
    for idx, color in enumerate([BLUE, SKY, GREEN, ORANGE, PURPLE]):
        ax.add_patch(Rectangle((4.32 + idx * 0.10, 2.28 + idx * 0.04), 0.50, 0.62, facecolor=color, edgecolor=INK, linewidth=0.50, zorder=2 + idx))
    ax.text(4.78, 2.12, "10 fixed seeds", ha="center", va="top", fontsize=4.7, fontweight="bold")
    arrow(ax, (5.18, 2.54), (5.55, 2.54), color=ORANGE, width=1.0, mutation=8)

    rounded_box(ax, 5.60, 2.12, 1.43, 0.84, "Mean score\nRank SD\nTop-500 frequency", face="#E8F3FA", edge=BLUE, fontsize=4.45, weight="bold")
    arrow(ax, (7.05, 2.54), (7.30, 2.54), color=ORANGE, width=0.9, mutation=7)
    funnel(ax, 7.34, 1.98, 0.90, 1.02)
    arrow(ax, (7.79, 1.96), (7.79, 1.69), color=ORANGE, width=0.75, mutation=6)
    ax.text(6.10, 1.68, "Robustness score", ha="center", va="bottom", fontsize=4.4, fontweight="bold", color=INK)

    rounded_box(ax, 4.34, 0.62, 1.02, 0.86, "0.50\ndegree-residual\npercentile", face="#E8F3FA", edge=BLUE, fontsize=4.05, weight="bold")
    ax.text(5.45, 1.05, "+", ha="center", va="center", fontsize=7.5, fontweight="bold", color=INK)
    rounded_box(ax, 5.56, 0.62, 1.02, 0.86, "0.25\ntop-500\nfrequency", face="#E8F5EF", edge=GREEN, fontsize=4.05, weight="bold")
    ax.text(6.67, 1.05, "+", ha="center", va="center", fontsize=7.5, fontweight="bold", color=INK)
    rounded_box(ax, 6.78, 0.62, 1.02, 0.86, "0.25\nrank-stability\npercentile", face="#FFF1DD", edge=ORANGE, fontsize=4.05, weight="bold")
    arrow(ax, (7.82, 1.05), (8.02, 1.05), color=VERMILION, width=0.9, mutation=7)
    ax.add_patch(Circle((8.27, 1.05), 0.24, facecolor="#FFF1DD", edgecolor=VERMILION, linewidth=0.9, zorder=5))
    ax.text(8.27, 1.05, "Robust\ntop 20", ha="center", va="center", fontsize=3.75, fontweight="bold", color=VERMILION, zorder=6)

    # Panel d: biological characterization
    panel(ax, 8.78, 0.18, 4.04, 3.35, "d", "Biological characterization")
    card_specs = [
        (9.02, 2.20, "TCGA-LUAD", "Paired expression\nStage · OS · new event", PURPLE),
        (10.69, 2.20, "GSEA", "Primary and degree-\nsensitivity analyses", BLUE),
        (9.02, 1.18, "GSE68465", "Expression and\nsurvival", GREEN),
        (10.69, 1.18, "CPTAC", "Tumor protein\nprofile", ORANGE),
    ]
    for x, y, title, body, color in card_specs:
        rounded_box(ax, x, y, 1.48, 0.78, f"{title}\n{body}", face="white", edge=color, fontsize=4.25, weight="bold")
    rounded_box(ax, 9.50, 0.40, 1.72, 0.54, "DepMap LUAD\nCRISPR dependency", face="white", edge=VERMILION, fontsize=4.1, weight="bold")
    evidence_center = (12.38, 0.76)
    for x, y in [(9.76, 2.18), (11.43, 2.18), (9.76, 1.16), (11.43, 1.16), (11.24, 0.67)]:
        arrow(ax, (x, y), evidence_center, color=GREY, width=0.45, mutation=4, zorder=2)
    ax.add_patch(Circle(evidence_center, 0.28, facecolor="#E8F5EF", edgecolor=GREEN, linewidth=0.9, zorder=6))
    ax.text(*evidence_center, "Integrated\nevidence", ha="center", va="center", fontsize=3.55, fontweight="bold", zorder=7)

    # Connect the analytical stages.
    arrow(ax, (2.02, 3.73), (2.02, 3.55), color=PURPLE, width=0.95, mutation=7)
    arrow(ax, (3.86, 1.86), (4.01, 1.86), color=ORANGE, width=1.2, mutation=9)
    arrow(ax, (8.61, 1.86), (8.76, 1.86), color=ORANGE, width=1.2, mutation=9)

    ax.text(
        6.5, 0.02,
        "Post hoc robust prioritization; external outcomes were not used for selection. Associations do not establish therapeutic efficacy or causality.",
        ha="center", va="bottom", fontsize=4.3, color=GREY, fontstyle="italic",
    )
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    base = OUT_DIR / BASE_NAME
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
    print(f"Reference-style framework schematic written to {OUT_DIR}")


if __name__ == "__main__":
    main()
