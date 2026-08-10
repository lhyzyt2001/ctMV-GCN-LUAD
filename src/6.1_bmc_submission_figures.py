from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_curve

from config import RESULT_ROOT

RESULTS = RESULT_ROOT
OUT = RESULTS / "manuscript_submission"
OUT.mkdir(parents=True, exist_ok=True)

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "yellow": "#E69F00",
    "pink": "#CC79A7",
    "purple": "#7A5195",
    "black": "#000000",
    "grey": "#777777",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 6.6,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=600, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def labels_from_frozen_inputs() -> np.ndarray:
    data_dir = RESULTS / "01_data"
    mapping = json.loads((data_dir / "node_mapping.json").read_text(encoding="utf-8"))
    positives = set(pd.read_csv(data_dir / "clinical_target_labels.csv")["symbol"].astype(str))
    y = np.zeros(len(mapping), dtype=int)
    for symbol in positives:
        node_id = mapping.get(symbol)
        if node_id is not None:
            y[int(node_id)] = 1
    return y


def repeat_curves(oof: pd.DataFrame, y_all: np.ndarray, method: str):
    roc_grid = np.linspace(0.0, 1.0, 501)
    recall_grid = np.linspace(0.0, 1.0, 501)
    tprs, precisions = [], []
    for repeat in sorted(oof["repeat"].unique()):
        part = oof[(oof["repeat"] == repeat) & (oof["method"] == method)].sort_values("node_id")
        node_ids = part["node_id"].to_numpy(dtype=int)
        y = y_all[node_ids]
        score = part["score"].to_numpy(float)
        fpr, tpr, _ = roc_curve(y, score)
        tprs.append(np.interp(roc_grid, fpr, tpr))
        precision, recall, _ = precision_recall_curve(y, score)
        order = np.argsort(recall)
        precisions.append(np.interp(recall_grid, recall[order], precision[order]))
    return roc_grid, np.asarray(tprs), recall_grid, np.asarray(precisions)


def figure_2_and_3() -> None:
    benchmark = RESULTS / "02_benchmark"
    oof = pd.read_csv(benchmark / "oof_predictions.csv.gz")
    summary = pd.read_csv(benchmark / "benchmark_summary.csv").set_index("Method")
    y = labels_from_frozen_inputs()
    prevalence = float(y.mean())

    proposed = "ctMV-GCN (local attention + class-weighted CE)"
    display = {
        "RWR (STRING graph)": "RWR (STRING graph)",
        "Network-degree logistic": "Network-degree logistic",
        proposed: "ctMV-GCN (primary)",
        "PPI-GraphSAGE + nnPU": "STRING-GraphSAGE + nnPU",
        "PPI-GAT + nnPU": "STRING-GAT + nnPU",
        "PPI-GCN + nnPU": "STRING-GCN + nnPU",
        "Logistic regression": "Feature logistic regression",
        "Random forest": "Feature random forest",
        "ctMV-GCN (dual attention + nnPU)": "ctMV-GCN (dual attention + nnPU)",
        "ctMV-GCN (equal fusion + nnPU)": "ctMV-GCN (equal fusion + nnPU)",
    }
    methods = [
        "RWR (STRING graph)",
        "Network-degree logistic",
        proposed,
        "PPI-GraphSAGE + nnPU",
        "PPI-GAT + nnPU",
        "PPI-GCN + nnPU",
        "Logistic regression",
        "Random forest",
    ]
    colors = [
        OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"],
        OKABE_ITO["pink"], OKABE_ITO["yellow"], OKABE_ITO["sky"],
        OKABE_ITO["black"], OKABE_ITO["purple"],
    ]

    def curves(method_list: list[str], color_list: list[str], stem: str, shade_all: bool) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(6.69, 3.75))
        for color, method in zip(color_list, method_list):
            xroc, tprs, xpr, prs = repeat_curves(oof, y, method)
            mean_tpr, sd_tpr = tprs.mean(axis=0), tprs.std(axis=0, ddof=1)
            mean_pr, sd_pr = prs.mean(axis=0), prs.std(axis=0, ddof=1)
            row = summary.loc[method]
            label = f"{display[method]} [{row.ROC_AUC_mean:.3f} | {row.AUPRC_mean:.3f}]"
            lw = 1.7 if method == proposed else 1.15
            axes[0].plot(xroc, mean_tpr, color=color, lw=lw, label=label)
            axes[1].plot(xpr, mean_pr, color=color, lw=lw, label=label)
            if shade_all or method in {"RWR (STRING graph)", "Network-degree logistic", proposed}:
                axes[0].fill_between(xroc, np.clip(mean_tpr - sd_tpr, 0, 1), np.clip(mean_tpr + sd_tpr, 0, 1), color=color, alpha=0.08, lw=0)
                axes[1].fill_between(xpr, np.clip(mean_pr - sd_pr, 0, 1), np.clip(mean_pr + sd_pr, 0, 1), color=color, alpha=0.08, lw=0)
        axes[0].plot([0, 1], [0, 1], color="0.5", ls="--", lw=1.0, label=f"Random baseline [0.500 | {prevalence:.3f}]")
        axes[1].axhline(prevalence, color="0.5", ls="--", lw=1.0)
        axes[0].set(title="a  ROC curves", xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
        axes[1].set(title="b  Precision–recall curves", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
        for ax in axes:
            ax.grid(color="0.9", lw=0.45)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.text(0.5, 0.185, "Legend values: mean outer-fold ROC-AUC | AUPRC", ha="center", fontsize=7)
        fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False, fontsize=6.3)
        fig.tight_layout(rect=(0, 0.27, 1, 1))
        save(fig, stem)

    curves(methods, colors, "Figure_2", shade_all=False)
    ablations = [
        proposed,
        "ctMV-GCN (dual attention + nnPU)",
        "ctMV-GCN (equal fusion + nnPU)",
        "PPI-GCN + nnPU",
    ]
    curves(
        ablations,
        [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"], OKABE_ITO["pink"]],
        "Figure_3",
        shade_all=True,
    )


def figure_4() -> None:
    root = RESULTS / "08_robustness"
    matched = pd.read_csv(root / "degree_matched_bootstrap_metrics.csv")
    matched = matched[matched["Metric"] == "AUPRC"].copy()
    order = ["ctMV-GCN (local attention + class-weighted CE)", "Network-degree logistic", "RWR (STRING graph)"]
    labels = ["ctMV-GCN", "Degree logistic", "RWR"]
    colors = [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"]]

    null = pd.read_csv(root / "degree_conditional_permutation_null.csv.gz")
    test = pd.read_csv(root / "degree_conditional_permutation_test.csv").iloc[0]
    residual = pd.read_csv(root / "degree_residualized_oof_metrics.csv")
    edge = pd.read_csv(root / "edge_dropout_rank_stability.csv")

    fig, axes = plt.subplots(2, 2, figsize=(6.69, 5.5))
    ax = axes[0, 0]
    for i, (method, label, color) in enumerate(zip(order, labels, colors)):
        r = matched[matched["Method"] == method].iloc[0]
        xerr = [[r.Bootstrap_Median - r.CI_Lower], [r.CI_Upper - r.Bootstrap_Median]]
        ax.errorbar(r.Bootstrap_Median, i, xerr=xerr, fmt="o", color=color, capsize=2.5, lw=1.2, ms=4.5)
        ax.text(r.Bootstrap_Median + 0.006, i, f"{r.Bootstrap_Median:.3f}", va="center", fontsize=6.7)
    ax.axvline(0.5, color="0.5", ls="--", lw=1)
    ax.set(yticks=range(3), yticklabels=labels, xlabel="Degree-matched AUPRC (95% bootstrap CI)", title="a  Degree-matched discrimination", xlim=(0.48, 0.78))
    ax.invert_yaxis()
    ax.grid(axis="x", color="0.9", lw=0.45)

    ax = axes[0, 1]
    null_col = [c for c in null.columns if "ROC" in c.upper() or "AUC" in c.upper()]
    values = null[null_col[0]].to_numpy(float) if null_col else null.select_dtypes(include=[np.number]).iloc[:, -1].to_numpy(float)
    ax.hist(values, bins=28, color="#B8C7D9", edgecolor="white", lw=0.4)
    ax.axvline(float(test.Observed_Degree_Stratified_ROC_AUC), color=OKABE_ITO["orange"], lw=1.8, label=f"Observed = {test.Observed_Degree_Stratified_ROC_AUC:.3f}")
    ax.axvline(float(test.Null_Mean), color="0.35", ls="--", lw=1.1, label=f"Null mean = {test.Null_Mean:.3f}")
    ax.set(title="b  Degree-conditional permutation", xlabel="Degree-stratified ROC-AUC", ylabel="Permutation count")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.98, 0.92, f"P = {test.One_Sided_P:.4f}", transform=ax.transAxes, ha="right", va="top", fontsize=7)

    ax = axes[1, 0]
    xpos = np.arange(2)
    width = 0.34
    raw = residual.iloc[0]
    adj = residual.iloc[1]
    ax.bar(xpos - width / 2, [raw.ROC_AUC, raw.AUPRC], width, color=OKABE_ITO["blue"], label="Raw GNN score")
    ax.bar(xpos + width / 2, [adj.ROC_AUC, adj.AUPRC], width, color=OKABE_ITO["orange"], label="Degree-residualized score")
    for x, val in zip(xpos - width / 2, [raw.ROC_AUC, raw.AUPRC]):
        ax.text(x, val + 0.02, f"{val:.3f}", ha="center", fontsize=6.5)
    for x, val in zip(xpos + width / 2, [adj.ROC_AUC, adj.AUPRC]):
        ax.text(x, val + 0.02, f"{val:.3f}", ha="center", fontsize=6.5)
    ax.axhline(0.0469, color="0.5", ls="--", lw=1, label="Original prevalence")
    ax.set(xticks=xpos, xticklabels=["ROC-AUC", "AUPRC"], ylabel="Metric value", ylim=(0, 1), title="c  Nonlinear degree residualization")
    ax.legend(frameon=False, loc="upper right", fontsize=6.2)
    ax.grid(axis="y", color="0.9", lw=0.45)

    ax = axes[1, 1]
    grouped = edge.groupby("Edge_Dropout_Fraction")
    x = np.array(sorted(grouped.groups)) * 100
    metrics = [
        ("Spearman_Rho_All_Unlabeled", "Rank correlation", OKABE_ITO["blue"], "o"),
        ("Top100_Jaccard", "Top-100 Jaccard", OKABE_ITO["orange"], "s"),
        ("Top500_Jaccard", "Top-500 Jaccard", OKABE_ITO["green"], "^")
    ]
    for col, label, color, marker in metrics:
        mean = grouped[col].mean().reindex(x / 100).to_numpy()
        sd = grouped[col].std().reindex(x / 100).to_numpy()
        ax.errorbar(x, mean, yerr=sd, marker=marker, color=color, capsize=2, label=label)
    ax.set(title="d  Inference-time edge removal", xlabel="Edges removed from each view (%)", ylabel="Ranking stability", ylim=(0.75, 1.01), xticks=x)
    ax.legend(frameon=False, loc="lower left", fontsize=6.2)
    ax.grid(color="0.9", lw=0.45)

    fig.tight_layout(pad=1.2, h_pad=1.6, w_pad=1.6)
    save(fig, "Figure_4")


def figure_5() -> None:
    data = pd.read_csv(RESULTS / "08_robustness" / "robust_top20.csv").sort_values("Robust_Candidate_Rank")
    fig, ax = plt.subplots(figsize=(4.78, 4.35))
    ordered = data.iloc[::-1]
    ax.barh(ordered["Gene_Symbol"], ordered["Robustness_Score"], color=OKABE_ITO["blue"])
    for y, value in enumerate(ordered["Robustness_Score"]):
        ax.text(value + 0.003, y, f"{value:.3f}", va="center", fontsize=6.2)
    ax.set(xlabel="Post hoc robustness score (not a probability)", ylabel="", xlim=(0.90, 1.01), title="Robust ctMV-GCN candidate shortlist")
    ax.grid(axis="x", color="0.9", lw=0.45)
    fig.tight_layout()
    save(fig, "Figure_5")


def _forest(ax, data, gene_col, estimate, lo, hi, fdr, title, xlabel, sig_color, xlim=None):
    ordered = data.sort_values(estimate, ascending=False).reset_index(drop=True)
    y = np.arange(len(ordered))
    sig = ordered[fdr].astype(float) < 0.05
    colors = np.where(sig, sig_color, "#888888")
    for i, row in ordered.iterrows():
        ax.errorbar(row[estimate], i, xerr=[[row[estimate] - row[lo]], [row[hi] - row[estimate]]], fmt="o", color=colors[i], ecolor=colors[i], capsize=2, lw=1.05, ms=3.8)
    ax.axvline(1.0 if "HR" in estimate else 0.0, color="0.5", ls="--", lw=1)
    ax.set(yticks=y, yticklabels=ordered[gene_col], xlabel=xlabel, title=title)
    ax.invert_yaxis()
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(axis="x", color="0.9", lw=0.45)
    return int(sig.sum())


def figure_8() -> None:
    paired = pd.read_csv(RESULTS / "07_validation" / "TCGA_paired_tumor_normal_robust_top20.csv")
    overall = pd.read_csv(RESULTS / "06_tcga" / "TCGA_LUAD_robust_top20_survival_results.csv")
    event = pd.read_csv(RESULTS / "06_tcga" / "TCGA_LUAD_robust_top20_new_tumor_event_results.csv")
    fig = plt.figure(figsize=(6.69, 7.55))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.08], hspace=0.43, wspace=0.44)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    n1 = _forest(ax1, paired, "Gene", "Median_Paired_Difference", "Difference_CI_Lower", "Difference_CI_Upper", "Wilcoxon_FDR", "a  Paired expression (16/20 FDR < 0.05)", "Median paired difference", OKABE_ITO["orange"])
    n2 = _forest(ax2, overall, "Gene", "Adjusted_HR", "CI_Lower", "CI_Upper", "Adjusted_Cox_FDR", "b  Overall survival (0/20 FDR < 0.05)", "Hazard ratio per expression SD", OKABE_ITO["blue"], (0.68, 1.42))
    n3 = _forest(ax3, event, "Gene", "Adjusted_HR", "CI_Lower", "CI_Upper", "Adjusted_Cox_FDR", "c  New-tumor event (2/20 FDR < 0.05)", "Hazard ratio per expression SD", OKABE_ITO["green"], (0.65, 1.37))
    assert (n1, n2, n3) == (16, 0, 2)
    fig.text(0.5, 0.006, "Colored points: FDR < 0.05; grey points: not significant. Error bars show 95% confidence intervals.", ha="center", fontsize=6.5)
    save(fig, "Figure_8")


def figure_10() -> None:
    data = pd.read_csv(RESULTS / "08_robustness" / "external_validation" / "depmap_luad_dependency_validation.csv")
    tested = data[data["Mapping_Status"] == "tested"].sort_values("Median_Gene_Effect").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(4.78, 4.45))
    y = np.arange(len(tested))
    ax.errorbar(
        tested["Median_Gene_Effect"], y,
        xerr=np.vstack([
            tested["Median_Gene_Effect"] - tested["IQR_Lower"],
            tested["IQR_Upper"] - tested["Median_Gene_Effect"],
        ]),
        fmt="o", color="#8C8C8C", ecolor="#8C8C8C", capsize=2, lw=1.1, ms=4.2,
    )
    ax.axvline(0, color="black", lw=1, label="No effect")
    ax.axvline(-0.5, color=OKABE_ITO["orange"], ls="--", lw=1.1, label="Dependency threshold (−0.5)")
    ax.set(yticks=y, yticklabels=tested["Gene"], xlabel="DepMap Chronos gene effect (median and IQR)", title="LUAD cell-line CRISPR dependency", xlim=(-0.53, 0.22))
    ax.invert_yaxis()
    ax.grid(axis="x", color="0.9", lw=0.45)
    ax.legend(frameon=False, loc="upper left")
    max_row = tested.loc[tested["Fraction_Gene_Effect_le_-0.5"].idxmax()]
    ax.text(
        0.98, 0.97,
        f"0/19 met the prespecified rule\nMaximum fraction ≤−0.5: {max_row.Gene} {100*max_row['Fraction_Gene_Effect_le_-0.5']:.1f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.7,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.95},
    )
    fig.tight_layout()
    save(fig, "Figure_10")


def package_existing_composites() -> None:
    figure_1 = RESULTS / "00_workflow" / "Figure_1_ctMV_GCN_framework_schematic"
    for suffix in (".pdf", ".png"):
        source = figure_1.with_suffix(suffix)
        if not source.exists():
            raise FileNotFoundError(f"Run 5.2_framework_schematic.py first: {source}")
        shutil.copy2(source, OUT / f"Figure_1{suffix}")

    composites = {
        6: RESULTS / "_composites" / "Figure_6_interpretability.png",
        7: RESULTS / "_composites" / "Figure_7_gsea_sensitivity.png",
    }
    for number, source in composites.items():
        if not source.exists():
            raise FileNotFoundError(source)
        png_target = OUT / f"Figure_{number}.png"
        pdf_target = OUT / f"Figure_{number}.pdf"
        shutil.copy2(source, png_target)
        with Image.open(source) as image:
            image.convert("RGB").save(pdf_target, "PDF", resolution=300.0)

    external = RESULTS / "08_robustness" / "external_validation" / "independent_external_validation"
    for suffix in (".pdf", ".png"):
        source = external.with_suffix(suffix)
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, OUT / f"Figure_9{suffix}")


def write_manifest() -> None:
    records = []
    for file in sorted(OUT.glob("Figure_*.*")):
        if file.suffix.lower() not in {".pdf", ".png"}:
            continue
        records.append(
            {
                "file": file.name,
                "bytes": file.stat().st_size,
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            }
        )
    (OUT / "FIGURE_MANIFEST.json").write_text(
        json.dumps({"release": "v1.0.2", "files": records}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure_2_and_3()
    figure_4()
    figure_5()
    figure_8()
    figure_10()
    package_existing_composites()
    write_manifest()
    print(f"Revised figures written to: {OUT}")


if __name__ == "__main__":
    main()
