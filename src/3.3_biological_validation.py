from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

from common import torch_load
from config import (
    MODEL_DIR,
    RAW_DATASET_FILE,
    ROBUSTNESS_DIR,
    TCGA_CLINICAL_FILE,
    TCGA_EXPRESSION_FILE,
    VALIDATION_DIR,
    ensure_result_dirs,
)
from plot_style import FULL_WIDTH, apply_bmc_style, save_figure


BOOTSTRAP_REPLICATES = 2000
RANDOM_SEED = 42


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    samples = rng.choice(values, size=(BOOTSTRAP_REPLICATES, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def signed_rank_biserial(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[values != 0]
    if not len(values):
        return 0.0
    ranks = rankdata(np.abs(values))
    return float(np.sum(np.sign(values) * ranks) / np.sum(ranks))


def paired_tumor_normal(expression: pd.DataFrame, genes: list[str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    expression = expression.groupby(level=0).mean()
    sample_data = expression.loc[genes].T
    sample_data.index = sample_data.index.astype(str)
    sample_data["patient"] = sample_data.index.str[:12]
    sample_data["sample_type"] = sample_data.index.str[13:15]
    tumor = sample_data[sample_data["sample_type"] == "01"].groupby("patient")[genes].mean()
    normal = sample_data[sample_data["sample_type"] == "11"].groupby("patient")[genes].mean()
    patients = tumor.index.intersection(normal.index)
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict] = []
    paired_frames: dict[str, pd.DataFrame] = {}
    for rank, gene in enumerate(genes, start=1):
        pair = pd.DataFrame({"Normal": normal.loc[patients, gene], "Tumor": tumor.loc[patients, gene]}).dropna()
        if pair.to_numpy().max() > 50:
            pair = np.log2(pair + 1)
        differences = (pair["Tumor"] - pair["Normal"]).to_numpy()
        try:
            p_value = float(wilcoxon(differences, zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            p_value = 1.0
        ci_lower, ci_upper = bootstrap_median_ci(differences, rng)
        rows.append({
            "Robust_Candidate_Rank": rank,
            "Gene": gene,
            "Paired_Patients": len(pair),
            "Normal_Median": float(pair["Normal"].median()),
            "Tumor_Median": float(pair["Tumor"].median()),
            "Median_Paired_Difference": float(np.median(differences)),
            "Difference_CI_Lower": ci_lower,
            "Difference_CI_Upper": ci_upper,
            "Rank_Biserial_Effect": signed_rank_biserial(differences),
            "Wilcoxon_P": p_value,
        })
        paired_frames[gene] = pair
    results = pd.DataFrame(rows)
    results["Wilcoxon_FDR"] = multipletests(results["Wilcoxon_P"], method="fdr_bh")[1]
    results.to_csv(VALIDATION_DIR / "TCGA_paired_tumor_normal_robust_top20.csv", index=False, encoding="utf-8-sig")
    return results, paired_frames


def plot_paired_results(results: pd.DataFrame, paired_frames: dict[str, pd.DataFrame]) -> None:
    apply_bmc_style()
    ordered = results.sort_values("Median_Paired_Difference").reset_index(drop=True)
    y = np.arange(len(ordered))
    colors = np.where(ordered["Wilcoxon_FDR"] < 0.05, "#D55E00", "#777777")
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.72, 5.2))
    for index, row in ordered.iterrows():
        ax.errorbar(
            row["Median_Paired_Difference"],
            index,
            xerr=np.array([[max(0.0, row["Median_Paired_Difference"] - row["Difference_CI_Lower"])], [max(0.0, row["Difference_CI_Upper"] - row["Median_Paired_Difference"])]]),
            fmt="o",
            color=colors[index],
            capsize=2,
        )
    ax.axvline(0, color="0.5", linestyle="--")
    ax.set(
        yticks=y,
        yticklabels=ordered["Gene"],
        xlabel="Median paired expression difference (tumor - adjacent normal)",
        title="TCGA-LUAD paired tumor-normal evidence",
    )
    ax.grid(axis="x", color="0.9", linewidth=0.5)
    fig.text(0.5, 0.012, "Orange: FDR < 0.05; grey: not significant. Error bars show bootstrap 95% CIs for the median.", ha="center", fontsize=6)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save_figure(fig, VALIDATION_DIR / "TCGA_paired_tumor_normal_forest")

    top = results.sort_values(["Wilcoxon_FDR", "Robust_Candidate_Rank"]).head(4)
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.5), sharex=True)
    for ax, row in zip(axes.flat, top.itertuples(index=False)):
        pair = paired_frames[row.Gene]
        for values in pair.to_numpy():
            ax.plot([0, 1], values, color="0.75", linewidth=0.45, alpha=0.45)
        ax.scatter(np.zeros(len(pair)), pair["Normal"], color="#0072B2", s=8, zorder=3)
        ax.scatter(np.ones(len(pair)), pair["Tumor"], color="#D55E00", s=8, zorder=3)
        ax.plot([0, 1], [pair["Normal"].median(), pair["Tumor"].median()], color="black", marker="D", markersize=3, linewidth=1.2)
        ax.set(xticks=[0, 1], xticklabels=["Adjacent normal", "Tumor"], title=f"{row.Gene} (FDR={row.Wilcoxon_FDR:.2g})")
        ax.grid(axis="y", color="0.9", linewidth=0.5)
    axes[0, 0].set_ylabel("Expression")
    axes[1, 0].set_ylabel("Expression")
    fig.suptitle("Robustness-aware candidates in paired TCGA samples", y=0.995, fontsize=9)
    fig.tight_layout()
    save_figure(fig, VALIDATION_DIR / "TCGA_paired_top4_expression")


def stage_association(expression: pd.DataFrame, clinical: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    expression = expression.groupby(level=0).mean().loc[genes].T.reset_index().rename(columns={"index": "sample"})
    expression = expression[expression["sample"].astype(str).str[13:15] == "01"]
    clinical = clinical[["sampleID", "pathologic_stage"]].drop_duplicates("sampleID").rename(columns={"sampleID": "sample"})
    merged = expression.merge(clinical, on="sample", how="inner")

    def stage_group(value):
        text = str(value).upper().replace("STAGE", "").strip()
        if text.startswith(("III", "IV")):
            return "Late (III-IV)"
        if text.startswith(("I", "II")):
            return "Early (I-II)"
        return np.nan

    merged["Stage_Group"] = merged["pathologic_stage"].map(stage_group)
    rows = []
    for rank, gene in enumerate(genes, start=1):
        frame = merged[[gene, "Stage_Group"]].dropna()
        values = frame[gene].astype(float)
        if values.max() > 50:
            values = np.log2(values + 1)
        early = values[frame["Stage_Group"] == "Early (I-II)"].to_numpy()
        late = values[frame["Stage_Group"] == "Late (III-IV)"].to_numpy()
        test = mannwhitneyu(late, early, alternative="two-sided")
        rows.append({
            "Robust_Candidate_Rank": rank,
            "Gene": gene,
            "Early_N": len(early),
            "Late_N": len(late),
            "Early_Median": float(np.median(early)),
            "Late_Median": float(np.median(late)),
            "Late_Minus_Early_Median": float(np.median(late) - np.median(early)),
            "Cliffs_Delta_Late_vs_Early": float(2 * test.statistic / (len(late) * len(early)) - 1),
            "Mann_Whitney_P": float(test.pvalue),
        })
    results = pd.DataFrame(rows)
    results["Mann_Whitney_FDR"] = multipletests(results["Mann_Whitney_P"], method="fdr_bh")[1]
    results.to_csv(VALIDATION_DIR / "TCGA_stage_association_robust_top20.csv", index=False, encoding="utf-8-sig")
    return results


def view_degree_bias(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = torch_load(RAW_DATASET_FILE)
    view_attributes = {
        "STRING": ("edge_index_ppi", "edge_weight_ppi"),
        "Cell-type similarity": ("edge_index_coexp", "edge_weight_coexp"),
        "Pathway": ("edge_index_pathway", "edge_weight_pathway"),
    }
    annotated = candidates.copy()
    summary_rows = []
    plot_data = []
    for view, (edge_name, weight_name) in view_attributes.items():
        edge_index = getattr(data, edge_name).cpu().numpy()
        edge_weight = getattr(data, weight_name).cpu().numpy()
        degree = np.bincount(edge_index[0], minlength=data.num_nodes)
        weighted_degree = np.bincount(edge_index[0], weights=edge_weight, minlength=data.num_nodes)
        node_ids = annotated["Node_ID"].to_numpy(dtype=int)
        annotated[f"{view}_Degree"] = degree[node_ids]
        annotated[f"{view}_Weighted_Degree"] = weighted_degree[node_ids]
        x = np.log1p(degree[node_ids])
        y = annotated["Prediction_Score_Mean"].to_numpy()
        rho, p_value = spearmanr(x, y)
        top_mask = annotated["Candidate_Rank"] <= 100
        test = mannwhitneyu(degree[node_ids][top_mask], degree[node_ids][~top_mask], alternative="two-sided")
        summary_rows.append({
            "View": view,
            "Spearman_Rho_Score_vs_LogDegree": float(rho),
            "Spearman_P": float(p_value),
            "Top100_Median_Degree": float(np.median(degree[node_ids][top_mask])),
            "Other_Median_Degree": float(np.median(degree[node_ids][~top_mask])),
            "Top100_vs_Other_P": float(test.pvalue),
        })
        plot_data.append((view, x, y))
    annotated.to_csv(VALIDATION_DIR / "unlabeled_candidates_with_network_degree.csv.gz", index=False, compression="gzip")
    summary = pd.DataFrame(summary_rows)
    summary["Spearman_FDR"] = multipletests(summary["Spearman_P"], method="fdr_bh")[1]
    summary["Top100_vs_Other_FDR"] = multipletests(summary["Top100_vs_Other_P"], method="fdr_bh")[1]
    summary.to_csv(VALIDATION_DIR / "view_degree_bias_summary.csv", index=False, encoding="utf-8-sig")

    apply_bmc_style()
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.45), sharey=True)
    for ax, (view, x, y), result in zip(axes, plot_data, summary.itertuples(index=False)):
        ax.hexbin(x, y, gridsize=35, mincnt=1, cmap="Blues", bins="log", linewidths=0)
        top = annotated.head(5)
        top_x = np.log1p(top[f"{view}_Degree"].to_numpy())
        ax.scatter(top_x, top["Prediction_Score_Mean"], color="#D55E00", s=11, zorder=3)
        ax.set(title=view, xlabel="log(1 + degree)")
        ax.text(0.04, 0.05, f"Spearman rho={result.Spearman_Rho_Score_vs_LogDegree:.2f}", transform=ax.transAxes, fontsize=6)
        ax.grid(color="0.92", linewidth=0.4)
    axes[0].set_ylabel("GNN prediction score")
    fig.suptitle("Network-degree bias assessment", y=1.01, fontsize=9)
    fig.tight_layout()
    save_figure(fig, VALIDATION_DIR / "network_degree_bias")
    return summary, annotated


def run_degree_bias() -> None:
    candidates = pd.read_csv(MODEL_DIR / "unlabeled_candidate_ranking.csv")
    degree_summary, _ = view_degree_bias(candidates)
    print(degree_summary.to_string(index=False))


def run_candidate_validation() -> None:
    robust_file = ROBUSTNESS_DIR / "robust_top20.csv"
    if not robust_file.exists():
        raise FileNotFoundError(
            f"Run 4.1_robustness.py before candidate validation: {robust_file}"
        )
    candidates = pd.read_csv(robust_file).sort_values("Robust_Candidate_Rank")
    expression = pd.read_csv(TCGA_EXPRESSION_FILE, sep="\t", index_col=0)
    clinical = pd.read_csv(TCGA_CLINICAL_FILE, sep="\t")
    genes = [gene for gene in candidates["Gene_Symbol"] if gene in expression.index]
    missing_genes = [gene for gene in candidates["Gene_Symbol"] if gene not in expression.index]

    paired_results, paired_frames = paired_tumor_normal(expression, genes)
    plot_paired_results(paired_results, paired_frames)
    stage_results = stage_association(expression, clinical, genes)
    degree_file = VALIDATION_DIR / "unlabeled_candidates_with_network_degree.csv.gz"
    if not degree_file.exists():
        raise FileNotFoundError(f"Run --degree-only before candidate validation: {degree_file}")
    degree_annotated = pd.read_csv(degree_file)

    paired_evidence = paired_results.drop(columns="Robust_Candidate_Rank").rename(columns={"Gene": "Gene_Symbol"})
    stage_evidence = stage_results.drop(columns="Robust_Candidate_Rank").rename(columns={"Gene": "Gene_Symbol"})
    evidence = candidates.merge(paired_evidence, on="Gene_Symbol", how="left")
    evidence = evidence.merge(stage_evidence, on="Gene_Symbol", how="left", suffixes=("", "_Stage"))
    missing_degree_columns = [
        column for column in degree_annotated.columns
        if "Degree" in column and column not in evidence.columns
    ]
    if missing_degree_columns:
        evidence = evidence.merge(
            degree_annotated[["Gene_Symbol", *missing_degree_columns]],
            on="Gene_Symbol", how="left",
        )
    evidence.to_csv(VALIDATION_DIR / "robust_top20_integrated_candidate_evidence.csv", index=False, encoding="utf-8-sig")

    significant_paired = paired_results.loc[paired_results["Wilcoxon_FDR"] < 0.05, "Gene"].tolist()
    significant_stage = stage_results.loc[stage_results["Mann_Whitney_FDR"] < 0.05, "Gene"].tolist()
    summary = {
        "candidate_selection": "Post hoc robustness-aware top 20; not preregistered or prespecified.",
        "tested_genes": genes,
        "unmapped_genes": missing_genes,
        "paired_patients_minimum": int(paired_results["Paired_Patients"].min()),
        "paired_tumor_normal_FDR_lt_0.05": significant_paired,
        "stage_association_FDR_lt_0.05": significant_stage,
        "interpretation": "These are secondary TCGA expression and clinical-association analyses, not independent experimental validation of therapeutic efficacy.",
    }
    (VALIDATION_DIR / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (VALIDATION_DIR / "validation_scope.txt").write_text(
        "The paired tumor-normal analysis uses matched TCGA-LUAD patient samples and the post hoc robustness-aware top 20. "
        "Stage associations use the same TCGA cohort. Network-degree analyses assess potential topology bias. "
        "These analyses strengthen biological plausibility and robustness assessment but are not independent cohort or experimental validation.\n",
        encoding="utf-8",
    )
    print(paired_results.to_string(index=False))
    print(stage_results.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--degree-only", action="store_true")
    group.add_argument("--candidate-validation-only", action="store_true")
    args = parser.parse_args()
    ensure_result_dirs()
    if args.degree_only:
        run_degree_bias()
    elif args.candidate_validation_only:
        run_candidate_validation()
    else:
        run_degree_bias()
        run_candidate_validation()


if __name__ == "__main__":
    main()
