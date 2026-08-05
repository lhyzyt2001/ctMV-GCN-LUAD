from __future__ import annotations

import argparse
import hashlib
import json
import re
import textwrap
from datetime import date

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from common import load_feature_table
from config import EXTERNAL_DATA_ROOT, FEATURE_FILE, INTERPRET_DIR, MODEL_DIR, ROBUSTNESS_DIR, VALIDATION_DIR, ensure_result_dirs
from plot_style import FULL_WIDTH, apply_bmc_style, save_figure


LIBRARIES = ("MSigDB_Hallmark_2020", "GO_Biological_Process_2021")
GENE_SET_FILES = {
    library: EXTERNAL_DATA_ROOT / "gene_sets" / f"Enrichr.{library}.gmt"
    for library in LIBRARIES
}
PERMUTATIONS = 2000
HUB_PERCENTILE = 0.95
STRUCTURAL_PATTERN = re.compile(r"^(?:RPL\d|RPS\d|MRPL\d|MRPS\d|MT-)", flags=re.IGNORECASE)
LEADING_EDGE_OVERLAP_THRESHOLD = 0.50


def download_gene_sets() -> dict[str, list[str]]:
    merged = {}
    manifest = {"retrieval_date": date.today().isoformat(), "permutations": PERMUTATIONS, "libraries": {}}
    for library in LIBRARIES:
        snapshot = GENE_SET_FILES[library]
        if snapshot.exists():
            genesets = {}
            with snapshot.open("r", encoding="utf-8") as handle:
                for line in handle:
                    term, _, *genes = line.rstrip("\n").split("\t")
                    genesets[term] = [gene for gene in genes if gene]
            source = "versioned local Enrichr GMT snapshot"
        else:
            genesets = gp.get_library(name=library, organism="Human")
            source = "downloaded from Enrichr via gseapy"
        manifest["libraries"][library] = {
            "gene_set_count": len(genesets),
            "source": source,
            "snapshot": f"external_data_root/gene_sets/{snapshot.name}" if snapshot.exists() else None,
            "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest() if snapshot.exists() else None,
        }
        for term, genes in genesets.items():
            merged[f"{library}::{term}"] = genes
    (INTERPRET_DIR / "gene_set_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return merged


def make_celltype_heatmap(candidates: pd.DataFrame) -> None:
    features = load_feature_table(FEATURE_FILE)
    genes = [gene for gene in candidates.head(20)["Gene_Symbol"] if gene in features.index]
    matrix = features.loc[genes]
    z = matrix.sub(matrix.mean(axis=1), axis=0).div(matrix.std(axis=1).replace(0, 1), axis=0)
    apply_bmc_style()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4.2))
    sns.heatmap(z, cmap="vlag", center=0, linewidths=0.2, linecolor="white", cbar_kws={"label": "Within-gene z score"}, ax=ax)
    ax.set(xlabel="Cell type", ylabel="Candidate gene", title="Cell-type expression specificity of top-ranked candidates")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    save_figure(fig, INTERPRET_DIR / "candidate_celltype_specificity")
    z.to_csv(INTERPRET_DIR / "candidate_celltype_specificity_zscores.csv", encoding="utf-8-sig")


def make_channel_perturbation(candidates: pd.DataFrame) -> None:
    top = candidates.head(20).copy()
    mean_columns = ["Score_Drop_Remove_PPI", "Score_Drop_Remove_CellTypeSimilarity", "Score_Drop_Remove_Pathway"]
    sd_columns = [f"{column}_SD" for column in mean_columns]
    missing = [column for column in sd_columns if column not in top]
    if missing:
        raise RuntimeError(f"Rerun 2.3_predict2.py before interpretation; missing columns: {missing}")
    plot = top.set_index("Gene_Symbol")[mean_columns]
    errors = top.set_index("Gene_Symbol")[sd_columns]
    plot.columns = errors.columns = ["STRING", "Cell-type similarity", "Pathway"]
    apply_bmc_style()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4.2))
    plot.plot(
        kind="bar", yerr=errors, capsize=1.5, ax=ax,
        color=["#0072B2", "#009E73", "#D55E00"], width=0.8,
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set(
        xlabel="Candidate gene",
        ylabel="Prediction-score change after channel removal",
        title="Mean ± SD channel perturbation across ensemble seeds",
    )
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    save_figure(fig, INTERPRET_DIR / "candidate_channel_perturbation")
    pd.concat({"Mean": plot, "SD_across_seeds": errors}, axis=1).to_csv(
        INTERPRET_DIR / "candidate_channel_perturbation.csv", encoding="utf-8-sig"
    )


def make_rankings(candidates: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    degree_file = VALIDATION_DIR / "unlabeled_candidates_with_network_degree.csv.gz"
    degree = pd.read_csv(degree_file)
    degree_columns = [
        "STRING_Degree",
        "STRING_Weighted_Degree",
        "Cell-type similarity_Degree",
        "Cell-type similarity_Weighted_Degree",
        "Pathway_Degree",
        "Pathway_Weighted_Degree",
    ]
    frame = candidates.merge(degree[["Node_ID", *degree_columns]], on="Node_ID", how="left", validate="one_to_one")
    x = np.log1p(frame[degree_columns].to_numpy(dtype=float))
    y = frame["Prediction_Score_Mean"].to_numpy(dtype=float)
    degree_model = make_pipeline(
        SplineTransformer(n_knots=5, degree=3, include_bias=False),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-4, 4, 25)),
    )
    degree_expected = degree_model.fit(x, y).predict(x)
    frame["Degree_Expected_Score"] = degree_expected
    frame["Degree_Residualized_Score"] = y - degree_expected

    hub_rank = frame["STRING_Degree"].rank(method="average", pct=True)
    structural = frame["Gene_Symbol"].astype(str).str.match(STRUCTURAL_PATTERN)
    hub = hub_rank > HUB_PERCENTILE
    reasons = np.where(hub & structural, "top-5%-STRING-degree;ribosomal-or-mitochondrial-structural", np.where(hub, "top-5%-STRING-degree", np.where(structural, "ribosomal-or-mitochondrial-structural", "")))
    audit = frame.loc[hub | structural, ["Node_ID", "Gene_Symbol", "Prediction_Score_Mean", "STRING_Degree"]].copy()
    audit["Removal_Reason"] = reasons[hub | structural]
    audit.to_csv(INTERPRET_DIR / "gsea_sensitivity_removed_genes.csv", index=False, encoding="utf-8-sig")

    rankings = {
        "primary": frame[["Gene_Symbol", "Prediction_Score_Mean"]].rename(columns={"Prediction_Score_Mean": "Score"}),
        "degree_residualized": frame[["Gene_Symbol", "Degree_Residualized_Score"]].rename(columns={"Degree_Residualized_Score": "Score"}),
        "hub_structural_filtered": frame.loc[~(hub | structural), ["Gene_Symbol", "Prediction_Score_Mean"]].rename(columns={"Prediction_Score_Mean": "Score"}),
    }
    for name, ranking in rankings.items():
        ranking = ranking.drop_duplicates("Gene_Symbol").sort_values("Score", ascending=False).reset_index(drop=True)
        rankings[name] = ranking
        ranking_file = INTERPRET_DIR / f"gsea_ranking_{name}.csv"
        unchanged = False
        if ranking_file.exists():
            previous = pd.read_csv(ranking_file)
            unchanged = (
                previous.shape == ranking.shape
                and previous["Gene_Symbol"].astype(str).equals(ranking["Gene_Symbol"].astype(str))
                and np.allclose(previous["Score"], ranking["Score"], rtol=1e-12, atol=1e-14, equal_nan=True)
            )
        if not unchanged:
            ranking.to_csv(ranking_file, index=False, encoding="utf-8-sig")
    metadata = {
        "degree_model_r_squared": float(degree_model.score(x, y)),
        "hub_definition": "STRING degree percentile > 0.95 among unlabeled candidates",
        "structural_gene_definition": STRUCTURAL_PATTERN.pattern,
        "removed_gene_count": int((hub | structural).sum()),
        "immune_genes_removed": False,
        "broad_housekeeping_list_removed": False,
        "reason": "Immune biology is relevant to LUAD and broad housekeeping removal lacks a justified versioned reference; both are retained.",
        "nonredundant_summary": (
            "Complete GSEA results are retained. Representative terms are selected separately within each library and NES direction "
            f"by greedy leading-edge overlap coefficient >= {LEADING_EDGE_OVERLAP_THRESHOLD:.2f}."
        ),
    }
    return rankings, audit, metadata


def run_gsea(ranking: pd.DataFrame, gene_sets: dict[str, list[str]]):
    return gp.prerank(
        rnk=ranking,
        gene_sets=gene_sets,
        min_size=10,
        max_size=500,
        permutation_num=PERMUTATIONS,
        threads=4,
        seed=42,
        outdir=None,
        verbose=False,
    )


def plot_gsea_bubble(table: pd.DataFrame, title: str, output_stem: str) -> None:
    fdr = pd.to_numeric(table["FDR q-val"], errors="coerce")
    nes = pd.to_numeric(table["NES"], errors="coerce")
    significant = table[(fdr < 0.05) & (nes > 0)].copy()
    if significant.empty:
        return
    significant["FDR"] = pd.to_numeric(significant["FDR q-val"], errors="coerce")
    significant["NES_numeric"] = pd.to_numeric(significant["NES"], errors="coerce")
    significant["minus_log10_FDR"] = -np.log10(significant["FDR"].clip(lower=1 / (PERMUTATIONS + 1)))
    significant["Hit_Count"] = significant["Lead_genes"].astype(str).str.split(";").map(len)
    significant["Display"] = significant["Term"].astype(str).str.split("::").str[-1].str.replace("_", " ")
    significant["Display"] = significant["Display"].map(lambda value: "\n".join(textwrap.wrap(value, 42)))
    top = significant.sort_values(["FDR", "NES_numeric"], ascending=[True, False]).head(10).sort_values("minus_log10_FDR")
    apply_bmc_style()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4.5))
    scatter = ax.scatter(top["minus_log10_FDR"], top["Display"], s=top["Hit_Count"] * 4, c=top["NES_numeric"], cmap="viridis", edgecolor="black", linewidth=0.3)
    fig.colorbar(scatter, ax=ax, label="NES")
    ax.set(xlabel=f"-log10(FDR); zero values shown at 1/{PERMUTATIONS + 1}", ylabel="", title=title)
    ax.grid(axis="x", color="0.9", linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, INTERPRET_DIR / output_stem)


def select_nonredundant_terms(table: pd.DataFrame) -> pd.DataFrame:
    """Greedily retain representative terms using leading-edge gene overlap.

    This changes only how significant terms are summarized; it does not alter the
    ranked gene list, enrichment statistics, or the complete GSEA output.
    """
    work = table.copy()
    work["FDR_numeric"] = pd.to_numeric(work["FDR q-val"], errors="coerce")
    work["NES_numeric"] = pd.to_numeric(work["NES"], errors="coerce")
    work = work[(work["FDR_numeric"] < 0.05) & work["NES_numeric"].notna()].copy()
    work["Library"] = work["Term"].astype(str).str.split("::").str[0]
    work["Direction"] = np.where(work["NES_numeric"] > 0, "positive", "negative")
    work["Abs_NES"] = work["NES_numeric"].abs()
    work["Leading_Edge_Set"] = work["Lead_genes"].fillna("").map(
        lambda value: {gene for gene in str(value).split(";") if gene}
    )
    representatives = []
    for (_, _), group in work.groupby(["Library", "Direction"], sort=False):
        group = group.sort_values(["FDR_numeric", "Abs_NES"], ascending=[True, False])
        clusters: list[dict] = []
        for _, row in group.iterrows():
            genes = row["Leading_Edge_Set"]
            matched = None
            for cluster in clusters:
                denominator = min(len(genes), len(cluster["genes"]))
                overlap = len(genes & cluster["genes"]) / denominator if denominator else 0.0
                if overlap >= LEADING_EDGE_OVERLAP_THRESHOLD:
                    matched = cluster
                    break
            if matched is None:
                clusters.append({"representative": row.copy(), "genes": genes, "terms": [row["Term"]]})
            else:
                matched["terms"].append(row["Term"])
        for cluster in clusters:
            row = cluster["representative"]
            row["Cluster_Size"] = len(cluster["terms"])
            row["Collapsed_Terms"] = " | ".join(map(str, cluster["terms"]))
            representatives.append(row)
    if not representatives:
        return work.drop(columns=["Leading_Edge_Set"], errors="ignore")
    result = pd.DataFrame(representatives).drop(columns=["Leading_Edge_Set"], errors="ignore")
    result["FDR_for_plot"] = result["FDR_numeric"].clip(lower=1 / (PERMUTATIONS + 1))
    result["FDR_reporting_note"] = np.where(
        result["FDR_numeric"] == 0,
        f"raw q=0; plot uses floor 1/{PERMUTATIONS + 1}",
        "raw q retained",
    )
    return result.sort_values(["FDR_numeric", "Abs_NES"], ascending=[True, False])


def _gsea_output(name: str):
    return INTERPRET_DIR / f"unlabeled_candidate_gsea_{name}.csv"


def _cache_is_current(name: str) -> bool:
    output = _gsea_output(name)
    ranking = INTERPRET_DIR / f"gsea_ranking_{name}.csv"
    return output.exists() and ranking.exists() and output.stat().st_mtime >= ranking.stat().st_mtime


def make_enrichment(candidates: pd.DataFrame, selected_analysis: str = "all", force: bool = False) -> None:
    rankings, _, metadata = make_rankings(candidates)
    needs_gene_sets = any(
        selected_analysis in {"all", name} and (force or not _cache_is_current(name))
        for name in rankings
    )
    gene_sets = download_gene_sets() if needs_gene_sets else {}
    results = {}
    summary_rows = []
    for name, ranking in rankings.items():
        should_run = selected_analysis in {"all", name}
        if not should_run and not _cache_is_current(name):
            continue
        if should_run and (force or not _cache_is_current(name)):
            result = run_gsea(ranking, gene_sets)
            table = result.res2d.copy()
            results[name] = result
            table.to_csv(_gsea_output(name), index=False, encoding="utf-8-sig")
        else:
            table = pd.read_csv(_gsea_output(name))
        if name == "primary":
            table.to_csv(INTERPRET_DIR / "unlabeled_candidate_gsea.csv", index=False, encoding="utf-8-sig")
        fdr = pd.to_numeric(table["FDR q-val"], errors="coerce")
        nes = pd.to_numeric(table["NES"], errors="coerce")
        representatives = select_nonredundant_terms(table)
        representatives.to_csv(
            INTERPRET_DIR / f"gsea_nonredundant_representatives_{name}.csv", index=False, encoding="utf-8-sig"
        )
        rep_positive = int((representatives.get("Direction", pd.Series(dtype=str)) == "positive").sum())
        rep_negative = int((representatives.get("Direction", pd.Series(dtype=str)) == "negative").sum())
        library = table["Term"].astype(str).str.split("::").str[0]
        summary_rows.append({
            "Analysis": name,
            "Ranked_Genes": len(ranking),
            "Positive_FDR_lt_0.05": int(((fdr < 0.05) & (nes > 0)).sum()),
            "Negative_FDR_lt_0.05": int(((fdr < 0.05) & (nes < 0)).sum()),
            "Hallmark_Positive_FDR_lt_0.05": int(((fdr < 0.05) & (nes > 0) & (library == "MSigDB_Hallmark_2020")).sum()),
            "GO_BP_Positive_FDR_lt_0.05": int(((fdr < 0.05) & (nes > 0) & (library == "GO_Biological_Process_2021")).sum()),
            "Nonredundant_Positive_Representatives": rep_positive,
            "Nonredundant_Negative_Representatives": rep_negative,
            "Zero_FDR_Count": int((fdr == 0).sum()),
        })
        hallmark = representatives[representatives.get("Library", pd.Series(index=representatives.index, dtype=str)) == "MSigDB_Hallmark_2020"].copy()
        go_bp = representatives[representatives.get("Library", pd.Series(index=representatives.index, dtype=str)) == "GO_Biological_Process_2021"].copy()
        hallmark.to_csv(
            INTERPRET_DIR / f"gsea_main_hallmark_{name}.csv", index=False, encoding="utf-8-sig"
        )
        go_bp.to_csv(
            INTERPRET_DIR / f"gsea_supplement_go_nonredundant_{name}.csv", index=False, encoding="utf-8-sig"
        )
        output_stem = "unlabeled_candidate_gsea_bubble" if name == "primary" else f"gsea_hallmark_{name}"
        plot_gsea_bubble(
            hallmark,
            f"Hallmark GSEA: {name.replace('_', ' ')} ranking",
            output_stem,
        )
        plot_gsea_bubble(
            go_bp,
            f"Nonredundant GO-BP GSEA: {name.replace('_', ' ')} ranking",
            f"gsea_go_bp_supplement_{name}",
        )
    pd.DataFrame(summary_rows).to_csv(INTERPRET_DIR / "gsea_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    (INTERPRET_DIR / "gsea_sensitivity_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if "primary" in results:
        primary = results["primary"]
        primary_table = primary.res2d.copy()
        fdr = pd.to_numeric(primary_table["FDR q-val"], errors="coerce")
        nes = pd.to_numeric(primary_table["NES"], errors="coerce")
        library = primary_table["Term"].astype(str).str.split("::").str[0]
        terms = primary_table[(fdr < 0.05) & (nes > 0) & (library == "MSigDB_Hallmark_2020")].sort_values("FDR q-val").head(2)["Term"]
        for index, term in enumerate(terms):
            if term in primary.results:
                apply_bmc_style()
                axes = gp.plot.gseaplot(rank_metric=primary.ranking, term=term, **primary.results[term])
                first_axis = axes[0] if isinstance(axes, (list, tuple)) else axes
                fig = first_axis.figure
                fig.set_size_inches(FULL_WIDTH, 3.8)
                save_figure(fig, INTERPRET_DIR / f"gsea_curve_{index + 1}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run primary and post hoc GSEA sensitivity analyses.")
    parser.add_argument(
        "--analysis",
        choices=("all", "primary", "degree_residualized", "hub_structural_filtered"),
        default="all",
        help="Run one analysis or all analyses; current cached results are reused.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute the selected analysis even if its cache is current.")
    args = parser.parse_args()
    ensure_result_dirs()
    candidates = pd.read_csv(MODEL_DIR / "unlabeled_candidate_ranking.csv")
    robust_file = ROBUSTNESS_DIR / "robust_top20.csv"
    if not robust_file.exists():
        raise FileNotFoundError("Run 4.1_robustness.py before 3.2_analysis3.py")
    robust_candidates = pd.read_csv(robust_file).sort_values("Robust_Candidate_Rank")
    make_celltype_heatmap(robust_candidates)
    make_channel_perturbation(robust_candidates)
    make_enrichment(candidates, selected_analysis=args.analysis, force=args.force)
    (INTERPRET_DIR / "analysis_scope.txt").write_text(
        "Primary enrichment uses all unlabeled candidates. Post hoc sensitivity analyses use degree-residualized scores and a diagnostic filter for the top 5% of STRING-degree genes plus ribosomal/mitochondrial structural genes; these analyses were not preregistered or prespecified. The main enrichment figure is restricted to Hallmark terms; nonredundant GO-BP representatives and complete tables are supplementary. Immune genes and broad housekeeping genes are not removed. KEGG is excluded because it was used to construct the pathway graph. The 15 TISCH columns are aggregated cell-type profiles, not individual cells.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
