from __future__ import annotations

import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from common import network_degree_features, torch_load
from config import (
    BENCHMARK_DIR,
    BOOTSTRAP_REPLICATES,
    FINAL_ENSEMBLE_SEEDS,
    MODEL_DIR,
    RAW_DATASET_FILE,
    ROBUSTNESS_DIR,
    SEED,
    VALIDATION_DIR,
    ensure_result_dirs,
)
from models import MultiViewGCN
from plot_style import FULL_WIDTH, apply_bmc_style, save_figure


PROPOSED = "ctMV-GCN (local attention + class-weighted CE)"
COMPARATORS = [PROPOSED, "Network-degree logistic", "RWR (STRING graph)"]
DISPLAY = {
    PROPOSED: "ctMV-GCN",
    "Network-degree logistic": "Degree logistic",
    "RWR (STRING graph)": "RWR",
}
STRUCTURAL_PATTERN = re.compile(r"^(?:RPL\d|RPS\d|MRPL\d|MRPS\d|MT-)", re.IGNORECASE)
CANONICAL_HOUSEKEEPING = {
    "ACTB", "B2M", "GAPDH", "GUSB", "HMBS", "HPRT1", "PGK1", "PPIA", "RPLP0", "TBP", "TFRC", "YWHAZ"
}
EDGE_DROPOUT_FRACTIONS = (0.05, 0.10, 0.20)
EDGE_DROPOUT_REPLICATES = 10


def degree_strata(degree: np.ndarray) -> np.ndarray:
    """Create up to ten empirical bins; tied zero degrees can reduce the count."""
    bins = pd.qcut(pd.Series(np.log1p(degree)), q=10, labels=False, duplicates="drop")
    return bins.fillna(0).astype(int).to_numpy()


def safe_metrics(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    if np.unique(y).size < 2:
        return np.nan, np.nan
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def degree_stratified_results(oof: pd.DataFrame, labels: np.ndarray, strata: np.ndarray) -> pd.DataFrame:
    rows = []
    for (repeat, method), part in oof[oof["method"].isin(COMPARATORS)].groupby(["repeat", "method"]):
        part = part.sort_values("node_id")
        node_ids = part["node_id"].to_numpy(dtype=int)
        for stratum in np.unique(strata):
            keep = strata[node_ids] == stratum
            y = labels[node_ids[keep]]
            roc_auc, auprc = safe_metrics(y, part.loc[keep, "score"].to_numpy())
            rows.append({
                "Repeat": int(repeat),
                "Method": method,
                "STRING_Degree_Stratum": int(stratum) + 1,
                "N": int(keep.sum()),
                "Positives": int(y.sum()),
                "ROC_AUC": roc_auc,
                "AUPRC": auprc,
                "Prevalence": float(y.mean()) if len(y) else np.nan,
            })
    return pd.DataFrame(rows)


def matched_bootstrap(oof: pd.DataFrame, labels: np.ndarray, strata: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivots = {
        int(repeat): part.pivot(index="node_id", columns="method", values="score").sort_index()
        for repeat, part in oof[oof["method"].isin(COMPARATORS)].groupby("repeat")
    }
    repeats = sorted(pivots)
    rng = np.random.default_rng(SEED)
    samples = {method: {"ROC_AUC": [], "AUPRC": []} for method in COMPARATORS}
    deltas = {method: [] for method in COMPARATORS if method != PROPOSED}
    for _ in range(BOOTSTRAP_REPLICATES):
        selected_repeats = rng.choice(repeats, size=len(repeats), replace=True)
        replicate = {method: [] for method in COMPARATORS}
        for repeat in selected_repeats:
            pivot = pivots[int(repeat)]
            nodes = pivot.index.to_numpy(dtype=int)
            y = labels[nodes]
            local_strata = strata[nodes]
            selected = []
            for stratum in np.unique(local_strata):
                pos = np.flatnonzero((local_strata == stratum) & (y == 1))
                neg = np.flatnonzero((local_strata == stratum) & (y == 0))
                if len(pos) and len(neg):
                    # Resample both classes. Keeping every positive fixed makes
                    # the nominal bootstrap interval anti-conservative.
                    selected.extend(rng.choice(pos, size=len(pos), replace=True).tolist())
                    selected.extend(rng.choice(neg, size=len(pos), replace=True).tolist())
            selected = np.asarray(selected, dtype=int)
            rng.shuffle(selected)
            for method in COMPARATORS:
                replicate[method].append(safe_metrics(y[selected], pivot[method].to_numpy()[selected]))
        means = {}
        for method in COMPARATORS:
            values = np.asarray(replicate[method], dtype=float)
            means[method] = values.mean(axis=0)
            samples[method]["ROC_AUC"].append(means[method][0])
            samples[method]["AUPRC"].append(means[method][1])
        for method in deltas:
            deltas[method].append(means[PROPOSED][1] - means[method][1])

    summary_rows = []
    for method in COMPARATORS:
        for metric in ("ROC_AUC", "AUPRC"):
            values = np.asarray(samples[method][metric])
            summary_rows.append({
                "Method": method,
                "Metric": metric,
                "Bootstrap_Median": float(np.median(values)),
                "CI_Lower": float(np.quantile(values, 0.025)),
                "CI_Upper": float(np.quantile(values, 0.975)),
                "Bootstrap_Replicates": BOOTSTRAP_REPLICATES,
                "Matching": "1:1 positive-negative within empirical STRING-degree stratum; both classes sampled with replacement",
                "Matched_Positive_Fraction": 0.5,
                "Random_Baseline": 0.5,
                "Original_Cohort_Positive_Fraction": float(labels.mean()),
                "Comparability_Note": "Matched AUPRC is conditional on 50% prevalence and is not directly comparable with AUPRC from the original imbalanced cohort.",
            })
    comparison_rows = []
    for method, values in deltas.items():
        values = np.asarray(values)
        p_lower = (np.count_nonzero(values <= 0) + 1) / (len(values) + 1)
        p_upper = (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)
        comparison_rows.append({
            "Compared_Method": method,
            "AUPRC_Delta_GNN_Minus_Comparator_Median": float(np.median(values)),
            "CI_Lower": float(np.quantile(values, 0.025)),
            "CI_Upper": float(np.quantile(values, 0.975)),
            "Two_Sided_Bootstrap_P": float(min(1.0, 2 * min(p_lower, p_upper))),
            "Matched_Positive_Fraction": 0.5,
            "Comparability_Note": "This paired delta is valid because both methods were evaluated on the same matched bootstrap samples.",
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(comparison_rows)


def degree_conditional_permutation(
    oof: pd.DataFrame, labels: np.ndarray, strata: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    proposed = oof[oof["method"] == PROPOSED].pivot(index="node_id", columns="repeat", values="score").sort_index()
    nodes = proposed.index.to_numpy(dtype=int)
    score = proposed.mean(axis=1).to_numpy()
    y = labels[nodes]
    local_strata = strata[nodes]

    def stratified_auc(outcome: np.ndarray) -> float:
        values, weights = [], []
        for stratum in np.unique(local_strata):
            keep = local_strata == stratum
            if np.unique(outcome[keep]).size == 2:
                values.append(roc_auc_score(outcome[keep], score[keep]))
                weights.append(keep.sum())
        return float(np.average(values, weights=weights))

    observed = stratified_auc(y)
    rng = np.random.default_rng(SEED)
    null = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        permuted = y.copy()
        for stratum in np.unique(local_strata):
            keep = np.flatnonzero(local_strata == stratum)
            permuted[keep] = rng.permutation(permuted[keep])
        null[index] = stratified_auc(permuted)
    p_value = (np.count_nonzero(null >= observed) + 1) / (len(null) + 1)
    summary = pd.DataFrame([{
        "Observed_Degree_Stratified_ROC_AUC": observed,
        "Null_Mean": float(null.mean()),
        "Null_SD": float(null.std(ddof=1)),
        "Null_95pct_Lower": float(np.quantile(null, 0.025)),
        "Null_95pct_Upper": float(np.quantile(null, 0.975)),
        "One_Sided_P": float(p_value),
        "Permutations": BOOTSTRAP_REPLICATES,
    }])
    return summary, pd.DataFrame({"Permutation": np.arange(1, len(null) + 1), "Null_Stratified_ROC_AUC": null})


def residualized_gnn_score(
    oof: pd.DataFrame, labels: np.ndarray, degree_features: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    proposed = oof[oof["method"] == PROPOSED].pivot(index="node_id", columns="repeat", values="score").sort_index()
    nodes = proposed.index.to_numpy(dtype=int)
    score = proposed.mean(axis=1).to_numpy()
    x = degree_features[nodes]
    model = make_pipeline(
        SplineTransformer(n_knots=5, degree=3, include_bias=False),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-4, 4, 25)),
    )
    expected = model.fit(x, score).predict(x)
    residual = score - expected
    y = labels[nodes]
    rows = []
    for name, values in (("Raw GNN OOF score", score), ("Degree-residualized GNN OOF score", residual)):
        roc_auc, auprc = safe_metrics(y, values)
        rows.append({"Score": name, "ROC_AUC": roc_auc, "AUPRC": auprc})
    annotated = pd.DataFrame({
        "Node_ID": nodes,
        "Clinical_Target_Label": y,
        "GNN_OOF_Score_Mean": score,
        "Degree_Expected_GNN_Score": expected,
        "Degree_Residualized_GNN_Score": residual,
    })
    return pd.DataFrame(rows), annotated


def drop_undirected_edges(data, edge_name: str, weight_name: str, fraction: float, rng: np.random.Generator) -> None:
    edge_index = getattr(data, edge_name).cpu().numpy()
    edge_weight = getattr(data, weight_name).cpu().numpy()
    low = np.minimum(edge_index[0], edge_index[1]).astype(np.int64)
    high = np.maximum(edge_index[0], edge_index[1]).astype(np.int64)
    keys = low * int(data.num_nodes) + high
    _, inverse = np.unique(keys, return_inverse=True)
    pair_count = int(inverse.max()) + 1
    keep_pairs = rng.random(pair_count) >= fraction
    keep = keep_pairs[inverse]
    setattr(data, edge_name, torch.tensor(edge_index[:, keep], dtype=torch.long))
    setattr(data, weight_name, torch.tensor(edge_weight[keep], dtype=torch.float))


def topology_robustness(labels: np.ndarray) -> pd.DataFrame:
    data = torch_load(MODEL_DIR / "GNN_Dataset_Final.pt")
    model = MultiViewGCN(in_channels=data.x.shape[1], attention_type="local")
    seed = FINAL_ENSEMBLE_SEEDS[0]
    model.load_state_dict(torch.load(MODEL_DIR / f"ctMV_GCN_local_weightedCE_seed{seed}.pth", map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        baseline = F.softmax(model(data), dim=1)[:, 1].cpu().numpy()
    unlabeled = labels == 0
    unlabeled_ids = np.flatnonzero(unlabeled)
    baseline_order = unlabeled_ids[np.argsort(-baseline[unlabeled])]
    baseline_top100 = set(baseline_order[:100])
    baseline_top500 = set(baseline_order[:500])
    rows = []
    for fraction in EDGE_DROPOUT_FRACTIONS:
        for replicate in range(EDGE_DROPOUT_REPLICATES):
            rng = np.random.default_rng(SEED + int(fraction * 1000) + replicate)
            perturbed = data.clone()
            for edge_name, weight_name in (
                ("edge_index_ppi", "edge_weight_ppi"),
                ("edge_index_coexp", "edge_weight_coexp"),
                ("edge_index_pathway", "edge_weight_pathway"),
            ):
                drop_undirected_edges(perturbed, edge_name, weight_name, fraction, rng)
            with torch.no_grad():
                score = F.softmax(model(perturbed), dim=1)[:, 1].cpu().numpy()
            order = unlabeled_ids[np.argsort(-score[unlabeled])]
            top100 = set(order[:100])
            top500 = set(order[:500])
            rows.append({
                "Edge_Dropout_Fraction": fraction,
                "Replicate": replicate + 1,
                "Spearman_Rho_All_Unlabeled": float(spearmanr(baseline[unlabeled], score[unlabeled]).statistic),
                "Median_Absolute_Score_Change": float(np.median(np.abs(baseline[unlabeled] - score[unlabeled]))),
                "Top100_Jaccard": len(baseline_top100 & top100) / len(baseline_top100 | top100),
                "Top500_Jaccard": len(baseline_top500 & top500) / len(baseline_top500 | top500),
                "Model_Seed": seed,
                "Inference_Only": True,
            })
    return pd.DataFrame(rows)


def build_robust_candidates(predictions: pd.DataFrame, degree: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"Rank_SD_Across_Seeds", "Top500_Frequency_Across_Seeds"}
    missing = required - set(predictions.columns)
    if missing:
        raise RuntimeError(f"Rerun 2.3_predict2.py before robustness analysis; missing columns: {sorted(missing)}")
    degree_columns = [
        "STRING_Degree", "STRING_Weighted_Degree", "Cell-type similarity_Degree",
        "Cell-type similarity_Weighted_Degree", "Pathway_Degree", "Pathway_Weighted_Degree",
    ]
    frame = predictions[predictions["Clinical_Target_Label"] == 0].merge(
        degree[["Node_ID", *degree_columns]], on="Node_ID", how="left", validate="one_to_one"
    )
    x = np.log1p(frame[degree_columns].to_numpy(dtype=float))
    y = frame["Prediction_Score_Mean"].to_numpy(dtype=float)
    model = make_pipeline(
        SplineTransformer(n_knots=5, degree=3, include_bias=False),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-4, 4, 25)),
    )
    expected = model.fit(x, y).predict(x)
    frame["Degree_Residualized_Score"] = y - expected
    frame["STRING_Degree_Percentile"] = frame["STRING_Degree"].rank(method="average", pct=True)
    frame["Is_Top5pct_STRING_Hub"] = frame["STRING_Degree_Percentile"] > 0.95
    frame["Is_Ribosomal_or_Mitochondrial_Structural"] = frame["Gene_Symbol"].astype(str).str.match(STRUCTURAL_PATTERN)
    frame["Is_Canonical_Housekeeping_Panel"] = frame["Gene_Symbol"].isin(CANONICAL_HOUSEKEEPING)
    frame["Robust_Eligibility"] = ~(
        frame["Is_Top5pct_STRING_Hub"]
        | frame["Is_Ribosomal_or_Mitochondrial_Structural"]
        | frame["Is_Canonical_Housekeeping_Panel"]
    )
    frame["Residual_Score_Percentile"] = frame["Degree_Residualized_Score"].rank(method="average", pct=True)
    frame["Rank_Stability_Percentile"] = 1.0 - frame["Rank_SD_Across_Seeds"].rank(method="average", pct=True)
    frame["Robustness_Score"] = (
        0.50 * frame["Residual_Score_Percentile"]
        + 0.25 * frame["Top500_Frequency_Across_Seeds"]
        + 0.25 * frame["Rank_Stability_Percentile"]
    )
    frame = frame.sort_values(["Robust_Eligibility", "Robustness_Score"], ascending=[False, False]).reset_index(drop=True)
    eligible = frame[frame["Robust_Eligibility"]].copy().sort_values("Robustness_Score", ascending=False).reset_index(drop=True)
    eligible.insert(0, "Robust_Candidate_Rank", np.arange(1, len(eligible) + 1))
    top20 = eligible.head(20).copy()
    frame.to_csv(ROBUSTNESS_DIR / "candidate_robustness_audit.csv.gz", index=False, compression="gzip")
    eligible.to_csv(ROBUSTNESS_DIR / "robust_candidate_ranking.csv", index=False, encoding="utf-8-sig")
    top20.to_csv(ROBUSTNESS_DIR / "robust_top20.csv", index=False, encoding="utf-8-sig")

    raw_top20 = frame.sort_values("Prediction_Score_Mean", ascending=False).head(20).copy()
    raw_top20 = raw_top20[[
        "Gene_Symbol", "Prediction_Score_Mean", "STRING_Degree",
        "Is_Top5pct_STRING_Hub", "Is_Ribosomal_or_Mitochondrial_Structural",
        "Is_Canonical_Housekeeping_Panel", "Robust_Eligibility",
    ]].rename(columns={
        "Gene_Symbol": "Raw_Gene", "Prediction_Score_Mean": "Raw_Prediction_Score",
    }).reset_index(drop=True)
    raw_top20.insert(0, "Raw_Rank", np.arange(1, len(raw_top20) + 1))
    robust_view = top20[[
        "Robust_Candidate_Rank", "Gene_Symbol", "Robustness_Score",
        "Degree_Residualized_Score", "STRING_Degree",
    ]].rename(columns={
        "Gene_Symbol": "Robust_Gene", "STRING_Degree": "Robust_STRING_Degree",
    }).reset_index(drop=True)
    pd.concat([raw_top20, robust_view], axis=1).to_csv(
        ROBUSTNESS_DIR / "raw_vs_robust_top20.csv", index=False, encoding="utf-8-sig"
    )
    manifest = {
        "selection_timing": "Post hoc robustness analysis developed after diagnosing network-degree bias; not preregistered or prespecified.",
        "eligibility_exclusions": [
            "top 5% STRING degree among unlabeled genes",
            "ribosomal or mitochondrial structural gene symbol pattern",
            "12-gene canonical technical housekeeping panel",
        ],
        "immune_gene_exclusion": False,
        "immune_gene_reason": "Immune biology is directly relevant to lung adenocarcinoma; blanket removal would be biologically unjustified.",
        "score_formula": "0.50*degree-residual percentile + 0.25*top-500 seed frequency + 0.25*inverse rank-SD percentile",
        "external_results_used_for_selection": False,
        "canonical_housekeeping_panel": sorted(CANONICAL_HOUSEKEEPING),
    }
    (ROBUSTNESS_DIR / "robust_candidate_selection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return frame, top20


def make_figures(stratified: pd.DataFrame, topology: pd.DataFrame, top20: pd.DataFrame) -> None:
    apply_bmc_style()
    summary = stratified.groupby(["Method", "STRING_Degree_Stratum"])["AUPRC"].agg(["mean", "std"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.0))
    colors = ["#0072B2", "#D55E00", "#009E73"]
    for method, color in zip(COMPARATORS, colors):
        part = summary[summary["Method"] == method]
        axes[0].errorbar(part["STRING_Degree_Stratum"], part["mean"], yerr=part["std"].fillna(0), marker="o", markersize=3, capsize=2, color=color, label=DISPLAY[method])
    axes[0].set(title="a  Performance within degree strata", xlabel="Empirical STRING-degree stratum", ylabel="AUPRC")
    axes[0].grid(color="0.9", linewidth=0.5)
    axes[0].legend(frameon=False)
    topo = topology.groupby("Edge_Dropout_Fraction")[["Spearman_Rho_All_Unlabeled", "Top100_Jaccard"]].agg(["mean", "std"])
    x = topo.index.to_numpy() * 100
    axes[1].errorbar(x, topo[("Spearman_Rho_All_Unlabeled", "mean")], yerr=topo[("Spearman_Rho_All_Unlabeled", "std")], marker="o", capsize=2, label="Rank correlation")
    axes[1].errorbar(x, topo[("Top100_Jaccard", "mean")], yerr=topo[("Top100_Jaccard", "std")], marker="s", capsize=2, label="Top-100 Jaccard")
    axes[1].set(title="b  Edge-dropout sensitivity", xlabel="Edges removed from each view (%)", ylabel="Stability", ylim=(0, 1.02))
    axes[1].grid(color="0.9", linewidth=0.5)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, ROBUSTNESS_DIR / "degree_and_topology_robustness")

    plot = top20.sort_values("Robustness_Score")
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.72, 4.2))
    ax.barh(plot["Gene_Symbol"], plot["Robustness_Score"], color="#0072B2")
    ax.set(xlabel="Robustness score", ylabel="", title="Robust ctMV-GCN candidate shortlist")
    ax.grid(axis="x", color="0.9", linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, ROBUSTNESS_DIR / "robust_top20_candidates")


def main() -> None:
    ensure_result_dirs()
    data = torch_load(RAW_DATASET_FILE)
    labels = data.y.cpu().numpy()
    degree_features = network_degree_features(data)
    string_degree = np.expm1(degree_features[:, 0])
    strata = degree_strata(string_degree)
    oof = pd.read_csv(BENCHMARK_DIR / "oof_predictions.csv.gz")

    stratified = degree_stratified_results(oof, labels, strata)
    stratified.to_csv(ROBUSTNESS_DIR / "degree_stratified_oof_metrics.csv", index=False, encoding="utf-8-sig")
    matched, matched_comparisons = matched_bootstrap(oof, labels, strata)
    matched.to_csv(ROBUSTNESS_DIR / "degree_matched_bootstrap_metrics.csv", index=False, encoding="utf-8-sig")
    matched_comparisons.to_csv(ROBUSTNESS_DIR / "degree_matched_auprc_comparisons.csv", index=False, encoding="utf-8-sig")
    (ROBUSTNESS_DIR / "degree_matched_design.json").write_text(json.dumps({
        "design": "Within each empirical STRING-degree stratum, positives and an equal number of negatives are resampled with replacement.",
        "matched_positive_fraction": 0.5,
        "random_auprc": 0.5,
        "original_positive_fraction": float(labels.mean()),
        "interpretation": "Matched AUPRC quantifies discrimination after degree matching. It must not be numerically compared with AUPRC from the original imbalanced evaluation; only within-matched-sample method differences are directly comparable.",
    }, indent=2), encoding="utf-8")
    permutation, null = degree_conditional_permutation(oof, labels, strata)
    permutation.to_csv(ROBUSTNESS_DIR / "degree_conditional_permutation_test.csv", index=False, encoding="utf-8-sig")
    null.to_csv(ROBUSTNESS_DIR / "degree_conditional_permutation_null.csv.gz", index=False, compression="gzip")
    residual_metrics, residual_scores = residualized_gnn_score(oof, labels, degree_features)
    residual_metrics.to_csv(ROBUSTNESS_DIR / "degree_residualized_oof_metrics.csv", index=False, encoding="utf-8-sig")
    residual_scores.to_csv(ROBUSTNESS_DIR / "degree_residualized_oof_scores.csv.gz", index=False, compression="gzip")

    topology = topology_robustness(labels)
    topology.to_csv(ROBUSTNESS_DIR / "edge_dropout_rank_stability.csv", index=False, encoding="utf-8-sig")
    predictions = pd.read_csv(MODEL_DIR / "all_gene_predictions.csv.gz")
    degree = pd.read_csv(VALIDATION_DIR / "unlabeled_candidates_with_network_degree.csv.gz")
    _, top20 = build_robust_candidates(predictions, degree)
    make_figures(stratified, topology, top20)
    (ROBUSTNESS_DIR / "robustness_interpretation_note.txt").write_text(
        "Degree matching and conditional permutation test whether GNN scores retain label information within STRING-degree strata. "
        "The 1:1 matched analysis has 50% positive prevalence and random AUPRC 0.5, so its AUPRC values are not directly comparable with the original imbalanced benchmark. "
        "Degree residualization removes nonlinear score structure explained by all six graph-degree features. Edge dropout is an inference-time sensitivity analysis, not a retrained performance benchmark. "
        "The robustness rules are post hoc and were developed after network-degree bias was diagnosed; they were not preregistered or prespecified. "
        "Raw and robust top-20 lists are reported side by side. Immune genes were not blanket-filtered because immune biology is relevant to LUAD.\n",
        encoding="utf-8",
    )
    print("Degree-matched results\n", matched.to_string(index=False))
    print("\nDegree-matched comparisons\n", matched_comparisons.to_string(index=False))
    print("\nConditional permutation\n", permutation.to_string(index=False))
    print("\nDegree residualization\n", residual_metrics.to_string(index=False))
    print("\nRobust top 20\n", top20[["Robust_Candidate_Rank", "Gene_Symbol", "Robustness_Score"]].to_string(index=False))


if __name__ == "__main__":
    main()
