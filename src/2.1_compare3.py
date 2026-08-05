from __future__ import annotations

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import choose_f1_threshold, network_degree_features, seed_everything, stratified_bootstrap_indices, torch_load
from config import (
    BENCHMARK_DIR,
    BOOTSTRAP_REPLICATES,
    INNER_VAL_FRACTION,
    N_REPEATS,
    N_SPLITS,
    RAW_DATASET_FILE,
    SEED,
    ensure_result_dirs,
)
from experiment import run_rwr, train_graph_model, transform_graph_features
from models import MultiViewGCN, PpiGAT, PpiGCN, PpiGraphSAGE
from plot_style import FULL_WIDTH, apply_bmc_style, save_figure


PROPOSED = "scTDA-GCN (local attention + weighted BCE)"
DISPLAY_NAMES = {
    PROPOSED: "Local attention + weighted BCE",
    "scTDA-GCN (local attention + nnPU)": "Local attention + nnPU",
    "scTDA-GCN (dual attention + nnPU)": "Dual attention + nnPU",
    "scTDA-GCN (equal fusion + nnPU)": "Equal fusion + nnPU",
    "PPI-GCN + nnPU": "PPI-GCN + nnPU",
    "PPI-GraphSAGE + nnPU": "PPI-GraphSAGE + nnPU",
    "PPI-GAT + nnPU": "PPI-GAT + nnPU",
    "Logistic regression": "Logistic regression",
    "Random forest": "Random forest",
    "Network-degree logistic": "Network-degree logistic",
    "RWR (STRING graph)": "RWR (STRING graph)",
}


def write_benchmark_design() -> None:
    (BENCHMARK_DIR / "benchmark_design.json").write_text(json.dumps({
        "evaluation_type": "transductive node classification",
        "label_separation": "Outer-test labels are hidden from model fitting, early stopping, threshold selection, and RWR seed construction.",
        "graph_visibility": "All nodes and disease-independent graph topology are present during message passing; this is not inductive evaluation on a new graph or cohort.",
        "outer_validation": f"RepeatedStratifiedKFold with {N_SPLITS} folds and {N_REPEATS} repeats.",
        "inner_validation": f"{INNER_VAL_FRACTION:.0%} of each outer-training fold for early stopping and threshold selection.",
        "claim_boundary": "Results support transductive prioritization within the assembled gene graph, not generalization to an unseen graph.",
    }, indent=2), encoding="utf-8")
METHODS = {
    PROPOSED: (MultiViewGCN, {"attention_type": "local"}, False),
    "scTDA-GCN (dual attention + nnPU)": (MultiViewGCN, {"attention_type": "dual"}, True),
    "scTDA-GCN (equal fusion + nnPU)": (MultiViewGCN, {"attention_type": "equal"}, True),
    "PPI-GCN + nnPU": (PpiGCN, {}, True),
    "PPI-GraphSAGE + nnPU": (PpiGraphSAGE, {}, True),
    "PPI-GAT + nnPU": (PpiGAT, {}, True),
}


def calculate_metrics(y_true, score, threshold):
    pred = score >= threshold
    return {
        "ROC_AUC": roc_auc_score(y_true, score),
        "AUPRC": average_precision_score(y_true, score),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "Threshold": threshold,
    }


def bootstrap_statistics(oof: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    repeat_values = sorted(oof["repeat"].unique())
    pivots = {
        repeat: oof[oof["repeat"] == repeat].pivot(index="node_id", columns="method", values="score").sort_index()
        for repeat in repeat_values
    }
    methods = sorted(set.intersection(*(set(pivot.columns) for pivot in pivots.values())))
    rng = np.random.default_rng(SEED)
    observed = {
        method: float(np.mean([
            average_precision_score(labels[pivot.index.to_numpy()], pivot[method])
            for pivot in pivots.values()
        ]))
        for method in methods
    }
    samples = {m: [] for m in methods}
    differences = {m: [] for m in methods if m != PROPOSED}
    for _ in range(BOOTSTRAP_REPLICATES):
        selected_repeats = rng.choice(repeat_values, size=len(repeat_values), replace=True)
        replicate_scores = {method: [] for method in methods}
        for repeat in selected_repeats:
            pivot = pivots[int(repeat)]
            y = labels[pivot.index.to_numpy()]
            idx = stratified_bootstrap_indices(y, rng)
            for method in methods:
                replicate_scores[method].append(
                    average_precision_score(y[idx], pivot[method].to_numpy()[idx])
                )
        replicate_means = {method: float(np.mean(values)) for method, values in replicate_scores.items()}
        for method in methods:
            samples[method].append(replicate_means[method])
            if method != PROPOSED:
                differences[method].append(replicate_means[PROPOSED] - replicate_means[method])
    rows = []
    for method in methods:
        values = np.asarray(samples[method])
        row = {
            "Method": method,
            "Observed_AUPRC": observed[method],
            "AUPRC_CI_Lower": np.quantile(values, 0.025),
            "AUPRC_CI_Upper": np.quantile(values, 0.975),
            "Compared_to": PROPOSED,
            "Bootstrap_Replicates": BOOTSTRAP_REPLICATES,
            "Repeat_Aware": True,
        }
        if method == PROPOSED:
            row.update({"AUPRC_Difference_ProposedMinusMethod": 0.0, "Difference_CI_Lower": 0.0, "Difference_CI_Upper": 0.0, "Bootstrap_P": 1.0})
        else:
            delta = np.asarray(differences[method])
            lower_tail = (np.count_nonzero(delta <= 0) + 1) / (BOOTSTRAP_REPLICATES + 1)
            upper_tail = (np.count_nonzero(delta >= 0) + 1) / (BOOTSTRAP_REPLICATES + 1)
            row.update({
                "AUPRC_Difference_ProposedMinusMethod": observed[PROPOSED] - observed[method],
                "Difference_CI_Lower": np.quantile(delta, 0.025),
                "Difference_CI_Upper": np.quantile(delta, 0.975),
                "Bootstrap_P": min(1.0, 2 * min(lower_tail, upper_tail)),
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Observed_AUPRC", ascending=False)


def make_curves(oof: pd.DataFrame, labels: np.ndarray, methods: list[str], output_name: str) -> None:
    apply_bmc_style()
    primary = oof[(oof["repeat"] == 0) & oof["method"].isin(methods)]
    available = set(primary["method"].unique())
    missing = [method for method in methods if method not in available]
    if missing:
        raise ValueError(f"Cannot plot methods missing from OOF predictions: {missing}")
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#7A5195"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.45))
    for color, method in zip(colors, methods):
        part = primary[primary["method"] == method].sort_values("node_id")
        y = labels[part["node_id"].to_numpy()]
        score = part["score"].to_numpy()
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        roc_auc = roc_auc_score(y, score)
        auprc = average_precision_score(y, score)
        label = f"{DISPLAY_NAMES.get(method, method)} [{roc_auc:.3f} | {auprc:.3f}]"
        axes[0].plot(fpr, tpr, color=color, label=label)
        axes[1].plot(recall, precision, color=color, label=label)
    prevalence = float(labels.mean())
    baseline_label = f"Random baseline [0.500 | {prevalence:.3f}]"
    axes[0].plot([0, 1], [0, 1], color="0.5", linestyle="--", label=baseline_label)
    axes[1].axhline(prevalence, color="0.5", linestyle="--", label=baseline_label)
    axes[0].set(title="a  ROC curves", xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(title="b  Precision–recall curves", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    for ax in axes:
        ax.grid(color="0.9", linewidth=0.5)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    ncol = 2 if len(methods) <= 5 else 2
    fig.text(0.5, 0.185, "Legend values: ROC-AUC | AUPRC", ha="center", fontsize=6.2)
    fig.legend(handles, legend_labels, loc="lower center", ncol=ncol, frameon=False, fontsize=5.7)
    fig.tight_layout(rect=(0, 0.27, 1, 1))
    save_figure(fig, BENCHMARK_DIR / output_name)


def regenerate_figures() -> None:
    data_raw = torch_load(RAW_DATASET_FILE)
    labels = data_raw.y.cpu().numpy()
    oof = pd.read_csv(BENCHMARK_DIR / "oof_predictions.csv.gz")
    main_methods = ["RWR (STRING graph)", "Network-degree logistic", PROPOSED, "PPI-GraphSAGE + nnPU", "PPI-GAT + nnPU", "PPI-GCN + nnPU", "Logistic regression", "Random forest"]
    make_curves(oof, labels, main_methods, "benchmark_curves")
    ablation_methods = [PROPOSED, "scTDA-GCN (dual attention + nnPU)", "scTDA-GCN (equal fusion + nnPU)", "PPI-GCN + nnPU"]
    make_curves(oof, labels, ablation_methods, "ablation_curves")
    (BENCHMARK_DIR / "figure_scope.txt").write_text(
        "benchmark_curves contains the complete strong-baseline comparison, including RWR. "
        "ablation_curves contains only GNN variants that were evaluated in the same OOF design. "
        "It must not be described as a complete benchmark. Legend values report ROC-AUC and AUPRC.\n",
        encoding="utf-8",
    )
    write_benchmark_design()


def main() -> None:
    ensure_result_dirs()
    progress_file = BENCHMARK_DIR / "progress.txt"
    progress_file.write_text("benchmark started\n", encoding="utf-8")
    data_raw = torch_load(RAW_DATASET_FILE)
    labels = data_raw.y.cpu().numpy()
    degree_features = network_degree_features(data_raw)
    splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    metric_rows, oof_rows = [], []
    started = time.time()
    for split_number, (outer_train, outer_test) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        repeat = split_number // N_SPLITS
        fold = split_number % N_SPLITS
        split_seed = SEED + split_number
        core_train, inner_val = train_test_split(
            outer_train,
            test_size=INNER_VAL_FRACTION,
            stratify=labels[outer_train],
            random_state=split_seed,
        )
        data, _, _ = transform_graph_features(data_raw, core_train)
        test_y = labels[outer_test]
        print(f"Repeat {repeat + 1}/{N_REPEATS}, fold {fold + 1}/{N_SPLITS}")

        for method, (model_class, kwargs, use_nnpu) in METHODS.items():
            seed_everything(split_seed)
            model = model_class(in_channels=data.x.shape[1], **kwargs)
            _, all_scores, threshold, epochs, val_ap = train_graph_model(
                model, data, core_train, inner_val, use_nnpu=use_nnpu, seed=split_seed
            )
            test_scores = all_scores[outer_test]
            metrics = calculate_metrics(test_y, test_scores, threshold)
            metric_rows.append({"repeat": repeat, "fold": fold, "method": method, "epochs": epochs, "inner_val_AUPRC": val_ap, **metrics})
            oof_rows.extend({"repeat": repeat, "fold": fold, "node_id": int(node), "method": method, "score": float(score), "threshold": threshold} for node, score in zip(outer_test, test_scores))
            print(f"  {method}: AUPRC={metrics['AUPRC']:.4f}")
            with progress_file.open("a", encoding="utf-8") as handle:
                handle.write(f"repeat={repeat + 1}, fold={fold + 1}, method={method}, AUPRC={metrics['AUPRC']:.4f}\n")

        x_core = data.x[core_train].numpy()
        x_val = data.x[inner_val].numpy()
        x_test = data.x[outer_test].numpy()
        for method, estimator in [
            ("Logistic regression", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=split_seed)),
            ("Random forest", RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2, n_jobs=-1, random_state=split_seed)),
        ]:
            estimator.fit(x_core, labels[core_train])
            val_scores = estimator.predict_proba(x_val)[:, 1]
            threshold = choose_f1_threshold(labels[inner_val], val_scores)
            test_scores = estimator.predict_proba(x_test)[:, 1]
            metrics = calculate_metrics(test_y, test_scores, threshold)
            metric_rows.append({"repeat": repeat, "fold": fold, "method": method, "epochs": np.nan, "inner_val_AUPRC": average_precision_score(labels[inner_val], val_scores), **metrics})
            oof_rows.extend({"repeat": repeat, "fold": fold, "node_id": int(node), "method": method, "score": float(score), "threshold": threshold} for node, score in zip(outer_test, test_scores))

        degree_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=split_seed),
        )
        degree_model.fit(degree_features[core_train], labels[core_train])
        degree_val_scores = degree_model.predict_proba(degree_features[inner_val])[:, 1]
        degree_threshold = choose_f1_threshold(labels[inner_val], degree_val_scores)
        degree_test_scores = degree_model.predict_proba(degree_features[outer_test])[:, 1]
        degree_metrics = calculate_metrics(test_y, degree_test_scores, degree_threshold)
        degree_method = "Network-degree logistic"
        metric_rows.append({
            "repeat": repeat,
            "fold": fold,
            "method": degree_method,
            "epochs": np.nan,
            "inner_val_AUPRC": average_precision_score(labels[inner_val], degree_val_scores),
            **degree_metrics,
        })
        oof_rows.extend({
            "repeat": repeat,
            "fold": fold,
            "node_id": int(node),
            "method": degree_method,
            "score": float(score),
            "threshold": degree_threshold,
        } for node, score in zip(outer_test, degree_test_scores))

        seeds = core_train[labels[core_train] == 1]
        all_rwr = run_rwr(data.edge_index_ppi, data.edge_weight_ppi, data.num_nodes, seeds)
        threshold = choose_f1_threshold(labels[inner_val], all_rwr[inner_val])
        test_scores = all_rwr[outer_test]
        metrics = calculate_metrics(test_y, test_scores, threshold)
        method = "RWR (STRING graph)"
        metric_rows.append({"repeat": repeat, "fold": fold, "method": method, "epochs": np.nan, "inner_val_AUPRC": average_precision_score(labels[inner_val], all_rwr[inner_val]), **metrics})
        oof_rows.extend({"repeat": repeat, "fold": fold, "node_id": int(node), "method": method, "score": float(score), "threshold": threshold} for node, score in zip(outer_test, test_scores))
        with progress_file.open("a", encoding="utf-8") as handle:
            handle.write(f"repeat={repeat + 1}, fold={fold + 1} complete\n")

    metrics = pd.DataFrame(metric_rows)
    oof = pd.DataFrame(oof_rows)
    metrics.to_csv(BENCHMARK_DIR / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(BENCHMARK_DIR / "oof_predictions.csv.gz", index=False, compression="gzip")
    numeric = ["ROC_AUC", "AUPRC", "Accuracy", "Precision", "Recall", "F1"]
    summary = metrics.groupby("method")[numeric].agg(["mean", "std"]).reset_index()
    summary.columns = ["Method"] + [f"{metric}_{stat}" for metric, stat in summary.columns.tolist()[1:]]
    summary = summary.sort_values("AUPRC_mean", ascending=False)
    summary.to_csv(BENCHMARK_DIR / "benchmark_summary.csv", index=False, encoding="utf-8-sig")
    bootstrap = bootstrap_statistics(oof, labels)
    bootstrap.to_csv(BENCHMARK_DIR / "paired_bootstrap_auprc.csv", index=False, encoding="utf-8-sig")
    main_methods = ["RWR (STRING graph)", "Network-degree logistic", PROPOSED, "PPI-GraphSAGE + nnPU", "PPI-GAT + nnPU", "PPI-GCN + nnPU", "Logistic regression", "Random forest"]
    primary_summary = summary[summary["Method"].isin(main_methods)].copy()
    primary_summary["Benchmark_Role"] = np.select(
        [
            primary_summary["Method"].eq("RWR (STRING graph)"),
            primary_summary["Method"].eq("Network-degree logistic"),
            primary_summary["Method"].eq(PROPOSED),
        ],
        ["strong network-propagation comparator", "network-degree bias comparator", "proposed multiview GNN"],
        default="architecture or feature baseline",
    )
    primary_summary["Claim_Guardrail"] = np.where(
        primary_summary["Method"].eq(PROPOSED),
        "interpretability/multiview contribution; not the overall best predictor",
        "retained in the primary benchmark",
    )
    primary_summary.to_csv(BENCHMARK_DIR / "primary_benchmark_summary.csv", index=False, encoding="utf-8-sig")
    regenerate_figures()
    (BENCHMARK_DIR / "runtime_seconds.txt").write_text(f"{time.time() - started:.2f}\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    ensure_result_dirs()
    if args.manifest_only:
        write_benchmark_design()
    elif args.figures_only:
        regenerate_figures()
    else:
        main()
