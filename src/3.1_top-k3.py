from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from common import torch_load
from config import BENCHMARK_DIR, RAW_DATASET_FILE, TOPK_DIR, ensure_result_dirs
from plot_style import FULL_WIDTH, apply_bmc_style, save_figure


K_VALUES = (10, 20, 50, 100, 200, 500)


def main() -> None:
    ensure_result_dirs()
    data = torch_load(RAW_DATASET_FILE)
    labels = data.y.cpu().numpy()
    prevalence = float(labels.mean())
    oof = pd.read_csv(BENCHMARK_DIR / "oof_predictions.csv.gz")
    rows = []
    for (repeat, method), part in oof.groupby(["repeat", "method"]):
        part = part.sort_values("score", ascending=False)
        ranked_labels = labels[part["node_id"].to_numpy()]
        total_positive = int(ranked_labels.sum())
        for k in K_VALUES:
            actual_k = min(k, len(part))
            hits = int(ranked_labels[:actual_k].sum())
            precision = hits / actual_k
            recall = hits / total_positive
            rows.append({
                "repeat": int(repeat),
                "Method": method,
                "K": k,
                "Candidate_Pool_Size": len(part),
                "Positive_Targets": total_positive,
                "Hits": hits,
                "Precision": precision,
                "Recall": recall,
                "Enrichment_Factor": precision / prevalence,
                "Random_Expected_Hits": k * prevalence,
                "Random_Precision": prevalence,
            })
    raw = pd.DataFrame(rows)
    raw.to_csv(TOPK_DIR / "topk_metrics_by_repeat.csv", index=False, encoding="utf-8-sig")
    summary = raw.groupby(["Method", "K"])[["Hits", "Precision", "Recall", "Enrichment_Factor"]].agg(["mean", "std"]).reset_index()
    summary.columns = ["Method", "K"] + [f"{metric}_{stat}" for metric, stat in summary.columns.tolist()[2:]]
    summary.to_csv(TOPK_DIR / "topk_metrics_summary.csv", index=False, encoding="utf-8-sig")

    apply_bmc_style()
    preferred = [
        "RWR (STRING graph)",
        "Network-degree logistic",
        "scTDA-GCN (local attention + weighted BCE)",
        "PPI-GraphSAGE + nnPU",
        "PPI-GAT + nnPU",
        "PPI-GCN + nnPU",
        "Logistic regression",
    ]
    colors = ["#0072B2", "#7A5195", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#000000"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.35))
    for method, color in zip(preferred, colors):
        sub = summary[summary["Method"] == method].sort_values("K")
        if sub.empty:
            continue
        x = np.arange(len(sub))
        axes[0].errorbar(x, sub["Hits_mean"], yerr=sub["Hits_std"].fillna(0), marker="o", markersize=3, capsize=2, color=color, label=method)
        axes[1].errorbar(x, sub["Precision_mean"], yerr=sub["Precision_std"].fillna(0), marker="o", markersize=3, capsize=2, color=color, label=method)
    x = np.arange(len(K_VALUES))
    random_label = f"Random baseline ({prevalence:.2%})"
    axes[0].plot(x, np.asarray(K_VALUES) * prevalence, linestyle="--", color="0.5", label=random_label)
    axes[1].axhline(prevalence, linestyle="--", color="0.5", label=random_label)
    axes[0].set(title="a  Hidden target recovery", xlabel="Top-K ranked genes", ylabel="Recovered targets", xticks=x, xticklabels=K_VALUES)
    axes[1].set(title="b  Precision at K", xlabel="Top-K ranked genes", ylabel="Precision@K", xticks=x, xticklabels=K_VALUES)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    for ax in axes:
        ax.grid(color="0.9", linewidth=0.5)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=5.4)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    save_figure(fig, TOPK_DIR / "topk_comparison")
    (TOPK_DIR / "evaluation_protocol.txt").write_text(
        "Every node is scored only when it is in the outer test fold. Out-of-fold predictions are pooled within each repeat; no early-stopping validation label is counted as a hidden test target. The random baseline equals the clinical-target prevalence in the same candidate pool.\n",
        encoding="utf-8",
    )
    print(summary[summary["K"].isin([10, 20, 50])].to_string(index=False))


if __name__ == "__main__":
    main()
