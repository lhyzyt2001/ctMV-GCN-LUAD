from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt

import joblib
import numpy as np
import pandas as pd
import torch

from common import seed_everything, torch_load
from config import (
    BENCHMARK_DIR,
    FINAL_ENSEMBLE_SEEDS,
    MODEL_DIR,
    RAW_DATASET_FILE,
    ensure_result_dirs,
)
from experiment import train_fixed_epochs, transform_graph_features
from models import MultiViewGCN
from plot_style import FULL_WIDTH, apply_bmc_style, save_figure


def summarize_attention_stability(attention: pd.DataFrame) -> None:
    channels = ["PPI", "CellTypeSimilarity", "Pathway"]
    summary = attention[channels].agg(["mean", "std", "min", "max"]).T.reset_index()
    summary.columns = ["Channel", "Mean", "SD", "Minimum", "Maximum"]
    summary["Q2.5"] = [attention[channel].quantile(0.025) for channel in channels]
    summary["Q97.5"] = [attention[channel].quantile(0.975) for channel in channels]
    summary["Seed_N"] = len(attention)
    summary.to_csv(MODEL_DIR / "attention_stability_summary.csv", index=False, encoding="utf-8-sig")

    apply_bmc_style()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.62, 3.4))
    values = [attention[channel].to_numpy(dtype=float) for channel in channels]
    ax.boxplot(values, labels=["STRING", "Cell-type similarity", "Pathway"], widths=0.5, showfliers=False)
    for position, channel_values in enumerate(values, start=1):
        jitter = np.linspace(-0.08, 0.08, len(channel_values))
        ax.scatter(position + jitter, channel_values, s=12, color="#0072B2", alpha=0.75, zorder=3)
    ax.set(xlabel="Graph channel", ylabel="Mean node-level attention", title="Attention stability across ensemble seeds")
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, MODEL_DIR / "attention_stability_across_seeds")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final ensemble and summarize attention stability.")
    parser.add_argument("--summary-only", action="store_true", help="Reuse existing seed-level attention values without retraining.")
    args = parser.parse_args()
    if args.summary_only:
        attention = pd.read_csv(MODEL_DIR / "attention_stability_by_seed.csv")
        summarize_attention_stability(attention)
        print(attention.to_string(index=False))
        return
    ensure_result_dirs()
    raw = torch_load(RAW_DATASET_FILE)
    all_idx = np.arange(raw.num_nodes)
    data, scaler, pca = transform_graph_features(raw, all_idx)
    joblib.dump(scaler, MODEL_DIR / "feature_scaler.joblib")
    joblib.dump(pca, MODEL_DIR / "feature_pca.joblib")
    torch.save(data, MODEL_DIR / "GNN_Dataset_Final.pt")

    fold_file = BENCHMARK_DIR / "fold_metrics.csv"
    if fold_file.exists():
        folds = pd.read_csv(fold_file)
        selected = folds[folds["method"] == "ctMV-GCN (local attention + class-weighted CE)"]
        training_epochs = int(max(10, round(selected["epochs"].median())))
    else:
        training_epochs = 40

    rows = []
    for seed in FINAL_ENSEMBLE_SEEDS:
        seed_everything(seed)
        model = MultiViewGCN(in_channels=data.x.shape[1], attention_type="local")
        model = train_fixed_epochs(model, data, all_idx, use_nnpu=False, epochs=training_epochs, seed=seed)
        model.eval()
        with torch.no_grad():
            _, attention = model(data, return_attention=True)
        model_path = MODEL_DIR / f"ctMV_GCN_local_weightedCE_seed{seed}.pth"
        torch.save(model.state_dict(), model_path)
        mean = attention.mean(dim=0).cpu().numpy()
        rows.append({"seed": seed, "epochs": training_epochs, "PPI": mean[0], "CellTypeSimilarity": mean[1], "Pathway": mean[2]})
    attention = pd.DataFrame(rows)
    attention.to_csv(MODEL_DIR / "attention_stability_by_seed.csv", index=False, encoding="utf-8-sig")
    summarize_attention_stability(attention)
    (MODEL_DIR / "training_manifest.json").write_text(json.dumps({
        "architecture": "three-view GCN with node-level local attention",
        "loss": "class-weighted cross-entropy",
        "ensemble_seeds": list(FINAL_ENSEMBLE_SEEDS),
        "epochs_selected_from_cross_validation_median": training_epochs,
        "global_attention_removed_from_primary_model": True,
        "nnPU_removed_from_primary_model_because_cross_validation_showed_no_advantage": True,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(attention.to_string(index=False))


if __name__ == "__main__":
    main()
