from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import load_feature_table, torch_load
from config import (
    DATA_DIR,
    FEATURE_FILE,
    FINAL_ENSEMBLE_SEEDS,
    MODEL_DIR,
    NODE_MAPPING_FILE,
    ensure_result_dirs,
)
from models import MultiViewGCN


def main() -> None:
    ensure_result_dirs()
    data = torch_load(MODEL_DIR / "GNN_Dataset_Final.pt")
    gene_to_id = json.loads(NODE_MAPPING_FILE.read_text(encoding="utf-8"))
    id_to_gene = {idx: gene for gene, idx in gene_to_id.items()}
    raw_features = load_feature_table(FEATURE_FILE).loc[list(gene_to_id)]
    score_runs, attention_runs, perturbation_runs = [], [], []
    for seed in FINAL_ENSEMBLE_SEEDS:
        model = MultiViewGCN(in_channels=data.x.shape[1], attention_type="local")
        model.load_state_dict(torch.load(MODEL_DIR / f"scTDA_GCN_local_weightedBCE_seed{seed}.pth", map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            logits, attention = model(data, return_attention=True)
            full_score = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            score_runs.append(full_score)
            attention_runs.append(attention.cpu().numpy())
            drops = []
            for mask in ([0, 1, 1], [1, 0, 1], [1, 1, 0]):
                masked = F.softmax(model(data, channel_mask=mask), dim=1)[:, 1].cpu().numpy()
                drops.append(full_score - masked)
            perturbation_runs.append(np.column_stack(drops))
    scores = np.vstack(score_runs)
    attentions = np.stack(attention_runs)
    perturbations = np.stack(perturbation_runs)
    labels = data.y.cpu().numpy()
    unlabeled = labels == 0
    seed_ranks = np.full(scores.shape, np.nan, dtype=float)
    for run_index, run in enumerate(scores):
        seed_ranks[run_index, unlabeled] = pd.Series(-run[unlabeled]).rank(method="average").to_numpy()
    values = raw_features.to_numpy(dtype=float)
    row_mean = values.mean(axis=1, keepdims=True)
    row_std = values.std(axis=1, keepdims=True)
    specificity = (values - row_mean) / np.where(row_std > 0, row_std, 1.0)
    top_cell_idx = np.argmax(specificity, axis=1)
    genes = [id_to_gene[i] for i in range(data.num_nodes)]
    result = pd.DataFrame({
        "Node_ID": np.arange(data.num_nodes),
        "Gene_Symbol": genes,
        "Prediction_Score_Mean": scores.mean(axis=0),
        "Prediction_Score_SD": scores.std(axis=0, ddof=1),
        "Rank_Mean_Across_Seeds": seed_ranks.mean(axis=0),
        "Rank_SD_Across_Seeds": seed_ranks.std(axis=0, ddof=1),
        "Top100_Frequency_Across_Seeds": (seed_ranks <= 100).mean(axis=0),
        "Top500_Frequency_Across_Seeds": (seed_ranks <= 500).mean(axis=0),
        "Clinical_Target_Label": labels,
        "Top_Cell_Type": [raw_features.columns[i] for i in top_cell_idx],
        "Top_Cell_Type_Z": specificity[np.arange(len(top_cell_idx)), top_cell_idx],
        "Attention_PPI_Mean": attentions[:, :, 0].mean(axis=0),
        "Attention_CellTypeSimilarity_Mean": attentions[:, :, 1].mean(axis=0),
        "Attention_Pathway_Mean": attentions[:, :, 2].mean(axis=0),
        "Score_Drop_Remove_PPI": perturbations[:, :, 0].mean(axis=0),
        "Score_Drop_Remove_CellTypeSimilarity": perturbations[:, :, 1].mean(axis=0),
        "Score_Drop_Remove_Pathway": perturbations[:, :, 2].mean(axis=0),
    })
    attention_names = ("PPI", "CellTypeSimilarity", "Pathway")
    perturbation_names = ("PPI", "CellTypeSimilarity", "Pathway")
    for index, name in enumerate(attention_names):
        result[f"Attention_{name}_SD"] = attentions[:, :, index].std(axis=0, ddof=1)
        result[f"Attention_{name}_Q2.5"] = np.quantile(attentions[:, :, index], 0.025, axis=0)
        result[f"Attention_{name}_Q97.5"] = np.quantile(attentions[:, :, index], 0.975, axis=0)
    for index, name in enumerate(perturbation_names):
        result[f"Score_Drop_Remove_{name}_SD"] = perturbations[:, :, index].std(axis=0, ddof=1)
        result[f"Score_Drop_Remove_{name}_Q2.5"] = np.quantile(perturbations[:, :, index], 0.025, axis=0)
        result[f"Score_Drop_Remove_{name}_Q97.5"] = np.quantile(perturbations[:, :, index], 0.975, axis=0)
    result = result.sort_values("Prediction_Score_Mean", ascending=False)
    result.to_csv(MODEL_DIR / "all_gene_predictions.csv.gz", index=False, compression="gzip")
    candidates = result[result["Clinical_Target_Label"] == 0].copy().reset_index(drop=True)
    candidates.insert(0, "Candidate_Rank", np.arange(1, len(candidates) + 1))
    candidates.to_csv(MODEL_DIR / "unlabeled_candidate_ranking.csv", index=False, encoding="utf-8-sig")
    candidates.head(100).to_csv(MODEL_DIR / "top100_unlabeled_candidates.csv", index=False, encoding="utf-8-sig")
    print(candidates.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
