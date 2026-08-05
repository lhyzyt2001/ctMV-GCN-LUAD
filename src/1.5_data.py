from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from common import load_feature_table, save_json
from config import (
    COEXP_EDGE_FILE,
    COEXP_WEIGHT_FILE,
    DATA_DIR,
    FEATURE_FILE,
    LABEL_FILE,
    NODE_MAPPING_FILE,
    PATHWAY_EDGE_FILE,
    PATHWAY_WEIGHT_FILE,
    PPI_EDGE_FILE,
    PPI_WEIGHT_FILE,
    RAW_DATASET_FILE,
    ensure_result_dirs,
)


def main() -> None:
    ensure_result_dirs()
    gene_to_id = json.loads(NODE_MAPPING_FILE.read_text(encoding="utf-8"))
    features = load_feature_table(FEATURE_FILE).loc[list(gene_to_id)]
    positive_genes = set(pd.read_csv(LABEL_FILE)["symbol"].astype(str).str.upper())
    labels = np.asarray([gene in positive_genes for gene in features.index], dtype=np.int64)
    data = Data(
        x=torch.tensor(features.to_numpy(dtype=np.float32)),
        y=torch.tensor(labels, dtype=torch.long),
        edge_index_ppi=torch.tensor(np.load(PPI_EDGE_FILE), dtype=torch.long),
        edge_weight_ppi=torch.tensor(np.load(PPI_WEIGHT_FILE), dtype=torch.float),
        edge_index_coexp=torch.tensor(np.load(COEXP_EDGE_FILE), dtype=torch.long),
        edge_weight_coexp=torch.tensor(np.load(COEXP_WEIGHT_FILE), dtype=torch.float),
        edge_index_pathway=torch.tensor(np.load(PATHWAY_EDGE_FILE), dtype=torch.long),
        edge_weight_pathway=torch.tensor(np.load(PATHWAY_WEIGHT_FILE), dtype=torch.float),
    )
    data.cell_types = features.columns.tolist()
    data.label_definition = "Open Targets LUAD maxClinicalTrialPhase > 0"
    torch.save(data, RAW_DATASET_FILE)
    audit = pd.DataFrame({
        "node_id": range(len(features)),
        "gene_symbol": features.index,
        "clinical_target_label": labels,
    })
    audit.to_csv(DATA_DIR / "node_label_audit.csv", index=False, encoding="utf-8-sig")
    save_json({
        "nodes": int(data.num_nodes),
        "positive_labels": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "feature_count": int(data.x.shape[1]),
        "cell_types": data.cell_types,
        "ppi_edges": int(data.edge_index_ppi.shape[1]),
        "celltype_similarity_edges": int(data.edge_index_coexp.shape[1]),
        "pathway_edges": int(data.edge_index_pathway.shape[1]),
    }, DATA_DIR / "dataset_qc.json")
    print(data)
    print(f"Clinical target labels in graph: {int(labels.sum()):,}/{len(labels):,} ({labels.mean():.2%})")


if __name__ == "__main__":
    main()
