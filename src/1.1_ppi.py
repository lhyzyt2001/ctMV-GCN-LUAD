from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import load_feature_table, save_json
from config import (
    DATA_DIR,
    FEATURE_FILE,
    NODE_MAPPING_FILE,
    PPI_EDGE_FILE,
    PPI_WEIGHT_FILE,
    STRING_INFO_FILE,
    STRING_LINK_FILE,
    STRING_SCORE_THRESHOLD,
    ensure_result_dirs,
)


def main() -> None:
    ensure_result_dirs()
    features = load_feature_table(FEATURE_FILE)
    informative = features.var(axis=1) > 0
    features = features.loc[informative]
    gene_to_id = {gene: idx for idx, gene in enumerate(features.index)}
    valid_genes = set(gene_to_id)
    NODE_MAPPING_FILE.write_text(json.dumps(gene_to_id, ensure_ascii=False), encoding="utf-8")
    np.save(DATA_DIR / "node_features_raw.npy", features.to_numpy(dtype=np.float32))
    features.to_csv(DATA_DIR / "celltype_expression_features.csv", encoding="utf-8-sig")

    info = pd.read_csv(STRING_INFO_FILE, sep="\t", usecols=[0, 1])
    ensp_to_symbol = dict(zip(info.iloc[:, 0], info.iloc[:, 1].astype(str).str.upper()))
    edge_chunks: list[pd.DataFrame] = []
    raw_rows = 0
    for chunk in pd.read_csv(STRING_LINK_FILE, sep=r"\s+", chunksize=2_000_000):
        raw_rows += len(chunk)
        chunk = chunk.loc[chunk["combined_score"] >= STRING_SCORE_THRESHOLD].copy()
        chunk["source_gene"] = chunk["protein1"].map(ensp_to_symbol)
        chunk["target_gene"] = chunk["protein2"].map(ensp_to_symbol)
        chunk = chunk[
            chunk["source_gene"].isin(valid_genes)
            & chunk["target_gene"].isin(valid_genes)
            & (chunk["source_gene"] != chunk["target_gene"])
        ]
        edge_chunks.append(chunk[["source_gene", "target_gene", "combined_score"]])

    aligned = pd.concat(edge_chunks, ignore_index=True)
    aligned["source"] = aligned["source_gene"].map(gene_to_id)
    aligned["target"] = aligned["target_gene"].map(gene_to_id)
    reverse = aligned.rename(columns={"source": "target", "target": "source"})
    aligned = pd.concat([aligned, reverse], ignore_index=True)
    aligned = aligned.groupby(["source", "target"], as_index=False)["combined_score"].max()
    aligned = aligned.sort_values(["source", "target"])
    edge_index = aligned[["source", "target"]].to_numpy(dtype=np.int64).T
    edge_weight = (aligned["combined_score"].to_numpy(dtype=np.float32) / 1000.0)
    np.save(PPI_EDGE_FILE, edge_index)
    np.save(PPI_WEIGHT_FILE, edge_weight)
    aligned.to_csv(DATA_DIR / "string_edges_audit.csv.gz", index=False, compression="gzip")
    save_json({
        "network_name": "STRING high-confidence functional association network",
        "combined_score_threshold": STRING_SCORE_THRESHOLD,
        "raw_string_rows": raw_rows,
        "node_count": len(gene_to_id),
        "directed_edges_after_symmetrization": int(edge_index.shape[1]),
        "note": "STRING combined scores represent functional associations and are not described as physical PPI.",
    }, DATA_DIR / "ppi_qc.json")
    print(f"Nodes: {len(gene_to_id):,}; STRING directed edges: {edge_index.shape[1]:,}")


if __name__ == "__main__":
    main()
