from __future__ import annotations

import json

import numpy as np
from sklearn.neighbors import NearestNeighbors

from common import load_feature_table, save_json
from config import (
    CELLTYPE_K,
    COEXP_EDGE_FILE,
    COEXP_WEIGHT_FILE,
    DATA_DIR,
    FEATURE_FILE,
    NODE_MAPPING_FILE,
    ensure_result_dirs,
)


def main() -> None:
    ensure_result_dirs()
    features = load_feature_table(FEATURE_FILE)
    gene_to_id = json.loads(NODE_MAPPING_FILE.read_text(encoding="utf-8"))
    features = features.loc[list(gene_to_id)]
    values = features.to_numpy(dtype=np.float32)
    means = values.mean(axis=1, keepdims=True)
    stds = values.std(axis=1, keepdims=True)
    profiles = (values - means) / np.where(stds > 0, stds, 1.0)
    knn = NearestNeighbors(n_neighbors=CELLTYPE_K + 1, metric="cosine", n_jobs=-1)
    distances, indices = knn.fit(profiles).kneighbors(profiles)
    directed = {}
    for source in range(len(indices)):
        for distance, target in zip(distances[source, 1:], indices[source, 1:]):
            if source != target:
                directed[(source, int(target))] = max(0.0, 1.0 - float(distance))
    mutual = []
    for (source, target), weight in directed.items():
        if (target, source) in directed:
            mutual.append((source, target, (weight + directed[(target, source)]) / 2.0))
    mutual.sort(key=lambda item: (item[0], item[1]))
    edge_index = np.array([(s, t) for s, t, _ in mutual], dtype=np.int64).T
    edge_weight = np.array([w for _, _, w in mutual], dtype=np.float32)
    np.save(COEXP_EDGE_FILE, edge_index)
    np.save(COEXP_WEIGHT_FILE, edge_weight)
    save_json({
        "network_name": "cell-type expression-profile similarity network",
        "cell_types": features.columns.tolist(),
        "knn_k": CELLTYPE_K,
        "mutual_knn": True,
        "directed_edges": int(edge_index.shape[1]),
        "note": "The input contains cell-type aggregated expression, not a per-cell matrix; the graph is not called a single-cell co-expression graph.",
    }, DATA_DIR / "celltype_similarity_qc.json")
    print(f"Mutual cell-type-profile similarity edges: {edge_index.shape[1]:,}")


if __name__ == "__main__":
    main()
