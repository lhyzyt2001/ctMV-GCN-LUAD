from __future__ import annotations

import json

import numpy as np
import scipy.sparse as sp

from common import save_json
from config import (
    DATA_DIR,
    KEGG_GMT_FILE,
    NODE_MAPPING_FILE,
    PATHWAY_EDGE_FILE,
    PATHWAY_K,
    PATHWAY_WEIGHT_FILE,
    ensure_result_dirs,
)


def main() -> None:
    ensure_result_dirs()
    gene_to_id = json.loads(NODE_MAPPING_FILE.read_text(encoding="utf-8"))
    rows, cols, pathway_names, pathway_sizes = [], [], [], []
    with KEGG_GMT_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            genes = sorted({g.upper() for g in parts[2:] if g.upper() in gene_to_id})
            if len(genes) < 2:
                continue
            pathway_index = len(pathway_names)
            pathway_names.append(parts[0])
            pathway_sizes.append(len(genes))
            rows.extend(gene_to_id[g] for g in genes)
            cols.extend([pathway_index] * len(genes))
    incidence = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(gene_to_id), len(pathway_names)),
    )
    path_weight = sp.diags(1.0 / np.asarray(pathway_sizes, dtype=np.float32))
    similarity = (incidence @ path_weight @ incidence.T).tocsr()
    similarity.setdiag(0)
    similarity.eliminate_zeros()
    selected = {}
    for source in range(similarity.shape[0]):
        start, end = similarity.indptr[source], similarity.indptr[source + 1]
        targets = similarity.indices[start:end]
        weights = similarity.data[start:end]
        if len(targets) > PATHWAY_K:
            keep = np.argpartition(weights, -PATHWAY_K)[-PATHWAY_K:]
            targets, weights = targets[keep], weights[keep]
        for target, weight in zip(targets, weights):
            if source != target:
                selected[(source, int(target))] = float(weight)
                selected[(int(target), source)] = float(weight)
    edges = sorted(selected)
    edge_index = np.asarray(edges, dtype=np.int64).T
    edge_weight = np.asarray([selected[e] for e in edges], dtype=np.float32)
    if edge_weight.size:
        edge_weight /= edge_weight.max()
    np.save(PATHWAY_EDGE_FILE, edge_index)
    np.save(PATHWAY_WEIGHT_FILE, edge_weight)
    save_json({
        "network_name": "weighted shared-pathway similarity graph",
        "pathway_count": len(pathway_names),
        "top_k_per_gene_before_symmetrization": PATHWAY_K,
        "directed_edges_after_symmetrization": int(edge_index.shape[1]),
        "construction": "gene-pathway incidence H multiplied as H diag(1/pathway_size) H^T, followed by top-k sparsification",
        "alphabetical_ring_removed": True,
    }, DATA_DIR / "pathway_qc.json")
    print(f"Pathways: {len(pathway_names):,}; weighted pathway edges: {edge_index.shape[1]:,}")


if __name__ == "__main__":
    main()
