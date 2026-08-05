from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_feature_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str).str.strip().str.upper()
    frame = frame.loc[~frame.index.duplicated(keep="first")]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def choose_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        score = f1_score(y_true, scores >= threshold, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def torch_load(path: Path, device: torch.device | str = "cpu"):
    return torch.load(path, map_location=device, weights_only=False)


def network_degree_features(data) -> np.ndarray:
    """Return log-transformed unweighted and weighted degree for all graph views."""
    columns = []
    for edge_name, weight_name in (
        ("edge_index_ppi", "edge_weight_ppi"),
        ("edge_index_coexp", "edge_weight_coexp"),
        ("edge_index_pathway", "edge_weight_pathway"),
    ):
        edge_index = getattr(data, edge_name).cpu().numpy()
        edge_weight = getattr(data, weight_name).cpu().numpy()
        degree = np.bincount(edge_index[0], minlength=data.num_nodes)
        weighted_degree = np.bincount(edge_index[0], weights=edge_weight, minlength=data.num_nodes)
        columns.extend((np.log1p(degree), np.log1p(weighted_degree)))
    return np.column_stack(columns).astype(np.float32)


def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    return np.concatenate([
        rng.choice(pos, size=len(pos), replace=True),
        rng.choice(neg, size=len(neg), replace=True),
    ])
