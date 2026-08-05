from __future__ import annotations

import copy

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from common import choose_f1_threshold, seed_everything
from config import MAX_EPOCHS, PATIENCE, PCA_COMPONENTS, VALIDATION_INTERVAL
from models import nnpu_loss


def transform_graph_features(data, fit_indices: np.ndarray):
    scaler = StandardScaler()
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    raw = data.x.cpu().numpy()
    scaler.fit(raw[fit_indices])
    pca.fit(scaler.transform(raw[fit_indices]))
    transformed = pca.transform(scaler.transform(raw)).astype(np.float32)
    output = data.clone()
    output.x = torch.tensor(transformed, dtype=torch.float)
    return output, scaler, pca


def predict_scores(model, data) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(data)
        return F.softmax(logits, dim=1)[:, 1].cpu().numpy()


def train_graph_model(model, data, train_idx, val_idx, use_nnpu: bool, seed: int):
    seed_everything(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    labels = data.y
    train_tensor = torch.tensor(train_idx, dtype=torch.long)
    val_tensor = torch.tensor(val_idx, dtype=torch.long)
    train_labels = labels[train_tensor]
    positive_prior = float((train_labels == 1).float().mean().item())
    positive_count = max(int((train_labels == 1).sum()), 1)
    negative_count = max(int((train_labels == 0).sum()), 1)
    weights = torch.tensor([1.0, negative_count / positive_count], dtype=torch.float)
    best_ap, best_epoch, stale = -np.inf, 0, 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        if use_nnpu:
            loss = nnpu_loss(logits[train_tensor], train_labels, positive_prior)
        else:
            loss = F.cross_entropy(logits[train_tensor], train_labels, weight=weights)
        loss.backward()
        optimizer.step()
        if epoch % VALIDATION_INTERVAL == 0 or epoch == MAX_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                val_scores = F.softmax(model(data)[val_tensor], dim=1)[:, 1].cpu().numpy()
            val_ap = average_precision_score(labels[val_tensor].cpu().numpy(), val_scores)
            if val_ap > best_ap + 1e-4:
                best_ap, best_epoch, stale = val_ap, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    scores = predict_scores(model, data)
    threshold = choose_f1_threshold(labels[val_tensor].cpu().numpy(), scores[val_idx])
    return model, scores, threshold, best_epoch + 1, float(best_ap)


def train_fixed_epochs(model, data, train_idx, use_nnpu: bool, epochs: int, seed: int):
    seed_everything(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    idx = torch.tensor(train_idx, dtype=torch.long)
    y = data.y[idx]
    positive_prior = float((y == 1).float().mean().item())
    positive_count = max(int((y == 1).sum()), 1)
    negative_count = max(int((y == 0).sum()), 1)
    weights = torch.tensor([1.0, negative_count / positive_count], dtype=torch.float)
    for _ in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = nnpu_loss(logits[idx], y, positive_prior) if use_nnpu else F.cross_entropy(logits[idx], y, weight=weights)
        loss.backward()
        optimizer.step()
    return model


def run_rwr(edge_index, edge_weight, num_nodes: int, seed_idx, restart_prob: float = 0.3):
    edges = edge_index.cpu().numpy()
    weights = edge_weight.cpu().numpy()
    adjacency = sp.coo_matrix((weights, (edges[0], edges[1])), shape=(num_nodes, num_nodes)).tocsc()
    degree = np.asarray(adjacency.sum(axis=0)).ravel()
    degree[degree == 0] = 1.0
    transition = adjacency @ sp.diags(1.0 / degree)
    initial = np.zeros(num_nodes, dtype=float)
    initial[np.asarray(seed_idx, dtype=int)] = 1.0 / max(len(seed_idx), 1)
    score = initial.copy()
    for _ in range(200):
        updated = (1 - restart_prob) * transition.dot(score) + restart_prob * initial
        if np.linalg.norm(updated - score, 1) < 1e-9:
            score = updated
            break
        score = updated
    return score
