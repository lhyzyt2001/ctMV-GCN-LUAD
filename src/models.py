from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class MultiViewGCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 60, attention_type: str = "local"):
        super().__init__()
        self.attention_type = attention_type
        branch_dim = hidden_channels // 3
        self.ppi1 = GCNConv(in_channels, branch_dim)
        self.ppi2 = GCNConv(branch_dim, branch_dim)
        self.cell1 = GCNConv(in_channels, branch_dim)
        self.cell2 = GCNConv(branch_dim, branch_dim)
        self.path1 = GCNConv(in_channels, branch_dim)
        self.path2 = GCNConv(branch_dim, branch_dim)
        concat_dim = branch_dim * 3
        self.local_attention = torch.nn.Sequential(
            torch.nn.Linear(concat_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3),
        )
        self.global_attention = torch.nn.Parameter(torch.zeros(3))
        self.classifier = torch.nn.Linear(concat_dim, 2)

    def _branch(self, x, edge_index, edge_weight, conv1, conv2):
        h = F.relu(conv1(x, edge_index, edge_weight=edge_weight))
        h = F.dropout(h, p=0.5, training=self.training)
        return conv2(h, edge_index, edge_weight=edge_weight)

    def forward(self, data, return_attention: bool = False, channel_mask=None):
        h_ppi = self._branch(data.x, data.edge_index_ppi, data.edge_weight_ppi, self.ppi1, self.ppi2)
        h_cell = self._branch(data.x, data.edge_index_coexp, data.edge_weight_coexp, self.cell1, self.cell2)
        h_path = self._branch(data.x, data.edge_index_pathway, data.edge_weight_pathway, self.path1, self.path2)
        concat = torch.cat([h_ppi, h_cell, h_path], dim=1)
        local = F.softmax(self.local_attention(concat), dim=1)
        global_weight = F.softmax(self.global_attention, dim=0)
        if self.attention_type == "local":
            alpha = local
        elif self.attention_type == "dual":
            alpha = local * global_weight
            alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)
        elif self.attention_type == "equal":
            alpha = torch.full_like(local, 1.0 / 3.0)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")
        if channel_mask is not None:
            mask = torch.as_tensor(channel_mask, dtype=alpha.dtype, device=alpha.device).view(1, 3)
            alpha = alpha * mask
            alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)
        fused = torch.cat([
            h_ppi * alpha[:, 0:1],
            h_cell * alpha[:, 1:2],
            h_path * alpha[:, 2:3],
        ], dim=1)
        logits = self.classifier(fused)
        if return_attention:
            return logits, alpha
        return logits


class PpiGCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 2)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index_ppi, edge_weight=data.edge_weight_ppi))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, data.edge_index_ppi, edge_weight=data.edge_weight_ppi)


class PpiGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, 2)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index_ppi))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, data.edge_index_ppi)


class PpiGAT(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 16):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=2, dropout=0.3)
        self.conv2 = GATConv(hidden_channels * 2, 2, heads=1, concat=False, dropout=0.3)

    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index_ppi))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, data.edge_index_ppi)


def nnpu_loss(logits: torch.Tensor, labels: torch.Tensor, positive_prior: float) -> torch.Tensor:
    positive = labels == 1
    unlabeled = labels == 0
    if not positive.any() or not unlabeled.any():
        return F.cross_entropy(logits, labels)
    pos_positive = F.softplus(-logits[positive, 1] + logits[positive, 0]).mean()
    pos_negative = F.softplus(logits[positive, 1] - logits[positive, 0]).mean()
    unl_negative = F.softplus(logits[unlabeled, 1] - logits[unlabeled, 0]).mean()
    negative_risk = unl_negative - positive_prior * pos_negative
    return positive_prior * pos_positive + torch.clamp(negative_risk, min=0.0)
