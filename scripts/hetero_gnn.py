"""Heterogeneous GNN model for Pipeline C video performance node regression.

Two variants (--model sage | hgt):
  sage : HeteroConv + SAGEConv  — inductive, suited to cold-start
  hgt  : HGTConv                — attention-based ablation variant

Architecture
------------
  1. Input projection  : Linear(-1 → d) for video/author;
                         nn.Embedding(N_cat, d) for category.
  2. Message passing   : 2 × HeteroConv layers with ReLU in between.
  3. Readout head      : video nodes → MLP(d → 64 → 8)
                         outputs[:, :7] → sigmoid   (7 rate metrics)
                         outputs[:, 7:] → linear     (log-space watch_time)

Usage (import)
--------------
  from scripts.hetero_gnn import HeteroGNN
  model = HeteroGNN(data.metadata(), n_cat=data['category'].num_nodes,
                    d=128, layers=2, model='sage')
  out = model(data.x_dict, data.edge_index_dict)  # [N_video, 8]
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, HGTConv, SAGEConv, Linear


class HeteroGNN(nn.Module):
    """Video-centric heterogeneous GNN for multi-task node regression.

    Parameters
    ----------
    metadata   : tuple returned by HeteroData.metadata()
    n_cat      : number of category nodes (for nn.Embedding)
    d          : hidden/output dimension (default 128)
    layers     : number of message-passing layers (default 2)
    model      : 'sage' (GraphSAGE) or 'hgt' (Heterogeneous Graph Transformer)
    dropout    : dropout rate after each conv layer
    """

    def __init__(
        self,
        metadata: tuple,
        n_cat: int,
        d: int = 128,
        layers: int = 2,
        model: str = "sage",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        node_types, edge_types = metadata
        self.d = d
        self.model_type = model
        self.node_types  = node_types
        self.n_cat = n_cat

        # Input projection — Linear(-1, d) for video & author (have numeric x)
        self.proj = nn.ModuleDict()
        for nt in node_types:
            if nt != "category":
                self.proj[nt] = Linear(-1, d)

        # Category nodes: trainable embedding (no raw x)
        self.cat_embed = nn.Embedding(n_cat, d)

        # Message passing layers
        self.convs = nn.ModuleList()
        for _ in range(layers):
            if model == "sage":
                conv_dict = {
                    et: SAGEConv((-1, -1), d) for et in edge_types
                }
                conv = HeteroConv(conv_dict, aggr="sum")
            elif model == "hgt":
                conv = HGTConv(d, d, metadata, heads=4)
            else:
                raise ValueError(f"Unknown model type: {model!r}. Choose 'sage' or 'hgt'.")
            self.convs.append(conv)

        self.dropout = nn.Dropout(dropout)

        # Readout MLP on video nodes → 8 targets
        self.head = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 8),
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
    ) -> torch.Tensor:
        """Run forward pass.

        Parameters
        ----------
        x_dict         : node feature dict (video.x, author.x; category has no x)
        edge_index_dict: edge index dict from HeteroData

        Returns
        -------
        Tensor [N_video, 8]
            Columns 0-6: sigmoid-activated rates (like, comment, follow, collect,
                         forward, hate, effective_view)
            Column 7   : linear (log-space mean_watch_time — use expm1 at inference)
        """
        # Project each node type to d-dim
        h: dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            if nt == "category":
                # Use trainable embedding weight directly
                h[nt] = self.cat_embed.weight  # [N_cat, d]
            elif nt in x_dict:
                h[nt] = self.proj[nt](x_dict[nt])
            else:
                # Fallback: zero init (shouldn't happen with correct graph)
                raise KeyError(f"Node type '{nt}' has no x and is not category.")

        # Message passing
        for conv in self.convs:
            h = conv(h, edge_index_dict)
            h = {k: self.dropout(torch.relu(v)) for k, v in h.items()}

        # Readout head on video nodes only
        out = self.head(h["video"])         # [N_video, 8]
        rates = torch.sigmoid(out[:, :7])   # 7 rate metrics ∈ (0, 1)
        wt    = out[:, 7:8]                 # log-space watch_time (linear)
        return torch.cat([rates, wt], dim=1)


def build_model(data, args) -> HeteroGNN:
    """Convenience factory that reads graph metadata from a HeteroData object."""
    return HeteroGNN(
        metadata=data.metadata(),
        n_cat=data["category"].num_nodes,
        d=getattr(args, "d", 128),
        layers=getattr(args, "layers", 2),
        model=getattr(args, "model", "sage"),
        dropout=getattr(args, "dropout", 0.1),
    )
