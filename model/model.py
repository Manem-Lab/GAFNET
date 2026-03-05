import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class MultimodalGraphSAGE_AttentionFusion(nn.Module):
    """
    Three-modality GraphSAGE with learned attention fusion.

    Architecture
    ------------
    1. Per-modality linear projections + BN
    2. Per-modality SAGEConv (intra-graph message passing)
    3. Two inter-modality SAGEConv layers on the stacked node set
    4. Attention-weighted fusion of the three modality embeddings
    5. Binary classifier head
    """

    def __init__(
        self,
        in_channels1:    int,
        in_channels2:    int,
        in_channels3:    int,
        hidden_channels: int,
        out_channels:    int   = 1,
        dropout:         float = 0.6,
    ):
        super().__init__()

        # --- Input projections ---
        self.proj1 = nn.Linear(in_channels1, hidden_channels)
        self.proj2 = nn.Linear(in_channels2, hidden_channels)
        self.proj3 = nn.Linear(in_channels3, hidden_channels)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        self.dropout = nn.Dropout(dropout)

        # --- Intra-modality GNN layers ---
        self.conv_mod1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv_mod2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv_mod3 = SAGEConv(hidden_channels, hidden_channels)

        # --- Inter-modality GNN layers ---
        self.conv_inter1 = SAGEConv(hidden_channels, hidden_channels, project=True)
        self.conv_inter2 = SAGEConv(hidden_channels, hidden_channels, project=True)

        # --- Attention scoring heads (one per modality) ---
        def _attn_head():
            return nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Linear(hidden_channels // 2, 1),
            )
        self.attn1 = _attn_head()
        self.attn2 = _attn_head()
        self.attn3 = _attn_head()

        # --- Classifier ---
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x1, x2, x3,
        edge_index1, edge_index2, edge_index3,
        inter_edge_index,
        n_samples: int,
        return_attn: bool = True,
    ):
        # Project + normalise
        x1 = F.relu(self.bn1(self.proj1(x1)))
        x2 = F.relu(self.bn2(self.proj2(x2)))
        x3 = F.relu(self.bn3(self.proj3(x3)))

        # Intra-modality message passing
        x1 = F.elu(self.conv_mod1(x1, edge_index1))
        x2 = F.elu(self.conv_mod2(x2, edge_index2))
        x3 = F.elu(self.conv_mod3(x3, edge_index3))

        # Inter-modality message passing (stacked node set)
        x_all = torch.cat([x1, x2, x3], dim=0)
        x_all = F.gelu(self.conv_inter1(x_all, inter_edge_index))
        x_all = F.gelu(self.conv_inter2(x_all, inter_edge_index))

        x1_f = x_all[:n_samples]
        x2_f = x_all[n_samples:2 * n_samples]
        x3_f = x_all[2 * n_samples:]

        # Attention fusion
        scores = torch.cat([self.attn1(x1_f), self.attn2(x2_f), self.attn3(x3_f)], dim=1)
        alpha  = F.softmax(scores, dim=1)                       # (N, 3)
        fused  = alpha[:, 0:1] * x1_f + alpha[:, 1:2] * x2_f + alpha[:, 2:3] * x3_f

        logits = self.classifier(fused)
        preds  = torch.sigmoid(logits).view(-1)

        if return_attn:
            return preds, alpha.detach().cpu()
        return preds