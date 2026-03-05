import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data


def mutual_knn_graph(X: np.ndarray, k: int = 5):
    """Build a symmetric (mutual) kNN adjacency matrix."""
    knn       = kneighbors_graph(X, n_neighbors=k, mode="connectivity",
                                 include_self=True, metric="minkowski", p=2)
    mutual    = knn.multiply(knn.T)
    return mutual


def sparse_adj_to_edge_index(sparse_adj) -> torch.Tensor:
    """Convert a scipy sparse adjacency matrix to a PyG edge_index tensor."""
    coo        = sparse_adj.tocoo()
    edge_index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
    return edge_index


def build_inter_modality_edges(n_samples: int) -> torch.Tensor:
    """
    Create bidirectional identity edges between all three modality node sets.
    Node layout: [mod1 | mod2 | mod3], each block of size n_samples.
    """
    edges = []
    offsets = [(0, n_samples), (n_samples, 2 * n_samples), (0, 2 * n_samples)]
    for a, b in offsets:
        edges += [(a + i, b + i) for i in range(n_samples)]
        edges += [(b + i, a + i) for i in range(n_samples)]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def create_multimodal_graph(
    X1: np.ndarray,
    X2: np.ndarray,
    X3: np.ndarray,
    y:  np.ndarray,
    k:  int = 5,
) -> Data:
    """
    Build a PyG Data object with per-modality kNN graphs and
    cross-modality identity edges.
    """
    n = X1.shape[0]

    edge_index1 = sparse_adj_to_edge_index(mutual_knn_graph(X1, k))
    edge_index2 = sparse_adj_to_edge_index(mutual_knn_graph(X2, k))
    edge_index3 = sparse_adj_to_edge_index(mutual_knn_graph(X3, k))
    inter_edge_index = build_inter_modality_edges(n)

    return Data(
        x1=torch.tensor(X1, dtype=torch.float),
        x2=torch.tensor(X2, dtype=torch.float),
        x3=torch.tensor(X3, dtype=torch.float),
        edge_index1=edge_index1,
        edge_index2=edge_index2,
        edge_index3=edge_index3,
        inter_edge_index=inter_edge_index,
        y=torch.tensor(y, dtype=torch.float),
        n_samples=n,
    )
