import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from graph_utils import create_multimodal_graph
from model import MultimodalGraphSAGE_AttentionFusion


# ============================================================
# Single-epoch helpers
# ============================================================

def train_one_epoch(
    model:      nn.Module,
    data,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    train_mask: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out, _ = model(
        data.x1, data.x2, data.x3,
        data.edge_index1, data.edge_index2, data.edge_index3,
        data.inter_edge_index, data.n_samples,
    )
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def evaluate(
    model:    nn.Module,
    data,
    mask:     torch.Tensor,
) -> tuple[float, float, float, np.ndarray]:
    """Return (auc, accuracy, f1, attention_weights)."""
    model.eval()
    with torch.no_grad():
        out, attn = model(
            data.x1, data.x2, data.x3,
            data.edge_index1, data.edge_index2, data.edge_index3,
            data.inter_edge_index, data.n_samples,
        )
    preds  = out[mask].cpu().numpy()
    labels = data.y[mask].cpu().numpy()
    return (
        roc_auc_score(labels, preds),
        accuracy_score(labels, preds >= 0.5),
        f1_score(labels, preds >= 0.5),
        attn,
    )


# ============================================================
# Per-fold training with early stopping
# ============================================================

def train_fold(
    model:       nn.Module,
    data,
    train_mask:  torch.Tensor,
    val_mask:    torch.Tensor,
    num_epochs:  int   = 200,
    patience:    int   = 20,
    eval_every:  int   = 15,
    lr:          float = 5e-4,
    weight_decay: float = 5e-4,
    device:      torch.device = torch.device("cpu"),
) -> tuple[dict, float, float, float, np.ndarray]:
    """
    Train one CV fold with early stopping.

    Returns
    -------
    best_state, best_auc, best_acc, best_f1, best_attn
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    best_auc   = 0.0
    best_state = copy.deepcopy(model.state_dict())
    best_acc = best_f1 = 0.0
    best_attn  = None
    no_improve = 0

    for epoch in range(num_epochs):
        train_one_epoch(model, data, optimizer, criterion, train_mask)

        if epoch % eval_every == 0 or epoch == num_epochs - 1:
            auc, acc, f1, attn = evaluate(model, data, val_mask)
            if auc > best_auc:
                best_auc, best_acc, best_f1 = auc, acc, f1
                best_attn  = attn
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += eval_every

            if no_improve >= patience:
                break

    return best_state, best_auc, best_acc, best_f1, best_attn


# ============================================================
# Cross-validation
# ============================================================

def cross_validate(
    base_model:   nn.Module,
    X1: np.ndarray,
    X2: np.ndarray,
    X3: np.ndarray,
    y:  np.ndarray,
    k_neighbors:  int   = 5,
    num_folds:    int   = 5,
    num_epochs:   int   = 200,
    patience:     int   = 20,
    eval_every:   int   = 15,
    lr:           float = 5e-4,
    weight_decay: float = 5e-4,
    device:       torch.device = torch.device("cpu"),
) -> dict:
    """
    Stratified k-fold cross-validation.  A fresh graph is built per fold
    using only that fold's samples (no leakage).

    Returns a results dict with mean/std metrics and stacked attention maps.
    """
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=99)
    fold_aucs, fold_accs, fold_f1s = [], [], []
    attn_folds = []
    best_overall_auc   = 0.0
    best_overall_state = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X1, y), start=1):
        print(f"\n  Fold {fold}/{num_folds}")

        # --- Build per-fold graph (train + val concatenated, masks applied) ---
        data = create_multimodal_graph(
            np.concatenate([X1[train_idx], X1[val_idx]]),
            np.concatenate([X2[train_idx], X2[val_idx]]),
            np.concatenate([X3[train_idx], X3[val_idx]]),
            np.concatenate([y[train_idx],  y[val_idx]]),
            k=k_neighbors,
        ).to(device)

        n_train     = len(train_idx)
        train_mask  = torch.zeros(len(data.y), dtype=torch.bool, device=device)
        val_mask    = torch.zeros(len(data.y), dtype=torch.bool, device=device)
        train_mask[:n_train] = True
        val_mask[n_train:]   = True

        # --- Train ---
        model = copy.deepcopy(base_model).to(device)
        best_state, auc, acc, f1, attn = train_fold(
            model, data, train_mask, val_mask,
            num_epochs=num_epochs, patience=patience, eval_every=eval_every,
            lr=lr, weight_decay=weight_decay, device=device,
        )

        fold_aucs.append(auc)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        attn_folds.append(attn[val_mask.cpu()].numpy())

        if auc > best_overall_auc:
            best_overall_auc   = auc
            best_overall_state = copy.deepcopy(best_state)

        print(f"    AUC: {auc:.4f}  ACC: {acc:.4f}  F1: {f1:.4f}")

    return {
        "model_state":   best_overall_state,
        "auc":           np.mean(fold_aucs),
        "auc_std":       np.std(fold_aucs),
        "acc":           np.mean(fold_accs),
        "acc_std":       np.std(fold_accs),
        "f1":            np.mean(fold_f1s),
        "f1_std":        np.std(fold_f1s),
        "fold_results":  {"auc": fold_aucs, "acc": fold_accs, "f1": fold_f1s},
        "attention_map": np.vstack(attn_folds),
    }


# ============================================================
# Convenience wrapper (used by grid search)
# ============================================================

def train_with_params(
    X1: np.ndarray,
    X2: np.ndarray,
    X3: np.ndarray,
    y:  np.ndarray,
    params: dict,
    device: torch.device,
) -> dict:
    model = MultimodalGraphSAGE_AttentionFusion(
        in_channels1=X1.shape[1],
        in_channels2=X2.shape[1],
        in_channels3=X3.shape[1],
        hidden_channels=params["hidden_channels"],
        dropout=params["dropout"],
    )
    return cross_validate(
        model, X1, X2, X3, y,
        k_neighbors=params["k_neighbors"],
        num_folds=params.get("kfolds", 5),
        num_epochs=params["num_epochs"],
        patience=params["patience"],
        eval_every=params.get("eval_every", 15),
        lr=params["lr"],
        weight_decay=params.get("weight_decay", 5e-4),
        device=device,
    )