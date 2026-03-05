import argparse


# ============================================================
# Hyperparameter Grids
# ============================================================

grid_common = {
    # --- Model ---
    "hidden_channels": [64, 128],
    "dropout":         [0.1, 0.3, 0.4],

    # --- Optimizer ---
    "lr":              [1e-3, 2e-4, 3e-4],
    "weight_decay":    [5e-4],

    # --- Training / CV ---
    "num_epochs":      [200],
    "patience":        [20],
    "eval_every":      [15],
    "kfolds":          [5],
    # NOTE: kfolds kept separate from num_folds to avoid CLI/grid mismatch.
    # To grid-search over folds, set num_folds: [args.kfolds] in parameter_grid_from_args.
}

grid_knn_only = {
    "k_neighbors": [40],
}

grid_snf = {
    "snf_K":    [20],
    "snf_T":    [20],
    "snf_topk": [10, 15],
    "snf_mu":   [0.5],
}

grid_corr_only = {
    "cor_thresh": [0.9],
}

grid_attention_only = {
    # Uncomment to sweep attention hyperparameters:
}


# ============================================================
# Grid Builder Helpers
# ============================================================

def merge_dicts(*dicts: dict) -> dict:
    """Merge multiple param dicts, ensuring all values are lists."""
    out = {}
    for d in dicts:
        if not d:
            continue
        for k, v in d.items():
            out[k] = list(v) if isinstance(v, (list, tuple)) else [v]
    return out


def build_grid(
    common:    dict,
    extras:    list[dict] | None = None,
    snf:       dict | None       = None,
    overrides: dict | None       = None,
) -> dict:
    """
    Assemble a final param_grid from modular parts.

    Args:
        common:    Base grid shared across all scripts (grid_common).
        extras:    Optional script-specific grids (e.g. [grid_knn_only]).
        snf:       Optional SNF-specific grid; pass None if unused.
        overrides: Highest-precedence key-value overrides.

    Returns:
        Merged param_grid ready for sklearn's ParameterGrid.
    """
    parts = [common] + (extras or []) + ([snf] if snf else [])
    grid  = merge_dicts(*parts)
    if overrides:
        for k, v in overrides.items():
            grid[k] = list(v) if isinstance(v, (list, tuple)) else [v]
    return grid


def parameter_grid_from_args(args: argparse.Namespace, grid: dict) -> dict:
    """
    Pin grid values to CLI arguments, freezing them as single-element lists.
    Keys absent from args or set to None are silently skipped.

    Returns the updated grid (modified in-place and returned).
    """
    ARG_TO_GRID: dict[str, str] = {
        "num_epochs":      "num_epochs",
        "patience":        "patience",
        "eval_every":      "eval_every",
        "kfolds":          "num_folds",
        "hidden_channels": "hidden_channels",
        "dropout":         "dropout",
        "lr":              "lr",
        "weight_decay":    "weight_decay",
        "snf_K":           "snf_K",
        "snf_T":           "snf_T",
        "snf_topk":        "snf_topk",
        "snf_mu":          "snf_mu",
        "k_neighbors":     "k_neighbors",
        # "att_heads":    "att_heads",
        # "att_dropout":  "att_dropout",
    }

    for arg_name, grid_key in ARG_TO_GRID.items():
        val = getattr(args, arg_name, None)
        if val is not None and grid_key in grid:
            grid[grid_key] = [val]

    # Sync --kfolds → num_folds if the latter exists in the grid
    if "num_folds" in grid:
        kfolds_val = getattr(args, "kfolds", None)
        if kfolds_val is not None:
            grid["num_folds"] = [kfolds_val]

    return grid