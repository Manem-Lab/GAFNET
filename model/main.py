import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.model_selection import ParameterGrid

from hyperparams import (
    build_grid,
    grid_common,
    grid_knn_only,
    parameter_grid_from_args,
)
from train import train_with_params

# ============================================================
# Reproducibility
# ============================================================
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False


# ============================================================
# Helpers
# ============================================================

def resolve_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def load_dataset(data_path: str, dataset: str, args) -> tuple:
    """Load all four modalities for a single dataset folder."""
    base = f"{data_path}{dataset}minimax_var_1/"
    mrna     = pd.read_csv(base + args.mrna_file).values
    mirna    = pd.read_csv(base + args.mirna_file).values
    dna_myth = pd.read_csv(base + args.dna_file).values
    clinical = pd.read_csv(base + args.clinical_file)
    y        = clinical["median_value"].values
    return mrna, mirna, dna_myth, y


def run_grid_search(mrna, mirna, dna_myth, y, param_grid, device) -> tuple[dict, dict]:
    """Sweep param_grid and return (best_metrics, best_params)."""
    configs = list(ParameterGrid(param_grid))
    best_metrics, best_params, best_auc = None, None, -1.0

    for i, params in enumerate(configs, start=1):
        print(f"  [{i}/{len(configs)}] {params}")
        metrics = train_with_params(mrna, mirna, dna_myth, y, params, device)
        if metrics["auc"] > best_auc:
            best_auc     = metrics["auc"]
            best_metrics = metrics
            best_params  = params

    return best_metrics, best_params


def save_results(tag: str, output_dir: str, metrics: dict, best_params: dict) -> None:
    """Persist attention map, violin plot, and CSV result for one dataset."""
    attn_map = metrics["attention_map"]

    # Attention map array
    np.save(f"{output_dir}/{tag}_atten_map.npy", attn_map)

    # Violin plot
    df_attn = pd.DataFrame(attn_map, columns=["mRNA", "miRNA", "DNAmeth"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(data=df_attn, ax=ax)
    ax.set_title(f"{tag.upper()} — Modality Attention Weights")
    ax.set_ylabel("Attention Weight")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/{tag}_attn_dist.png", dpi=150)
    plt.close(fig)

    # CSV summary
    result_row = {
        "dataset":     tag.upper(),
        "best_params": str(best_params),
        "auc":         round(metrics["auc"],     4),
        "auc_std":     round(metrics["auc_std"], 4),
        "f1":          round(metrics["f1"],      4),
        "f1_std":      round(metrics["f1_std"],  4),
        "acc":         round(metrics["acc"],      4),
        "acc_std":     round(metrics["acc_std"],  4),
    }
    pd.DataFrame([result_row]).to_csv(f"{output_dir}/{tag}_result.csv", index=False)
    print(f"  Saved results → {output_dir}/{tag}_*")
    return result_row


# ============================================================
# Entry point
# ============================================================

DATASETS = [
    "stad", "lusc", "prad", "lihc", "kirp",
    "luad", "kirc", "brca", "ucec", "lgg",
    "hnsc", "paad", "skcm", "ov",   "thca",
    "cesc", "blca", "coad",]




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multimodal GraphSAGE — Pan-Cancer Classification"
    )
    parser.add_argument("--device",        type=str, choices=["auto","cuda","cpu","mps"], default="cpu")
    parser.add_argument("--amp",           action="store_true", help="Mixed precision (CUDA only)")
    parser.add_argument("--data_path",     type=str, default="/Users/sevinjyolchuyeva/Downloads/Pan_models/github_version/dataset/")
    parser.add_argument("--dna_file",      type=str, default="dna.csv")
    parser.add_argument("--mirna_file",    type=str, default="mirna.csv")
    parser.add_argument("--mrna_file",     type=str, default="rna.csv")
    parser.add_argument("--clinical_file", type=str, default="clinical.csv")
    parser.add_argument("--top_k",         type=int, default=100)
    parser.add_argument("--k_neighbors",   type=int, default=None)
    parser.add_argument("--kfolds",        type=int, default=5)
    parser.add_argument("--output_dir",    type=str, default="/Users/sevinjyolchuyeva/Downloads/Pan_models/github_version/github_code/github",
                        help="Directory to save attention maps, plots, and CSVs")
    parser.add_argument("--datasets",      type=str, nargs="+", default=["stad"],
                        help="One or more dataset names to run, e.g. --datasets stad brca luad")
    return parser.parse_args()
    ##return parser.parse_args(args=[])   # swap for parse_args() when running from CLI
    return parser.parse_args()



def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device}\n")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    param_grid = build_grid(common=grid_common, extras=[grid_knn_only])
    param_grid = parameter_grid_from_args(args, param_grid)

    all_results = []

    for dataset in args.datasets:
        tag = dataset.upper()
        print(f"\n{'='*60}\n  {tag}\n{'='*60}")

        mrna, mirna, dna_myth, y = load_dataset(args.data_path, dataset + "/", args)
        print(f"  mRNA {mrna.shape}  miRNA {mirna.shape}  DNA {dna_myth.shape}  y {y.shape}")

        best_metrics, best_params = run_grid_search(
            mrna, mirna, dna_myth, y, param_grid, device
        )
        row = save_results(dataset, args.output_dir, best_metrics, best_params)
        all_results.append(row)

        print(f"\n  Best → AUC {row['auc']} ± {row['auc_std']}  "
              f"F1 {row['f1']} ± {row['f1_std']}  ACC {row['acc']} ± {row['acc_std']}")

    print(f"\n{'='*60}\n  All {len(args.datasets)} datasets complete.\n{'='*60}")


if __name__ == "__main__":
    main()