import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
import os

# ---------------------------- #
# Compute Median Absolute Deviation
# ---------------------------- #
def compute_mad(df):
    return np.abs(df - df.median()).median()

# ---------------------------- #
# Omic preprocessing function
# ---------------------------- #
def preprocess_omic(df, top_k, var_threshold=None):
    df_filtered = df.loc[:, df.mean() != 0]
    if var_threshold is not None:
        df_filtered = df_filtered.loc[:, df_filtered.var() >= var_threshold]
    mad_scores = compute_mad(df_filtered)
    top_features = mad_scores.sort_values(ascending=False).head(top_k).index
    return df_filtered[top_features]

# ---------------------------- #
# Scaling function
# ---------------------------- #
def scale_df(df):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd

def select_k_best_df(X, y, k):
    selector = SelectKBest(score_func=chi2, k=k)
    X_new = selector.fit_transform(X, y)
    selected_cols = X.columns[selector.get_support()]
    return pd.DataFrame(X_new, columns=selected_cols, index=X.index)

# ---------------------------- #
# Main processing function
# ---------------------------- #
def main(args):
    # =======================
    # Data Loading
    # =======================
    dna = pd.read_csv(args.data_path + args.dna_file)
    mirna = pd.read_csv(args.data_path + args.mirna_file)
    rna = pd.read_csv(args.data_path + args.mrna_file)
    clinical = pd.read_csv(args.data_path + args.clinical_file)
    y_train = clinical['median_value'].values

    print('Original shape',rna.shape, mirna.shape, dna.shape, clinical.shape )

    # Drop 'submitter_id' if exists
    for df in [rna, mirna, dna]:
        if 'submitter_id' in df.columns:
            df.drop(columns=['submitter_id'], inplace=True)

    # Preprocess each omic type
    rna_proc = preprocess_omic(rna, top_k=int(rna.shape[1] * args.feature_frac), var_threshold=args.var_thresh)
    dna_proc = preprocess_omic(dna, top_k=int(dna.shape[1] * args.feature_frac), var_threshold=args.var_thresh)
    mirna_proc = preprocess_omic(mirna, top_k=int(mirna.shape[1] * args.feature_frac), var_threshold=args.var_thresh)

    # Scale each dataset
    rna_scaled = scale_df(rna_proc)
    dna_scaled = scale_df(dna_proc)
    mirna_scaled = scale_df(mirna_proc)

    # =======================
    # Feature Selection
    # =======================

    rna_scaled = select_k_best_df(rna_scaled, y_train, args.top_k)
    dna_scaled = select_k_best_df(dna_scaled, y_train, args.top_k)
    mirna_scaled = select_k_best_df(mirna_scaled, y_train, args.top_k)


    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Save outputs
    omics_1 = os.path.basename(args.mrna_file).split("_")[0]
    omics_2 = os.path.basename(args.mirna_file).split("_")[0]
    omics_3 = os.path.basename(args.dna_file).split("_")[0]
    clinical_nam = os.path.basename(args.clinical_file).split("_")[0]

    rna_scaled.to_csv(os.path.join(args.output_path, f"{omics_1}"), index=False)
    dna_scaled.to_csv(os.path.join(args.output_path, f"{omics_3}.csv"), index=False)
    mirna_scaled.to_csv(os.path.join(args.output_path, f"{omics_2}"), index=False)
    clinical.to_csv(os.path.join(args.output_path, f"{clinical_nam}"), index=False)


    # Report -Final shape
    print("Saved processed data to:", args.output_path)
    print("mRNA shape:", rna_scaled.shape)
    print("DNA methylation shape:", dna_scaled.shape)
    print("miRNA shape:", mirna_scaled.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess omics data with MAD selection and scaling.")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to input directory containing rna_br.csv, mirna_br.csv, dna_myth_br.csv, clinical_br.csv')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Directory to save processed output files')
    parser.add_argument('--feature_frac', type=float, default=0.5,
                        help='Fraction of top features to retain based on MAD (e.g., 0.5 = keep top 50%)')
    parser.add_argument('--var_thresh', type=float, default=0.001,
                        help='Variance threshold for filtering low-variance features')

    parser.add_argument("--dna_file", type=str, required=True, help="DNA methylation CSV file")
    parser.add_argument("--mirna_file", type=str, required=True, help="miRNA CSV file")
    parser.add_argument("--mrna_file", type=str, required=True, help="mRNA CSV file")
    parser.add_argument("--clinical_file", type=str, required=True, help="Clinical CSV file")
    parser.add_argument("--top_k", type=int, default=100, help="Top K features to select using chi2")
    args = parser.parse_args()

    main(args)
