import os
import subprocess



data_path_or='/Users/sevinjyolchuyeva/Downloads/postdoc/R_code/dataset_2/lusc/'
clinical_file="clinical.project-tcga-lusc.2025-07-03.json" #'clinical.project-tcga-brca.2025-10-07.json'
omics_files = ["rna_LUSC.csv", "mirna_LUSC.csv", "gene_level_methylation_global_top4500_LUSC.csv"]
output_dir = "/Users/sevinjyolchuyeva/Downloads/Pan_models/github_version/dataset/lusc2/"  # Single output directory for everything
os.makedirs(output_dir, exist_ok=True)
print("🧪 Running data_processing_stage_1")

subprocess.run([
    "python", os.path.join("data_processing_stage_1.py"),
    "--dna_file", data_path_or+omics_files[2],
    "--mirna_file", data_path_or+omics_files[1],
    "--rna_file", data_path_or+omics_files[0],
    "--clinical_file", data_path_or+clinical_file,
    "--output_path",output_dir,

], check=True)


print("\n✅  First stage complete!")


# Set all necessary paths
data_path = output_dir# contains data files
clinical_file = "clinical.csv"
omics_files = ["rna.csv", "mirna.csv", "dna_myth.csv"]
output_dir = os.path.join(data_path, "minimax_var_1")  # Single output directory for everything
os.makedirs(output_dir, exist_ok=True)
print("🧪 Running data_processing_stage_2 ")

subprocess.run([
    "python", os.path.join("data_processing_stage_2.py"),
    "--data_path", data_path,
    "--dna_file", omics_files[2],
    "--mirna_file", omics_files[1],
    "--mrna_file", omics_files[0],
    "--clinical_file", clinical_file,
    "--output_path",output_dir,

], check=True)


print("\n✅ Pipeline complete!")
