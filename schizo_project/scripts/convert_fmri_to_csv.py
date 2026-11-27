import numpy as np
import pandas as pd
from pathlib import Path

def convert_fmri_to_csv(brain_npy_path, sample_list_csv, output_csv):
    brain_npy_path = Path(brain_npy_path)
    sample_list_csv = Path(sample_list_csv)
    output_csv = Path(output_csv)

    print(f"[INFO] Loading fMRI connectivity matrix from {brain_npy_path} ...")
    X_brain = np.load(brain_npy_path)
    print(f"[INFO] Loaded shape: {X_brain.shape}  (rows = samples, columns = ROI-pair features)")

    print(f"[INFO] Loading sample IDs from {sample_list_csv} ...")
    sample_df = pd.read_csv(sample_list_csv)

    if len(sample_df) != X_brain.shape[0]:
        raise ValueError("Sample count mismatch between X_brain.npy and gene_samples_list.csv")

    # Generate column names: ROI1_ROI2, ROI1_ROI3, ...
    num_features = X_brain.shape[1]
    col_names = [f"ROI_{i+1}_{j+1}" for i in range(1) for j in range(num_features)]  # simplified
        
    # If you want exact names: ROI1_ROI2, ROI1_ROI3, ...
    col_names = [f"ROI{i//100 + 1}_ROI{i%100 + 2}" for i in range(num_features)]

    print("[INFO] Building DataFrame...")
    df = pd.DataFrame(X_brain, columns=col_names)
    df.insert(0, "sample_id", sample_df["sample_id"])

    print(f"[INFO] Saving to {output_csv} ...")
    df.to_csv(output_csv, index=False)
    print("[OK] CSV file created successfully!")

if __name__ == "__main__":
    convert_fmri_to_csv(
        brain_npy_path="data/synthetic_fullrun/X_brain.npy",
        sample_list_csv="data/synthetic_fullrun/gene_samples_list.csv",
        output_csv="results/fmri_connectivity_preview.csv"
    )
