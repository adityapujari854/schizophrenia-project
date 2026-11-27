import numpy as np
import os

# Your full .npy file path
npy_path = r"C:\Users\adity\Documents\Schizophrenia Project\schizo_project\data\synthetic_signal_test\X_brain_signal.npy"

# Auto-generate CSV path
csv_path = os.path.splitext(npy_path)[0] + ".csv"

# Load .npy
data = np.load(npy_path)

# Save as CSV
np.savetxt(csv_path, data, delimiter=",")

print(f"Converted '{npy_path}' → '{csv_path}'")
