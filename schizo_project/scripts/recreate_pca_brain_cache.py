# scripts/recreate_pca_brain_cache.py
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path

X = np.load("data/synthetic_signal_targeted/X_brain.npy")
print("Loaded:", X.shape)

pca = PCA(n_components=500)
X_pca = pca.fit_transform(X)

Path("results").mkdir(exist_ok=True)
np.savez("results/pca_brain.npz",
         components=pca.components_,
         mean=pca.mean_,
         explained_variance=pca.explained_variance_,
         explained_variance_ratio=pca.explained_variance_ratio_)
print("✅ Saved PCA cache to results/pca_brain.npz")
