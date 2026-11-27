#!/usr/bin/env python3
# scripts/create_single_sample.py
import argparse, numpy as np, pathlib
p = argparse.ArgumentParser()
p.add_argument("--dataset-dir", required=True)
p.add_argument("--index", type=int, default=0)
p.add_argument("--out-dir", default="uploads")
args = p.parse_args()
out = pathlib.Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
D = pathlib.Path(args.dataset_dir)
for name in ("X_genes.npy","X_brain.npy","X_snps.npy"):
    path = D / name
    if not path.exists():
        print("skip missing", path)
        continue
    X = np.load(str(path))
    s = X[[args.index],:]
    dest = out / ("patient_" + name.replace("X_","").replace(".npy","") + "_raw.npy")
    np.save(str(dest), s)
    print("wrote", dest, s.shape)
