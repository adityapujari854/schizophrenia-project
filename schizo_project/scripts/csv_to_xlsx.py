import pandas as pd
from pathlib import Path

p = Path("data/processed/genes/gene_sample_metadata_inferred.csv")
df = pd.read_csv(p)
out = p.with_suffix(".xlsx")
df.to_excel(out, index=False)
print("Wrote:", out)
