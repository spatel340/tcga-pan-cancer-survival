import gzip
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------
# Paths
# ------------------
EXPR_FILE = Path("data/processed/expression_tcga_primary_6cancers.tsv.gz")
META_FILE = Path("data/processed/metadata_tcga_primary_6cancers.tsv")

OUT_FIG = Path("figures")
OUT_TAB = Path("results/tables")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

# ------------------
# Parameters
# ------------------
TOP_VAR_GENES = 5000
RANDOM_STATE = 42

# ------------------
# Load metadata
# ------------------
print("Loading metadata...")
meta = pd.read_csv(META_FILE, sep="\t")
meta = meta.set_index("sample")

# ------------------
# Load expression (streaming-safe)
# ------------------
print("Loading expression matrix...")
with gzip.open(EXPR_FILE, "rt") as f:
    expr = pd.read_csv(f, sep="\t", index_col=0)

# ensure column order matches metadata
# ensure we only keep samples present in BOTH expression and metadata
common = meta.index.intersection(expr.columns)
missing_in_expr = meta.index.difference(expr.columns)
missing_in_meta = expr.columns.difference(meta.index)

print(f"Samples in metadata: {meta.shape[0]}")
print(f"Samples in expression: {expr.shape[1]}")
print(f"Common samples: {len(common)}")
print(f"Missing in expression (will drop): {len(missing_in_expr)}")
print(f"Missing in metadata (will drop): {len(missing_in_meta)}")

# subset and reorder consistently
meta = meta.loc[common]
expr = expr.loc[:, common]


print(f"Expression shape: {expr.shape}")

# ------------------
# Log transform
# ------------------
expr_log = np.log2(expr + 1)

# ------------------
# Select highly variable genes
# ------------------
print(f"Selecting top {TOP_VAR_GENES} variable genes...")
gene_var = expr_log.var(axis=1)
top_genes = gene_var.sort_values(ascending=False).head(TOP_VAR_GENES).index
expr_hvg = expr_log.loc[top_genes]

# ------------------
# Scale
# ------------------
scaler = StandardScaler()
X = scaler.fit_transform(expr_hvg.T)

# ------------------
# PCA
# ------------------
print("Running PCA...")
pca = PCA(n_components=20, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame(
    X_pca[:, :5],
    columns=[f"PC{i+1}" for i in range(5)],
    index=expr_hvg.columns
)

pca_df = pca_df.join(meta)

# ------------------
# Save variance explained
# ------------------
var_df = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
    "variance_explained": pca.explained_variance_ratio_
})
var_df.to_csv(OUT_TAB / "pca_variance_explained.tsv", sep="\t", index=False)

# ------------------
# Plot PCA
# ------------------
print("Plotting PCA...")
plt.figure(figsize=(9, 7))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="primary disease or tissue",
    s=30,
    linewidth=0,
    alpha=0.85
)
plt.title("TCGA Pan-Cancer PCA (Top 5k Variable Genes)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
plt.savefig(OUT_FIG / "pca_by_cancer.png", dpi=300)
plt.close()

print("Saved:")
print(f" - {OUT_FIG / 'pca_by_cancer.png'}")
print(f" - {OUT_TAB / 'pca_variance_explained.tsv'}")
