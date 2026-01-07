import gzip
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

EXPR_FILE = Path("data/processed/expression_tcga_primary_6cancers.tsv.gz")
META_FILE = Path("data/processed/metadata_tcga_primary_6cancers.tsv")
SIG_FILE  = Path("results/tables/one_vs_rest_top_genes.tsv")

OUT_FIG = Path("figures")
OUT_FIG.mkdir(parents=True, exist_ok=True)

TOP_PER_CANCER = 30  # keep heatmap readable

def main():
    meta = pd.read_csv(META_FILE, sep="\t", encoding="latin1").set_index("sample")
    sig = pd.read_csv(SIG_FILE, sep="\t")

    # choose top genes per cancer
    sig_small = (sig.sort_values(["cancer","rank"])
                   .groupby("cancer")
                   .head(TOP_PER_CANCER))
    genes = sig_small["gene"].unique().tolist()

    with gzip.open(EXPR_FILE, "rt") as f:
        expr = pd.read_csv(f, sep="\t", index_col=0)

    common = meta.index.intersection(expr.columns)
    meta = meta.loc[common]
    expr = expr.loc[:, common]

    expr_log = np.log2(expr + 1)
    expr_sig = expr_log.loc[expr_log.index.intersection(genes)]

    # z-score genes across samples for heatmap
    X = expr_sig.values
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    expr_z = pd.DataFrame(X, index=expr_sig.index, columns=expr_sig.columns)

    # order samples by cancer
    meta["cancer"] = meta["primary disease or tissue"].astype(str)
    order = meta.sort_values("cancer").index
    expr_z = expr_z[order]
    cancer_order = meta.loc[order, "cancer"]

    # colors
    cancers = cancer_order.unique().tolist()
    palette = sns.color_palette("tab10", n_colors=len(cancers))
    lut = dict(zip(cancers, palette))
    col_colors = cancer_order.map(lut)

    plt.figure(figsize=(12, 10))
    sns.heatmap(expr_z, cmap="vlag", center=0, xticklabels=False, yticklabels=False,
                cbar_kws={"label":"Gene z-score"})
    plt.title(f"Pan-cancer signature heatmap (top {TOP_PER_CANCER} genes per cancer)")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "signature_heatmap.png", dpi=300)
    plt.close()

    # also save legend separately
    plt.figure(figsize=(8, 1.2))
    for c in cancers:
        plt.scatter([], [], color=lut[c], label=c)
    plt.legend(ncol=3, frameon=False, loc="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "signature_heatmap_legend.png", dpi=300)
    plt.close()

    print("Saved:")
    print(" - figures/signature_heatmap.png")
    print(" - figures/signature_heatmap_legend.png")

if __name__ == "__main__":
    main()
