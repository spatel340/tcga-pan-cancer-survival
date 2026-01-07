import gzip
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

EXPR_FILE = Path("data/processed/expression_tcga_primary_6cancers.tsv.gz")
META_FILE = Path("data/processed/metadata_tcga_primary_6cancers.tsv")

OUT_DIR = Path("results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 200          # save top N genes per cancer for the signature heatmap
MIN_SAMPLES = 30     # guardrail

def load_data():
    meta = pd.read_csv(META_FILE, sep="\t", encoding="latin1").set_index("sample")
    with gzip.open(EXPR_FILE, "rt") as f:
        expr = pd.read_csv(f, sep="\t", index_col=0)
    common = meta.index.intersection(expr.columns)
    meta = meta.loc[common]
    expr = expr.loc[:, common]
    return expr, meta

def one_vs_rest(expr_log, labels, cancer):
    idx_pos = (labels == cancer).values
    idx_neg = (labels != cancer).values
    X_pos = expr_log.loc[:, idx_pos].values
    X_neg = expr_log.loc[:, idx_neg].values

    # effect size: mean difference in log space
    mean_pos = X_pos.mean(axis=1)
    mean_neg = X_neg.mean(axis=1)
    log2fc = mean_pos - mean_neg

    # Welch t-test per gene
    stat, p = ttest_ind(X_pos.T, X_neg.T, equal_var=False, nan_policy="omit")
    p = np.asarray(p)

    # FDR
    padj = multipletests(p, method="fdr_bh")[1]

    out = pd.DataFrame({
        "gene": expr_log.index,
        "cancer": cancer,
        "n_pos": idx_pos.sum(),
        "n_neg": idx_neg.sum(),
        "log2fc": log2fc,
        "pval": p,
        "padj_fdr": padj
    }).sort_values(["padj_fdr", "pval"], ascending=True)

    return out

def main():
    expr, meta = load_data()
    print("Expression:", expr.shape)

    labels = meta["primary disease or tissue"].astype(str)
    cancers = labels.value_counts().index.tolist()
    print("Cancer counts:\n", labels.value_counts())

    # log2(TPM+1)
    expr_log = np.log2(expr + 1)

    all_res = []
    sig_rows = []

    for cancer in cancers:
        n = (labels == cancer).sum()
        if n < MIN_SAMPLES:
            print(f"Skipping {cancer} (n={n} < {MIN_SAMPLES})")
            continue

        print("DE:", cancer)
        res = one_vs_rest(expr_log, labels, cancer)
        all_res.append(res)

        top = res.head(TOP_N).copy()
        top["rank"] = np.arange(1, top.shape[0] + 1)
        sig_rows.append(top[["cancer", "rank", "gene", "log2fc", "padj_fdr"]])

    de_all = pd.concat(all_res, ignore_index=True)
    de_all.to_csv(OUT_DIR / "one_vs_rest_de_all.tsv.gz", sep="\t", index=False, compression="gzip")

    sig = pd.concat(sig_rows, ignore_index=True)
    sig.to_csv(OUT_DIR / "one_vs_rest_top_genes.tsv", sep="\t", index=False)

    print("Wrote:")
    print(" -", OUT_DIR / "one_vs_rest_de_all.tsv.gz")
    print(" -", OUT_DIR / "one_vs_rest_top_genes.tsv")

if __name__ == "__main__":
    main()
