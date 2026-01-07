#!/usr/bin/env python3
"""
Subset TCGA Toil expression matrix to selected primary tumor cancers.

Input:
- data/raw/toil_expression_matrix.tsv.gz
- data/raw/toil_expression_metadata.tsv.gz

Output:
- data/processed/metadata_tcga_primary_6cancers.tsv
- data/processed/samples_tcga_primary_6cancers.txt
- data/processed/expression_tcga_primary_6cancers.tsv.gz
"""

import gzip
import pandas as pd
from pathlib import Path

# --------------------
# Paths
# --------------------
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPR_FILE = RAW_DIR / "toil_expression_matrix.tsv.gz"
META_FILE = RAW_DIR / "toil_expression_metadata.tsv.gz"

OUT_META = OUT_DIR / "metadata_tcga_primary_6cancers.tsv"
OUT_SAMPLES = OUT_DIR / "samples_tcga_primary_6cancers.txt"
OUT_EXPR = OUT_DIR / "expression_tcga_primary_6cancers.tsv.gz"

# --------------------
# Target cancers (EXACT strings from metadata)
# --------------------
TARGET_DISEASES = [
    "Glioblastoma Multiforme",
    "Brain Lower Grade Glioma",
    "Breast Invasive Carcinoma",
    "Lung Adenocarcinoma",
    "Colon Adenocarcinoma",
    "Skin Cutaneous Melanoma",
]

# --------------------
# Load and filter metadata
# --------------------
print("Loading metadata...")
meta = pd.read_csv(
    META_FILE,
    sep="\t",
    compression="gzip",
    encoding="latin1",
    low_memory=False
)


meta = meta[
    (meta["_study"] == "TCGA") &
    (meta["_sample_type"] == "Primary Tumor") &
    (meta["primary disease or tissue"].isin(TARGET_DISEASES))
].copy()

print("Samples after filtering:", meta.shape[0])

meta.to_csv(OUT_META, sep="\t", index=False)

samples = meta["sample"].tolist()
with open(OUT_SAMPLES, "w") as f:
    for s in samples:
        f.write(s + "\n")

sample_set = set(samples)

# --------------------
# Stream expression matrix
# --------------------
print("Subsetting expression matrix (streaming)...")

with gzip.open(EXPR_FILE, "rt") as fin, gzip.open(OUT_EXPR, "wt") as fout:
    header = fin.readline().rstrip("\n").split("\t")

    gene_col = header[0]
    expr_samples = header[1:]

    keep_idx = [i for i, s in enumerate(expr_samples) if s in sample_set]
    keep_samples = [expr_samples[i] for i in keep_idx]

    fout.write(gene_col + "\t" + "\t".join(keep_samples) + "\n")

    for line in fin:
        parts = line.rstrip("\n").split("\t")
        gene = parts[0]
        values = [parts[i + 1] for i in keep_idx]
        fout.write(gene + "\t" + "\t".join(values) + "\n")

print("Done.")
print("Wrote:")
print(" -", OUT_META)
print(" -", OUT_SAMPLES)
print(" -", OUT_EXPR)
