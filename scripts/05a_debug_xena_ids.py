from pathlib import Path
import pandas as pd
import xenaPython as xena

SAMPLES_FILE = Path("data/processed/samples_tcga_primary_6cancers.txt")

HUB = "https://gdc.xenahubs.net"
DATASET = "TCGA_phenotype_denseDataOnlyDownload"
FIELD = "OS.time"

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def fetch(ids):
    _, values = xena.dataset_probe_values(HUB, DATASET, ids, [FIELD])
    return pd.Series(values[0], index=ids)

def score(series):
    s = to_num(series)
    return int(s.notna().sum()), float(s.notna().mean())

def main():
    samples = [s.strip() for s in open(SAMPLES_FILE) if s.strip()][:100]
    patient12 = [s[:12] for s in samples]
    sample15 = [s[:15] for s in samples]  # TCGA-XX-YYYY-01
    # also try keeping full
    full = samples

    for name, ids in [("full_sample", full), ("patient12", patient12), ("sample15", sample15)]:
        try:
            ser = fetch(ids)
            n, frac = score(ser)
            print(f"{name}: non-missing OS.time = {n}/{len(ids)} ({frac:.2%})")
            # show a few numeric examples
            ex = to_num(ser).dropna().head(5)
            print(" examples:", ex.to_dict())
        except Exception as e:
            print(f"{name}: failed with {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
