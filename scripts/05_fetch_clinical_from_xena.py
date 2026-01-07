# scripts/05_fetch_clinical_from_xena.py
from pathlib import Path
import pandas as pd

SAMPLES_FILE = Path("data/processed/samples_tcga_primary_6cancers.txt")
META_FILE = Path("data/processed/metadata_tcga_primary_6cancers.tsv")
SURV_FILE = Path("data/raw/pancanatlas_survival.tsv")
OUT_FILE = Path("results/tables/survival_clinical_6cancers.tsv")

def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load our 6-cancer sample list and metadata
    samples = [s.strip() for s in open(SAMPLES_FILE) if s.strip()]
    meta = pd.read_csv(META_FILE, sep="\t", encoding="latin1").set_index("sample")
    samples = [s for s in samples if s in meta.index]

    if not SURV_FILE.exists():
        raise FileNotFoundError(
            f"Missing {SURV_FILE}. Download with:\n"
            "wget -O data/raw/pancanatlas_survival.tsv "
            "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/"
            "Survival_SupplementalTable_S1_20171025_xena_sp"
        )

    # Load survival table (sample-level)
    surv = pd.read_csv(SURV_FILE, sep="\t")
    sample_col = pick_col(surv, ["sample", "Sample", "sampleID"])
    if sample_col is None:
        raise RuntimeError(f"Could not find sample id column in survival file. Columns: {list(surv.columns)[:30]}")

    surv[sample_col] = surv[sample_col].astype(str).str.strip()

    # Subset survival to our samples (direct join on sample IDs)
    surv_sub = surv[surv[sample_col].isin(samples)].copy()
    surv_sub = surv_sub.set_index(sample_col)

    # Identify OS columns (these exist in your file: OS and OS.time)
    os_time_col = pick_col(surv_sub, ["OS.time", "OS_time", "os_time"])
    os_event_col = pick_col(surv_sub, ["OS", "OS_event", "os", "vital_status"])

    if os_time_col is None or os_event_col is None:
        raise RuntimeError(
            f"Could not find OS columns. Have OS_time_col={os_time_col}, OS_event_col={os_event_col}. "
            f"Columns: {list(surv_sub.columns)[:40]}"
        )

    out = pd.DataFrame(index=surv_sub.index)

    # cancer label from your Toil metadata subset
    out["cancer"] = meta.loc[out.index, "primary disease or tissue"].astype(str)

    # normalize OS_time and OS_event
    out["OS_time"] = pd.to_numeric(surv_sub[os_time_col], errors="coerce")

    ev = surv_sub[os_event_col]
    if ev.dtype == object:
        vs = ev.astype(str).str.lower().str.strip()
        out["OS_event"] = vs.map({"dead": 1, "deceased": 1, "alive": 0, "living": 0})
    else:
        out["OS_event"] = pd.to_numeric(ev, errors="coerce")
        out.loc[out["OS_event"] > 1, "OS_event"] = pd.NA

    # optional covariates if present in survival file
    age_col = pick_col(surv_sub, ["age_at_initial_pathologic_diagnosis", "age"])
    sex_col = pick_col(surv_sub, ["gender", "_gender", "sex"])
    out["age"] = pd.to_numeric(surv_sub[age_col], errors="coerce") if age_col else pd.NA
    out["sex"] = surv_sub[sex_col].astype(str) if sex_col else pd.NA

    # keep rows with OS_time
    out = out.dropna(subset=["OS_time"])

    out.to_csv(OUT_FILE, sep="\t", index_label="sample")
    print("Wrote:", OUT_FILE)
    print("Rows:", out.shape[0])
    print("Events:", int(out["OS_event"].fillna(0).sum()))
    print("Missing OS_event:", int(out["OS_event"].isna().sum()))

if __name__ == "__main__":
    main()
