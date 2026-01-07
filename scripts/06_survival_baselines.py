from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lifelines import KaplanMeierFitter, CoxPHFitter

CLIN = Path("results/tables/survival_clinical_6cancers.tsv")
OUT_TAB = Path("results/tables")
OUT_FIG = Path("figures")
OUT_TAB.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(CLIN, sep="\t").dropna(subset=["OS_time"])
    df["OS_time"] = pd.to_numeric(df["OS_time"], errors="coerce")
    df["OS_event"] = pd.to_numeric(df["OS_event"], errors="coerce").fillna(0).astype(int)

    # Basic cleaning
    df = df.dropna(subset=["cancer", "OS_time"])
    df = df[df["OS_time"] >= 0]

    # KM by cancer
    plt.figure(figsize=(10, 7))
    kmf = KaplanMeierFitter()

    cancers = df["cancer"].value_counts().index.tolist()
    med_rows = []
    for cancer in cancers:
        sub = df[df["cancer"] == cancer]
        if sub.shape[0] < 20:
            continue
        kmf.fit(sub["OS_time"], event_observed=sub["OS_event"], label=f"{cancer} (n={sub.shape[0]})")
        kmf.plot(ci_show=False, linewidth=2)
        med_rows.append({"cancer": cancer, "n": sub.shape[0], "median_OS_time": kmf.median_survival_time_})

    plt.title("Kaplan–Meier survival by cancer (TCGA primary tumors)")
    plt.xlabel("Days")
    plt.ylabel("Survival probability")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "km_by_cancer.png", dpi=300)
    plt.close()

    pd.DataFrame(med_rows).to_csv(OUT_TAB / "km_median_survival_by_cancer.tsv", sep="\t", index=False)

    # Baseline Cox: cancer + age + sex (if present)
    cox = df.copy()
    # one-hot cancer
    cox = pd.get_dummies(cox, columns=["cancer", "sex"], drop_first=True)

    # keep only numeric columns + duration/event
    keep = ["OS_time", "OS_event"] + [c for c in cox.columns if c not in ["sample"] and c not in ["OS_time", "OS_event"]]
    cox = cox[keep]

    # Remove columns with all NA or zero variance
    for c in list(cox.columns):
        if c in ["OS_time", "OS_event"]:
            continue
        if cox[c].isna().all() or cox[c].nunique(dropna=True) <= 1:
            cox = cox.drop(columns=[c])
    cox = cox.dropna()

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox, duration_col="OS_time", event_col="OS_event")

    out = cph.summary.reset_index().rename(columns={"index": "term"})
    out.to_csv(OUT_TAB / "cox_baseline.tsv", sep="\t", index=False)

    print("Saved:")
    print(" - figures/km_by_cancer.png")
    print(" - results/tables/km_median_survival_by_cancer.tsv")
    print(" - results/tables/cox_baseline.tsv")
    print("Rows used in Cox:", cox.shape[0])

if __name__ == "__main__":
    main()
