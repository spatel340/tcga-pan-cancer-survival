#!/usr/bin/env python3
"""
07_expression_survival_model.py

Elastic-net Cox survival model using gene expression (train-only feature selection)
on TCGA primary tumor subset (6 cancers).

Inputs
------
data/processed/expression_tcga_primary_6cancers.tsv.gz
data/processed/metadata_tcga_primary_6cancers.tsv
results/tables/survival_clinical_6cancers.tsv   (created from pancanatlas survival)

Outputs
-------
results/tables/elasticnet_cv_summary.csv
results/tables/elasticnet_selected_features.csv
results/tables/elasticnet_summary.csv
figures/km_elasticnet_high_low_test.png
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index


# -----------------------------
# Config (tune for laptop speed)
# -----------------------------
EXPR_FILE = Path("data/processed/expression_tcga_primary_6cancers.tsv.gz")
META_FILE = Path("data/processed/metadata_tcga_primary_6cancers.tsv")
SURV_FILE = Path("results/tables/survival_clinical_6cancers.tsv")

OUT_CV = Path("results/tables/elasticnet_cv_summary.csv")
OUT_SELECTED = Path("results/tables/elasticnet_selected_features.csv")
OUT_SUMMARY = Path("results/tables/elasticnet_summary.csv")
OUT_FIG = Path("figures/km_elasticnet_high_low_test.png")

RANDOM_SEED = 7
TEST_FRAC = 0.20

TOPN_VARIABLE_GENES = 1000  # keep <= 2000 on a laptop
N_SPLITS = 3                # 3-fold CV is fine for a flagship repo
PENALIZERS = [1.0, 3.0, 10.0, 30.0]
L1_RATIOS = [0.5, 0.8]      # elastic net mix

USE_AGE = True              # include age as clinical covariate


# -----------------------------
# Utilities
# -----------------------------
def ensure_parents(*paths: Path) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)


def load_expression_matrix(expr_path: Path) -> pd.DataFrame:
    """
    Returns samples x genes (rows=samples, cols=genes).
    Handles common Xena-style format: first column is gene id, remaining columns are samples.
    """
    df = pd.read_csv(expr_path, sep="\t", compression="gzip")
    if df.shape[1] < 2:
        raise RuntimeError(f"Expression file looks empty: {expr_path}")

    first_col = df.columns[0]

    # If first column is gene-like and remaining columns are samples, transpose
    # Typical: first_col == "sample" but actually contains gene names.
    gene_ids = df[first_col].astype(str).values
    mat = df.drop(columns=[first_col])
    mat.index = gene_ids

    # Convert to numeric (coerce bad values)
    mat = mat.apply(pd.to_numeric, errors="coerce")

    # Samples x genes
    expr = mat.T
    expr.index.name = "sample"

    return expr


def select_top_variable_genes(X_train: pd.DataFrame, top_n: int) -> list[str]:
    """
    Train-only feature selection by variance (after log1p).
    """
    if top_n <= 0:
        raise ValueError("top_n must be > 0")

    # Use variance ignoring NaNs
    v = X_train.var(axis=0, skipna=True)
    v = v.replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        raise RuntimeError("No valid genes after variance computation.")
    top = v.sort_values(ascending=False).head(min(top_n, v.shape[0])).index.tolist()
    return top


def stratified_train_test_split(samples: np.ndarray, strata: np.ndarray, frac_test: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic stratified split by 'strata' labels.
    """
    rng = np.random.default_rng(seed)
    test_mask = np.zeros(len(samples), dtype=bool)

    df = pd.DataFrame({"sample": samples, "strata": strata})
    for _, sub in df.groupby("strata"):
        idx = sub.index.to_numpy()
        n_test = int(np.ceil(len(idx) * frac_test))
        if n_test <= 0:
            continue
        pick = rng.choice(idx, size=n_test, replace=False)
        test_mask[pick] = True

    train = samples[~test_mask]
    test = samples[test_mask]
    return train, test


def km_plot_high_low(time: pd.Series, event: pd.Series, risk: pd.Series, out_png: Path, title: str) -> None:
    """
    Kaplan-Meier plot for high vs low risk (median split).
    """
    import matplotlib.pyplot as plt

    ensure_parents(out_png)
    kmf = KaplanMeierFitter()

    med = np.nanmedian(risk.values)
    group = np.where(risk.values >= med, "High risk", "Low risk")

    fig, ax = plt.subplots(figsize=(7, 5))
    for g in ["Low risk", "High risk"]:
        m = (group == g)
        kmf.fit(durations=time[m], event_observed=event[m], label=g)
        kmf.plot_survival_function(ax=ax, ci_show=False)

    ax.set_title(title)
    ax.set_xlabel("Days")
    ax.set_ylabel("Survival probability")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def safe_standardize(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    StandardScaler that cannot produce NaNs/inf outputs, even with constant columns.
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    train_arr = train.to_numpy(dtype=np.float32)
    test_arr = test.to_numpy(dtype=np.float32)

    train_scaled = scaler.fit_transform(train_arr)
    test_scaled = scaler.transform(test_arr)

    X_train_scaled = pd.DataFrame(train_scaled, index=train.index, columns=train.columns)
    X_test_scaled = pd.DataFrame(test_scaled, index=test.index, columns=test.columns)

    # Safety: if scaling introduced NaNs/inf, neutralize them
    X_train_scaled = X_train_scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test_scaled = X_test_scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X_train_scaled, X_test_scaled


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ensure_parents(OUT_CV, OUT_SELECTED, OUT_SUMMARY, OUT_FIG)

    # Load metadata + survival
    meta = pd.read_csv(META_FILE, sep="\t", encoding="latin1")
    if "sample" not in meta.columns:
        raise RuntimeError(f"Expected 'sample' column in {META_FILE}")
    meta = meta.set_index("sample")

    surv = pd.read_csv(SURV_FILE, sep="\t").set_index("sample").copy()

    # Hard type coercion
    surv["OS_time"] = pd.to_numeric(surv["OS_time"], errors="coerce")
    surv["OS_event"] = pd.to_numeric(surv["OS_event"], errors="coerce")
    surv = surv.dropna(subset=["OS_time", "OS_event", "cancer"])
    surv = surv[surv["OS_time"] > 0]
    surv["OS_event"] = (surv["OS_event"] > 0).astype(int)

    if "age" in surv.columns:
        surv["age"] = pd.to_numeric(surv["age"], errors="coerce")

    # Load expression (samples x genes)
    expr = load_expression_matrix(EXPR_FILE)

    # Keep common samples
    common = sorted(set(expr.index) & set(meta.index) & set(surv.index))
    if len(common) < 200:
        raise RuntimeError(f"Too few common samples after intersection: {len(common)}")
    expr = expr.loc[common]
    surv = surv.loc[common]
    meta = meta.loc[common]

    # Log1p transform (common for TPM-like features); keep float32
    X_full = np.log1p(expr.astype(np.float32))

    # Build strata for split: cancer + event
    strata = (surv["cancer"].astype(str) + "_e" + surv["OS_event"].astype(str)).values
    samples = np.array(common)

    train_samples, test_samples = stratified_train_test_split(samples, strata, frac_test=TEST_FRAC, seed=RANDOM_SEED)

    # Final index order
    train_samples = np.array([s for s in train_samples if s in X_full.index])
    test_samples = np.array([s for s in test_samples if s in X_full.index])

    # Split
    X_train_full = X_full.loc[train_samples].copy()
    X_test_full = X_full.loc[test_samples].copy()

    y_train = surv.loc[train_samples, ["OS_time", "OS_event", "cancer"]].copy()
    y_test = surv.loc[test_samples, ["OS_time", "OS_event", "cancer"]].copy()

    # Feature selection using TRAIN only
    top_genes = select_top_variable_genes(X_train_full, TOPN_VARIABLE_GENES)
    X_train = X_train_full[top_genes].copy()
    X_test = X_test_full[top_genes].copy()

    # Optional clinical covariate: age (train/test separately, but impute using train median)
    if USE_AGE and "age" in surv.columns:
        age_train = pd.to_numeric(surv.loc[train_samples, "age"], errors="coerce").astype(np.float32)
        age_test = pd.to_numeric(surv.loc[test_samples, "age"], errors="coerce").astype(np.float32)
        X_train = pd.concat([X_train, age_train.rename("age")], axis=1)
        X_test = pd.concat([X_test, age_test.rename("age")], axis=1)

    # Robust cleanup + train-only imputation
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    train_median = X_train.median(axis=0, skipna=True).fillna(0.0)
    X_train = X_train.fillna(train_median)
    X_test = X_test.fillna(train_median)

    # Drop constant columns in TRAIN
    nunique = X_train.nunique(axis=0)
    keep_cols = nunique[nunique > 1].index.tolist()
    X_train = X_train[keep_cols]
    X_test = X_test[keep_cols]

    # Standardize
    X_train_scaled, X_test_scaled = safe_standardize(X_train, X_test)

    # Debug checks
    print("NaNs in X_train_scaled:", int(np.isnan(X_train_scaled.to_numpy()).sum()))
    print("NaNs in y_train OS_time:", int(pd.isna(y_train["OS_time"]).sum()))

    # CV setup (stratify folds by cancer + event in TRAIN)
    train_strata = (y_train["cancer"].astype(str) + "_e" + y_train["OS_event"].astype(str)).values
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    # CV grid search
    cv_rows = []
    best = None  # (mean_cindex, penalizer, l1_ratio)
    X_tr_np = X_train_scaled.to_numpy(dtype=np.float32)
    y_tr_df = y_train.copy()

    # Suppress noisy convergence warnings in CV loop (we still record failures)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    for l1 in L1_RATIOS:
        for pen in PENALIZERS:
            fold_scores = []
            failures = 0
            selected_counts = []
            print(f"Trying penalizer={pen} l1_ratio={l1}", flush=True)

            for fold_i, (idx_tr, idx_va) in enumerate(skf.split(X_tr_np, train_strata), start=1):
                print(f"  Fold {fold_i}/{N_SPLITS}", flush=True)

                X_tr = X_train_scaled.iloc[idx_tr].copy()
                X_va = X_train_scaled.iloc[idx_va].copy()

                # Build Cox dataframe: covariates + duration/event
                df_tr = X_tr.copy()
                df_tr["OS_time"] = y_tr_df.iloc[idx_tr]["OS_time"].values
                df_tr["OS_event"] = y_tr_df.iloc[idx_tr]["OS_event"].values

                df_va = X_va.copy()
                df_va["OS_time"] = y_tr_df.iloc[idx_va]["OS_time"].values
                df_va["OS_event"] = y_tr_df.iloc[idx_va]["OS_event"].values

                # Safety: lifelines requires no NaNs
                df_tr = df_tr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                df_va = df_va.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                try:
                    cph = CoxPHFitter(penalizer=float(pen), l1_ratio=float(l1))
                    cph.fit(df_tr, duration_col="OS_time", event_col="OS_event", show_progress=False)

                    # Evaluate on validation via concordance (higher risk -> shorter survival)
                    va_risk = cph.predict_partial_hazard(df_va).values.reshape(-1)
                    cidx = concordance_index(df_va["OS_time"].values, -va_risk, df_va["OS_event"].values)
                    fold_scores.append(float(cidx))

                    # How many features got non-zero-ish coefficients?
                    coefs = cph.params_.copy()
                    selected_counts.append(int((np.abs(coefs.values) > 1e-10).sum()))
                    print(f"    fold c-index={cidx:.3f}", flush=True)

                except Exception as e:
                    failures += 1
                    print(f"[CV failure] penalizer={pen} l1_ratio={l1} fold error: {type(e).__name__}: {e}", flush=True)

            mean_c = float(np.mean(fold_scores)) if fold_scores else np.nan
            std_c = float(np.std(fold_scores)) if fold_scores else np.nan
            nsel = float(np.mean(selected_counts)) if selected_counts else np.nan

            cv_rows.append(
                {
                    "penalizer": float(pen),
                    "l1_ratio": float(l1),
                    "cv_cindex_mean": mean_c,
                    "cv_cindex_std": std_c,
                    "cv_failures": int(failures),
                    "n_selected_mean": nsel,
                }
            )

            if fold_scores and failures == 0:
                if best is None or mean_c > best[0]:
                    best = (mean_c, float(pen), float(l1))

    cv_df = pd.DataFrame(cv_rows).sort_values(["cv_cindex_mean"], ascending=False)
    cv_df.to_csv(OUT_CV, index=False)

    if best is None:
        raise RuntimeError(
            "All CV settings failed. Reduce TOPN_VARIABLE_GENES and/or increase penalizer grid."
        )

    best_mean, best_pen, best_l1 = best
    print(f"\nBest: mean_cv_cindex={best_mean:.3f} penalizer={best_pen} l1_ratio={best_l1}", flush=True)

    # Fit final model on full TRAIN
    train_df = X_train_scaled.copy()
    train_df["OS_time"] = y_train["OS_time"].values
    train_df["OS_event"] = y_train["OS_event"].values
    train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    test_df = X_test_scaled.copy()
    test_df["OS_time"] = y_test["OS_time"].values
    test_df["OS_event"] = y_test["OS_event"].values
    test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    final_cph = CoxPHFitter(penalizer=float(best_pen), l1_ratio=float(best_l1))
    final_cph.fit(train_df, duration_col="OS_time", event_col="OS_event", show_progress=False)

    # Train/test C-index
    train_risk = final_cph.predict_partial_hazard(train_df).values.reshape(-1)
    test_risk = final_cph.predict_partial_hazard(test_df).values.reshape(-1)

    c_train = concordance_index(train_df["OS_time"].values, -train_risk, train_df["OS_event"].values)
    c_test = concordance_index(test_df["OS_time"].values, -test_risk, test_df["OS_event"].values)

    print(f"ElasticNet Cox C-index train: {c_train:.3f}")
    print(f"ElasticNet Cox C-index test : {c_test:.3f}")

    # Save selected features (coefficients)
    coef = final_cph.params_.copy()
    selected = coef[np.abs(coef.values) > 1e-10].sort_values(ascending=False)
    selected_df = pd.DataFrame({"feature": selected.index, "coef": selected.values})
    selected_df.to_csv(OUT_SELECTED, index=False)

    # Save summary
    summary = final_cph.summary.reset_index().rename(columns={"index": "covariate"})
    summary.to_csv(OUT_SUMMARY, index=False)

    # KM plot on TEST (median split based on TRAIN median risk)
    train_med = float(np.median(train_risk))
    test_group_risk = pd.Series(test_risk, index=test_df.index, name="risk")
    # shift by train median so split consistent
    km_plot_high_low(
        time=test_df["OS_time"],
        event=test_df["OS_event"],
        risk=test_group_risk - train_med,
        out_png=OUT_FIG,
        title="Elastic-net Cox risk groups (test set, median split from train)",
    )

    print("\nWrote:")
    print(f" - {OUT_CV}")
    print(f" - {OUT_SELECTED}")
    print(f" - {OUT_SUMMARY}")
    print(f" - {OUT_FIG}")


if __name__ == "__main__":
    main()
