#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
EXPR_FILE = Path("data/processed/expression_tcga_primary_6cancers.tsv.gz")
SURV_FILE = Path("results/tables/survival_clinical_6cancers.tsv")

OUT_SUMMARY = Path("results/tables/pytorch_survival_summary.tsv")
OUT_KM = Path("figures/km_pytorch_risk_strat_test.png")

SEED = 42
TEST_FRAC = 0.25
TOPN_GENES = 2000
USE_AGE = True
EPOCHS = 100
LR = 1e-3

torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------
# Load + prepare data
# -----------------------------
print("Loading data...")
expr = pd.read_csv(EXPR_FILE, sep="\t", compression="gzip", index_col=0).T
surv = pd.read_csv(SURV_FILE, sep="\t").set_index("sample")

common = expr.index.intersection(surv.index)
expr = expr.loc[common]
surv = surv.loc[common]

if USE_AGE:
    expr["age"] = surv["age"]

# -----------------------------
# Feature filtering
# -----------------------------
print("Selecting top variable genes...")
vars_ = expr.var(axis=0, ddof=0)
top = vars_.sort_values(ascending=False).head(TOPN_GENES).index
X = expr[top].copy()

T = surv["OS_time"].astype(float)
E = surv["OS_event"].astype(int)

valid = X.notnull().all(axis=1) & T.notnull() & E.notnull()
X = X.loc[valid]
T = T.loc[valid]
E = E.loc[valid]

# -----------------------------
# Split and scale
# -----------------------------
strata = surv.loc[valid, "cancer"] + "_E" + E.astype(str)
X_train, X_test, T_train, T_test, E_train, E_test = train_test_split(
    X, T, E, test_size=TEST_FRAC, stratify=strata.loc[valid], random_state=SEED
)

scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
X_test_s = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

Xtr_tensor = torch.tensor(X_train_s.values, dtype=torch.float32)
Xte_tensor = torch.tensor(X_test_s.values, dtype=torch.float32)
Ttr_tensor = torch.tensor(T_train.values, dtype=torch.float32)
Tte_tensor = torch.tensor(T_test.values, dtype=torch.float32)
Etr_tensor = torch.tensor(E_train.values, dtype=torch.float32)
Ete_tensor = torch.tensor(E_test.values, dtype=torch.float32)

# -----------------------------
# Cox loss
# -----------------------------
def cox_ph_loss(risk_scores, times, events):
    order = torch.argsort(times, descending=True)
    times = times[order]
    events = events[order]
    risk_scores = risk_scores[order]

    hazard = risk_scores
    log_cumsum_hazard = torch.logcumsumexp(hazard, dim=0)
    uncensored_likelihood = hazard - log_cumsum_hazard
    neg_log_likelihood = -torch.sum(uncensored_likelihood * events)
    return neg_log_likelihood / events.sum()

# -----------------------------
# Model
# -----------------------------
class CoxMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

model = CoxMLP(Xtr_tensor.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Training loop
# -----------------------------
print("Training...")
model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    out = model(Xtr_tensor)
    loss = cox_ph_loss(out, Ttr_tensor, Etr_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")

# -----------------------------
# Evaluation
# -----------------------------
model.eval()
with torch.no_grad():
    risk_train = model(Xtr_tensor).numpy()
    risk_test = model(Xte_tensor).numpy()

cidx_train = concordance_index(T_train, -risk_train, E_train)
cidx_test = concordance_index(T_test, -risk_test, E_test)

print(f"Train C-index: {cidx_train:.3f}")
print(f"Test  C-index: {cidx_test:.3f}")

# -----------------------------
# KM stratification (test set)
# -----------------------------
risk_df = pd.Series(risk_test, index=X_test.index, name="risk")
cutoff = risk_df.median()
grp = (risk_df > cutoff).map({True: "high_risk", False: "low_risk"})

from lifelines import KaplanMeierFitter
plt.figure(figsize=(6,5))
kmf = KaplanMeierFitter()
for label in ["low_risk", "high_risk"]:
    idx = grp == label
    kmf.fit(T_test.loc[idx], E_test.loc[idx], label=label)
    kmf.plot(ci_show=False, linewidth=2)
plt.title("KM Stratification (PyTorch Cox MLP)")
plt.xlabel("Days")
plt.ylabel("Survival probability")
plt.tight_layout()
OUT_KM.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_KM, dpi=200)
plt.close()
print("Saved:", OUT_KM)

# -----------------------------
# Save summary
# -----------------------------
OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame([{
    "model": "pytorch_cox_mlp",
    "top_variable_genes": TOPN_GENES,
    "use_age": USE_AGE,
    "train_c_index": round(cidx_train, 4),
    "test_c_index": round(cidx_test, 4),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "events_train": int(E_train.sum()),
    "events_test": int(E_test.sum()),
}]).to_csv(OUT_SUMMARY, sep="\t", index=False)
print("Saved:", OUT_SUMMARY)
